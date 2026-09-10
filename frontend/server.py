#!/usr/bin/env python3
"""Local demo server for the MPC verified LLM intents frontend.

Serves the chat UI on 127.0.0.1:8787 and exposes two endpoints:

  POST /api/intent  body {"prompt": str} -> NDJSON stream of pipeline stages
  GET  /api/state   live balances, block height, cluster state (for polling)

Stdlib only. Signs with the localnet validator key (plaintext, demo only).
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")

RPC_URL = os.environ.get("NEAR_RPC_URL", "http://127.0.0.1:3030")
MLX_URL = os.environ.get("MLX_URL", "http://127.0.0.1:8080")
NEAR_CLI = os.environ.get("NEAR_CLI", os.path.expanduser("~/.cargo/bin/near"))
VALIDATOR_KEY_FILE = os.path.expanduser("~/.near/mpc-localnet/validator_key.json")

NETWORK = "mpc-localnet"
CALLER = "test.near"  # user A
WALLET = "ai-wallet.test.near"
DEFAULT_RECIPIENT = "alice.test.near"  # user B
MPC_CONTRACT = "mpc-contract.test.near"

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
LLM_DOMAIN_ID = 4
SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "action": {"enum": ["transfer"]},
            "to": {"type": "string"},
            "amount": {"type": "integer"},
        },
        "required": ["action", "to", "amount"],
        "additionalProperties": False,
    }
)
YOCTO_PER_NEAR = 10**24

NODE_METRIC_PORTS = [3031, 3032, 3033, 3034]

with open(VALIDATOR_KEY_FILE) as f:
    CALLER_SECRET_KEY = json.load(f)["secret_key"]


def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": "1", "method": method, "params": params}).encode()
    req = urllib.request.Request(RPC_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["result"]


def view_account(account_id):
    return rpc(
        "query",
        {"request_type": "view_account", "finality": "final", "account_id": account_id},
    )


def balance_near(account_id):
    try:
        return int(view_account(account_id)["amount"]) / YOCTO_PER_NEAR
    except Exception:
        return None


def block_height():
    try:
        return int(rpc("status", [])["sync_info"]["latest_block_height"])
    except Exception:
        return None


def unwrap_call_result(result):
    value = result["result"]
    if isinstance(value, list):
        return json.loads(bytes(value).decode())
    return value


def view_state():
    return unwrap_call_result(
        rpc(
            "query",
            {
                "request_type": "call_function",
                "finality": "final",
                "account_id": MPC_CONTRACT,
                "method_name": "state",
                "args_base64": base64.b64encode(b"{}").decode(),
            },
        )
    )


def cluster_state():
    try:
        state = view_state()
        return "Running" if "Running" in state else next(iter(state))
    except Exception:
        return "Unknown"


def public_key_domain4():
    return unwrap_call_result(
        rpc(
            "query",
            {
                "request_type": "call_function",
                "finality": "final",
                "account_id": MPC_CONTRACT,
                "method_name": "public_key",
                "args_base64": base64.b64encode(json.dumps({"domain_id": LLM_DOMAIN_ID}).encode()).decode(),
            },
        )
    )


DOMAIN4_PUBLIC_KEY = None
CLUSTER_CACHE = {"state": None, "at": 0.0}
CACHE_LOCK = threading.Lock()


def cached_cluster_state():
    global CLUSTER_CACHE
    with CACHE_LOCK:
        now = time.time()
        if CLUSTER_CACHE["state"] is None or now - CLUSTER_CACHE["at"] > 5:
            CLUSTER_CACHE["state"] = cluster_state()
            CLUSTER_CACHE["at"] = now
        return CLUSTER_CACHE["state"]


def node_llm_response_counts():
    counts = {}
    for port in NODE_METRIC_PORTS:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as resp:
                text = resp.read().decode()
            match = re.search(r"^mpc_num_llm_inference_responses_indexed (\d+)$", text, re.M)
            counts[port] = int(match.group(1)) if match else 0
        except Exception:
            counts[port] = None
    return counts


def run_near_cli(args):
    cmd = [NEAR_CLI] + args + [
        "network-config", NETWORK,
        "sign-with-plaintext-private-key", CALLER_SECRET_KEY, "send",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = proc.stdout + proc.stderr
    match = re.search(r"Transaction ID: ([A-Za-z0-9]+)", out)
    if not match:
        raise RuntimeError(f"near CLI did not return a transaction hash: {out[-800:]}")
    return match.group(1)


def request_llm_inference(prompt):
    args_json = json.dumps(
        {
            "request": {
                "domain_id": LLM_DOMAIN_ID,
                "model_id": MODEL,
                "prompt": prompt,
                "schema": SCHEMA,
            }
        }
    )
    return run_near_cli([
        "contract", "call-function", "as-transaction", MPC_CONTRACT,
        "request_llm_inference", "json-args", args_json,
        "prepaid-gas", "100.0 Tgas", "attached-deposit", "1 yoctoNEAR",
        "sign-as", CALLER,
    ])


def wallet_execute(args):
    return run_near_cli([
        "contract", "call-function", "as-transaction", WALLET,
        "execute", "file-args", args,
        "prepaid-gas", "120.0 Tgas", "attached-deposit", "0 NEAR",
        "sign-as", CALLER,
    ])


def tx_success_value(tx_hash, signer):
    result = rpc("tx", [tx_hash, signer])
    status = result.get("status", {})
    if "SuccessValue" in status:
        return base64.b64decode(status["SuccessValue"]), None
    if "Failure" in status:
        return None, json.dumps(status["Failure"])
    return None, "transaction still pending"


def wait_for_success(tx_hash, signer, timeout=60):
    deadline = time.time() + timeout
    last_error = "no status"
    while time.time() < deadline:
        value, err = tx_success_value(tx_hash, signer)
        if value is not None:
            return value, None
        last_error = err
        time.sleep(1)
    return None, last_error


def deterministic_output(prompt):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Extract the wallet intent as JSON matching the given schema. Output only the JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "intent", "schema": json.loads(SCHEMA)},
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{MLX_URL}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        parsed = json.load(resp)
    return parsed["choices"][0]["message"]["content"]


def borsh_string(s):
    b = s.encode()
    return len(b).to_bytes(4, "little") + b


def compute_payload_hash(prompt, output):
    payload = (
        LLM_DOMAIN_ID.to_bytes(8, "little")
        + borsh_string(MODEL)
        + borsh_string(prompt)
        + borsh_string(SCHEMA)
        + borsh_string(MODEL)
        + borsh_string(output)
    )
    return hashlib.sha256(payload).hexdigest()


def intent_prompt(user_text):
    return (
        "You are a payment assistant for an AI wallet on NEAR. The user says: "
        f"{user_text}. Reply with ONLY a JSON object matching the schema. "
        "action is always transfer. to is the recipient account id. amount is the "
        "integer amount in whole NEAR tokens (for 'send 1 NEAR' the amount is 1, "
        "for 'send 0.5 NEAR' the amount is 0)."
    )


class IntentPipeline:
    def __init__(self, user_text, send):
        self.send = send
        self.user_text = user_text
        self.start = time.time()

    def emit(self, stage, **fields):
        self.send(json.dumps({"stage": stage, "elapsed_ms": int((time.time() - self.start) * 1000), **fields}))

    def run(self):
        prompt = intent_prompt(self.user_text)

        self.emit("submitted", label="Submitting inference request to the MPC contract")
        tx_hash = request_llm_inference(prompt)
        self.emit("submitted", label="Request on chain", tx_hash=tx_hash, explorer=f"{RPC_URL}/{tx_hash}")

        self.emit("inferring", label="MPC nodes are running the prompt", nodes={})
        seen = {}
        while True:
            value, err = tx_success_value(tx_hash, CALLER)
            counts = node_llm_response_counts()
            for port, count in counts.items():
                if count and seen.get(port) != count:
                    seen[port] = count
                    self.emit("inferring", label=f"node {port - 3030} observed a response", nodes={p: c for p, c in counts.items() if c})
            if value is not None:
                break
            if err is not None and err != "transaction still pending":
                self.emit("error", label="inference request failed", message=err)
                return
            time.sleep(0.5)

        total = sum(1 for _ in seen)
        self.emit("inferring", label=f"{total}/{len(NODE_METRIC_PORTS)} nodes agree", nodes=seen)

        response = json.loads(value)
        payload_hash = response["payload_hash"]
        signature = response["signature"]
        self.emit(
            "signed",
            label="Threshold signature produced",
            payload_hash=payload_hash,
            signature=signature,
        )

        self.emit("output", label="Reconstructing the signed output")
        output = deterministic_output(prompt)
        recomputed = compute_payload_hash(prompt, output)
        if recomputed != payload_hash:
            self.emit(
                "error",
                label="Output reconstruction mismatch",
                message=f"local hash {recomputed} != signed hash {payload_hash}",
            )
            return
        self.emit(
            "output",
            label="Output matches the signature",
            output=output,
            output_text=output,
        )

        self.emit("verified", label="Wallet verifying signature and policy")
        recipient = json.loads(output).get("to", DEFAULT_RECIPIENT)
        balances_before = {CALLER: balance_near(WALLET), recipient: balance_near(recipient)}
        execute_args_file = "/tmp/wallet-execute-args.json"
        with open(execute_args_file, "w") as f:
            json.dump(
                {
                    "output": output,
                    "response": response,
                    "model_id": MODEL,
                    "prompt": prompt,
                    "schema": SCHEMA,
                },
                f,
            )
        try:
            exec_hash = wallet_execute(execute_args_file)
        except RuntimeError as e:
            message = str(e)
            reason = "rejected by the wallet contract"
            match = re.search(r"(amount \d+ NEAR exceeds the per transfer cap of \d+ NEAR|action \S+ is not whitelisted[^\\\\\"]*)", message)
            if match:
                reason = "policy: " + match.group(1)
            elif "transfer failed" in message:
                reason = f"transfer to {recipient} failed (recipient account may not exist)"
            self.emit("rejected", label=reason, message=message[-500:])
            return
        self.emit(
            "verified",
            label="Policy passed, executing transfer",
            tx_hash=exec_hash,
            recipient=recipient,
        )

        exec_value, err = wait_for_success(exec_hash, CALLER)
        if exec_value is None:
            self.emit("error", label="wallet execution failed", message=err)
            return
        balances_after = {CALLER: balance_near(WALLET), recipient: balance_near(recipient)}
        self.emit(
            "executed",
            label="Transfer executed",
            result=exec_value.decode(),
            recipient=recipient,
            balances_before=balances_before,
            balances_after=balances_after,
            tx_hash=exec_hash,
        )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/state":
            self._send_json(
                {
                    "caller": CALLER,
                    "wallet": WALLET,
                    "wallet_balance": balance_near(WALLET),
                    "recipient": DEFAULT_RECIPIENT,
                    "recipient_balance": balance_near(DEFAULT_RECIPIENT),
                    "height": block_height(),
                    "cluster": cached_cluster_state(),
                    "domain_key": DOMAIN4_PUBLIC_KEY,
                }
            )
        elif self.path == "/" or self.path == "/index.html":
            self._serve_static("index.html", "text/html; charset=utf-8")
        else:
            self._serve_static(self.path.lstrip("/"), None)

    def _serve_static(self, name, content_type):
        path = os.path.normpath(os.path.join(STATIC, name))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            self.send_error(404)
            return
        if content_type is None:
            guess = {
                ".js": "text/javascript",
                ".css": "text/css",
                ".html": "text/html; charset=utf-8",
                ".svg": "image/svg+xml",
            }
            content_type = guess.get(os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/intent":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        user_text = body.get("prompt", "").strip()
        if not user_text:
            self._send_json({"error": "prompt is required"}, status=400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        done = threading.Event()

        def send(line):
            try:
                self.wfile.write((line + "\n").encode())
                self.wfile.flush()
            except Exception:
                done.set()

        try:
            IntentPipeline(user_text, send).run()
        except Exception as e:
            send(json.dumps({"stage": "error", "label": "pipeline error", "message": str(e)[:500]}))
        done.set()


def main():
    global DOMAIN4_PUBLIC_KEY
    try:
        DOMAIN4_PUBLIC_KEY = public_key_domain4()
    except Exception:
        DOMAIN4_PUBLIC_KEY = None
    server = ThreadingHTTPServer(("127.0.0.1", 8787), Handler)
    print("frontend server on http://127.0.0.1:8787")
    server.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# End to end demo: natural language in, MPC-verified transfer out.
#
#   "send 1 NEAR to alice.test.near"
#     -> request_llm_inference on the MPC contract (Llm domain, id 4)
#     -> every node runs the prompt through the pinned Qwen model
#     -> outputs agree, network signs SHA256(borsh(LlmSignPayload))
#     -> the caller's transaction returns the signed LlmInferenceResponse
#     -> ai wallet verifies the signature, checks whitelist + amount cap, transfers
#
# Prerequisites:
#   - localnet launched via scripts/launch-localnet.sh (contract Running, all 5
#     domains added including the Llm domain)
#   - ai-wallet-contract deployed to ai-wallet.test.near (holds transfer funds)
#   - mlx_lm.server running on 127.0.0.1:8080 with the configured model
#   - near CLI configured for the mpc-localnet network
#
# Usage:
#   ./demo-llm-intent.sh "send 1 NEAR to alice.test.near"

set -euo pipefail

PROMPT="${1:?usage: demo-llm-intent.sh \"send 1 NEAR to alice.test.near\"}"
MPC_CONTRACT="mpc-contract.test.near"
WALLET_CONTRACT="ai-wallet.test.near"
CALLER="test.near"
NETWORK="mpc-localnet"
MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SCHEMA='{"type":"object","properties":{"action":{"enum":["transfer"]},"to":{"type":"string"},"amount":{"type":"integer"}},"required":["action","to","amount"],"additionalProperties":false}'
INFERENCE_SYSTEM_PROMPT="You are a payment assistant for an AI wallet on NEAR. The user says: ${PROMPT}. Reply with ONLY a JSON object matching the schema. action is always transfer. to is the recipient account id. amount is the integer amount in yoctoNEAR where 1 NEAR = 1000000000000000000000000."

RPC="${NEAR_RPC_URL:-http://127.0.0.1:3030}"
tx_status() {
  curl -s "$RPC" -X POST -H "Content-Type: application/json" \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tx\",\"params\":[\"$1\",\"$2\"]}"
}

echo "==> 1. Submitting the prompt to the MPC contract"
echo "    prompt: ${PROMPT}"

ARGS=$(jq -nc --arg m "${MODEL}" --arg p "${INFERENCE_SYSTEM_PROMPT}" --arg s "${SCHEMA}" \
  '{request:{domain_id:4,model_id:$m,prompt:$p,schema:$s}}')

TX_OUT=$(near contract call-function as-transaction "${MPC_CONTRACT}" \
  request_llm_inference json-args "${ARGS}" \
  prepaid-gas '100.0 Tgas' attached-deposit '1 yoctoNEAR' \
  sign-as "${CALLER}" network-config "${NETWORK}" sign-with-keychain send 2>&1)
TX_HASH=$(echo "${TX_OUT}" | grep -oE '[A-Za-z0-9]{43}' | tail -1)
echo "    tx: ${TX_HASH}"

echo "==> 2. Waiting for the MPC network to infer, agree and sign"
RESPONSE_B64=""
for i in $(seq 1 60); do
  RESPONSE_B64=$(tx_status "${TX_HASH}" "${CALLER}" | jq -r '.result.status.SuccessValue // empty')
  if [ -n "${RESPONSE_B64}" ]; then
    echo "    signed response returned after ${i} polls"
    break
  fi
  sleep 1
done
if [ -z "${RESPONSE_B64}" ]; then
  echo "    no signed response after 60s" >&2
  exit 1
fi

echo "==> 3. Extracting the signed output"
echo "${RESPONSE_B64}" | base64 -d > /tmp/llm-response.json
OUTPUT=$(jq -r '.output // empty' /tmp/llm-response.json 2>/dev/null || true)
if [ -z "${OUTPUT}" ]; then
  # The response is the bare LlmInferenceResponse; the output is recovered by
  # re-running the same deterministic inference locally.
  echo "    reconstructing the output via local deterministic inference"
  OUTPUT=$(curl -s "http://127.0.0.1:8080/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg m "${MODEL}" --arg p "${INFERENCE_SYSTEM_PROMPT}" --arg s "${SCHEMA}" \
      '{model:$m,messages:[{role:"system",content:"Extract the wallet intent as JSON matching the given schema. Output only the JSON object."},{role:"user",content:$p}],temperature:0.0,response_format:{type:"json_schema",json_schema:{name:"intent",schema:($s|fromjson)}}}')" \
    | jq -r '.choices[0].message.content')
fi
echo "    output: ${OUTPUT}"

echo "==> 4. Executing the intent through the AI wallet"
python3 - "$OUTPUT" "${MODEL}" "${INFERENCE_SYSTEM_PROMPT}" "${SCHEMA}" /tmp/llm-response.json <<'EOF' > /tmp/wallet-execute-args.json
import hashlib, json, sys

def borsh_string(s):
    b = s.encode()
    return len(b).to_bytes(4, "little") + b

output, model_id, prompt, schema, response_path = sys.argv[1:6]
response = json.load(open(response_path))

payload_hash = hashlib.sha256(
    (4).to_bytes(8, "little")
    + borsh_string(model_id)
    + borsh_string(prompt)
    + borsh_string(schema)
    + borsh_string(model_id)
    + borsh_string(output)
).hexdigest()

json.dump(
    {
        "output": output,
        "response": response,
        "model_id": model_id,
        "prompt": prompt,
        "schema": schema,
    },
    sys.stdout,
)
EOF

EXEC_TX=$(near contract call-function as-transaction "${WALLET_CONTRACT}" \
  execute file-args /tmp/wallet-execute-args.json \
  prepaid-gas '120.0 Tgas' attached-deposit '0 NEAR' \
  sign-as "${CALLER}" network-config "${NETWORK}" sign-with-keychain send 2>&1)
EXEC_HASH=$(echo "${EXEC_TX}" | grep -oE '[A-Za-z0-9]{43}' | tail -1)
echo "    tx: ${EXEC_HASH}"

RESULT=""
for i in $(seq 1 30); do
  RESULT=$(tx_status "${EXEC_HASH}" "${CALLER}" | jq -r '.result.status.SuccessValue // empty')
  [ -n "${RESULT}" ] && break
  sleep 1
done
echo "    wallet result: ${RESULT:-<pending>}"
echo "==> done"

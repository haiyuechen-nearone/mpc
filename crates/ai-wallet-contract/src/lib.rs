//! Demo AI wallet: executes NEAR transfers authorized by an MPC-verified LLM
//! intent.
//!
//! The MPC network signs `SHA256(borsh(LlmSignPayload))` only when every node
//! produced byte-identical output for the prompt. This contract trusts that
//! signature, so `execute` checks it against the `Llm` domain root key of the
//! MPC contract, then parses the signed output JSON and enforces the two
//! policies the schema cannot express: the action whitelist and the amount
//! cap. The schema constrains shape, not magnitudes.

use near_mpc_contract_interface::types::{
    DomainId, LlmInferenceResponse, LlmSignPayload, PublicKey, SignatureResponse,
};
use near_sdk::{
    AccountId, Gas, NearToken, Promise, PromiseError, PromiseOrValue, env, log, near, serde_json,
};

/// Account the MPC network runs at on this localnet.
const MPC_CONTRACT_ID: &str = "mpc-contract.test.near";

/// Domain id of the Llm purpose domain in the localnet domain registry.
const LLM_DOMAIN_ID: DomainId = DomainId(4);

/// Upper bound for any transfer this wallet executes. The intent schema
/// cannot express this, so it is policed here.
const MAX_TRANSFER_AMOUNT_YOCTONEAR: u128 = 1_000_000_000_000_000_000_000_000; // 1 NEAR

/// The wallet has no state; the MPC contract id and domain id are compile
/// time constants for the demo localnet.
#[near(contract_state)]
#[derive(Debug, Default)]
pub struct AiWalletContract;

#[derive(Debug, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct Intent {
    pub action: String,
    pub to: AccountId,
    pub amount: u128,
}

#[near]
impl AiWalletContract {
    /// Executes the transfer described by the MPC-signed LLM output.
    ///
    /// Verifies the threshold signature against the Llm domain root key,
    /// parses the signed output, and enforces the whitelist and amount cap.
    pub fn execute(
        &self,
        output: String,
        response: LlmInferenceResponse,
        model_id: String,
        prompt: String,
        schema: String,
    ) -> PromiseOrValue<String> {
        env::log_str(&format!(
            "execute: predecessor={:?}, output={output}",
            env::predecessor_account_id(),
        ));

        let SignatureResponse::Secp256k1(_) = &response.signature else {
            env::panic_str("signature scheme mismatch: expected secp256k1");
        };

        // Rebuild the exact payload the MPC network signed, from the
        // caller-supplied request fields, and check it hashes to what the
        // signature covers.
        let payload = LlmSignPayload {
            request: near_mpc_contract_interface::types::LlmInferenceRequest {
                domain_id: LLM_DOMAIN_ID,
                model_id: model_id.clone(),
                prompt,
                schema,
            },
            model_id,
            output: output.clone(),
        };
        let payload_hash = payload
            .compute_msg_hash()
            .expect("borsh serialization of a plain struct cannot fail");
        if payload_hash != response.payload_hash {
            env::panic_str("response payload hash does not match the provided output");
        }

        let intent: Intent = serde_json::from_str(&output).unwrap_or_else(|err| {
            env::panic_str(&format!("signed output is not valid JSON: {err}"))
        });
        Self::check_policy(&intent);

        let json_args = serde_json::json!({ "domain_id": LLM_DOMAIN_ID });
        let mpc: AccountId = MPC_CONTRACT_ID
            .parse()
            .expect("hardcoded contract id is valid");
        Promise::new(mpc)
            .function_call(
                "public_key".to_string(),
                serde_json::to_vec(&json_args).expect("json args of a view call serialize"),
                NearToken::from_near(0),
                Gas::from_tgas(10),
            )
            .then(
                Self::ext(env::current_account_id())
                    .with_static_gas(Gas::from_tgas(40))
                    .execute_with_key(response, intent),
            )
            .into()
    }

    #[private]
    pub fn execute_with_key(
        #[callback_result] public_key: Result<PublicKey, PromiseError>,
        response: LlmInferenceResponse,
        intent: Intent,
    ) -> PromiseOrValue<String> {
        let public_key = match public_key {
            Ok(PublicKey::Secp256k1(key)) => key,
            Ok(_) => env::panic_str("Llm domain key is not a secp256k1 key"),
            Err(err) => env::panic_str(&format!("failed to fetch the Llm domain key: {err:?}")),
        };

        let SignatureResponse::Secp256k1(signature) = &response.signature else {
            env::panic_str("signature scheme mismatch: expected secp256k1");
        };
        let payload_hash: [u8; 32] = response.payload_hash.0;
        near_mpc_signature_verifier::verify_ecdsa_signature(signature, &payload_hash, &public_key)
            .unwrap_or_else(|err| {
                env::panic_str(&format!("signature verification failed: {err:?}"))
            });

        log!(
            "intent verified: {} {} yoctoNEAR",
            intent.action,
            intent.amount
        );
        Promise::new(intent.to)
            .transfer(NearToken::from_yoctonear(intent.amount))
            .then(
                Self::ext(env::current_account_id())
                    .with_static_gas(Gas::from_tgas(10))
                    .execute_finished(),
            )
            .into()
    }

    #[private]
    pub fn execute_finished(#[callback_result] result: Result<(), PromiseError>) -> String {
        match result {
            Ok(()) => "transfer executed".to_string(),
            Err(err) => env::panic_str(&format!("transfer failed: {err:?}")),
        }
    }

    /// The closed action list is the real security boundary: a prompt
    /// injection can only pick actions this contract understands, and the
    /// amount cap bounds the blast radius of any single intent.
    fn check_policy(intent: &Intent) {
        if intent.action != "transfer" {
            env::panic_str(&format!(
                "action {} is not whitelisted; this wallet only executes transfer",
                intent.action
            ));
        }
        if intent.amount > MAX_TRANSFER_AMOUNT_YOCTONEAR {
            env::panic_str(&format!(
                "amount {} exceeds the per transfer cap of {MAX_TRANSFER_AMOUNT_YOCTONEAR}",
                intent.amount
            ));
        }
    }
}

/// External interface for the MPC public key view call.
#[cfg(all(test, not(target_arch = "wasm32")))]
#[expect(non_snake_case)]
mod tests {
    use super::*;
    use k256::ecdsa::SigningKey;
    use k256::elliptic_curve::PrimeField;
    use near_mpc_contract_interface::types::K256Signature;
    use near_sdk::test_utils::VMContextBuilder;
    use near_sdk::testing_env;

    fn signed_response(payload: &LlmSignPayload, key: &SigningKey) -> LlmInferenceResponse {
        let payload_hash = payload.compute_msg_hash().unwrap();
        let (signature, recovery_id) = key.sign_prehash_recoverable(&payload_hash.0).unwrap();
        LlmInferenceResponse {
            payload_hash,
            signature: SignatureResponse::Secp256k1(K256Signature::from_ecdsa_recoverable(
                &signature,
                recovery_id,
            )),
        }
    }

    fn test_payload() -> LlmSignPayload {
        LlmSignPayload {
            request: near_mpc_contract_interface::types::LlmInferenceRequest {
                domain_id: LLM_DOMAIN_ID,
                model_id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit".to_string(),
                prompt: "send 1 NEAR to alice.near".to_string(),
                schema: "{}".to_string(),
            },
            model_id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit".to_string(),
            output: r#"{"action":"transfer","to":"bob.near","amount":500}"#.to_string(),
        }
    }

    #[test]
    fn check_policy__should_accept_transfer_within_cap() {
        // Given
        let intent = Intent {
            action: "transfer".to_string(),
            to: "bob.near".parse().unwrap(),
            amount: MAX_TRANSFER_AMOUNT_YOCTONEAR,
        };

        // When
        AiWalletContract::check_policy(&intent);

        // Then: no panic
    }

    #[test]
    #[should_panic(expected = "exceeds the per transfer cap")]
    fn check_policy__should_reject_amount_over_cap() {
        // Given: the injection probe magnitude, far beyond any real balance.
        let intent = Intent {
            action: "transfer".to_string(),
            to: "evil.near".parse().unwrap(),
            amount: u128::MAX,
        };

        // When
        AiWalletContract::check_policy(&intent);
    }

    #[test]
    #[should_panic(expected = "is not whitelisted")]
    fn check_policy__should_reject_action_outside_the_whitelist() {
        // Given
        let intent = Intent {
            action: "drain".to_string(),
            to: "evil.near".parse().unwrap(),
            amount: 1,
        };

        // When
        AiWalletContract::check_policy(&intent);
    }

    #[test]
    #[should_panic(expected = "not valid JSON")]
    fn execute__should_reject_output_that_is_not_json_before_any_promise() {
        // Given
        let context = VMContextBuilder::new()
            .attached_deposit(NearToken::from_yoctonear(1))
            .build();
        testing_env!(context);
        let payload = test_payload();
        let tampered = LlmSignPayload {
            output: "I cannot do that".to_string(),
            ..payload.clone()
        };

        // When: the payload hash matches the tampered output, so the JSON
        // parse is what must fail.
        let response = signed_response(
            &tampered,
            &SigningKey::from_bytes(&k256::Scalar::from_u128(1).to_bytes()).unwrap(),
        );

        // Then
        let _ = AiWalletContract::execute(
            &AiWalletContract,
            tampered.output,
            response,
            tampered.model_id,
            tampered.request.prompt,
            tampered.request.schema,
        );
    }
}

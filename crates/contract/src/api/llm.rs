//! LLM inference: the request path, its response callback and the
//! pending-request view. Clones the foreign-tx yield/resume flow; the schema
//! is stored and hashed opaquely and validated node-side.

use crate::crypto_shared::types::PublicKeyExtended;
use crate::dto_mapping::args_into_llm_inference_request;
use crate::errors::{Error, InvalidState, RespondError, TeeError};
use crate::{MpcContract, MpcContractExt, pending_requests};
use near_mpc_contract_interface::deposits::SIGN_DEPOSIT_YOCTONEAR;
use near_mpc_contract_interface::method_names;
use near_mpc_contract_interface::types as dtos;
use near_sdk::{CryptoHash, Gas, NearToken, Promise, PromiseError, PromiseOrValue, env, log, near};

#[near]
impl MpcContract {
    fn add_llm_inference_request(
        &mut self,
        request: dtos::LlmInferenceRequest,
        data_id: CryptoHash,
    ) {
        pending_requests::push_pending_yield(
            &mut self.pending_llm_inference_requests,
            request,
            data_id,
        );
    }

    /// Submit an LLM inference request. MPC nodes run the prompt through the
    /// pinned model independently and sign only on byte-identical output.
    #[handle_result]
    #[payable]
    pub fn request_llm_inference(&mut self, request: dtos::LlmInferenceRequestArgs) {
        log!(
            "request_llm_inference: predecessor={:?}, request={:?}",
            env::predecessor_account_id(),
            request
        );

        self.check_request_preconditions(
            request.domain_id,
            dtos::DomainPurpose::Llm,
            Gas::from_tgas(self.config.sign_call_gas_attachment_requirement_tera_gas),
            NearToken::from_yoctonear(SIGN_DEPOSIT_YOCTONEAR),
        );

        let callback_gas = Gas::from_tgas(
            self.config
                .return_signature_and_clean_state_on_success_call_tera_gas,
        );

        let request = args_into_llm_inference_request(request);
        let callback_args = serde_json::to_vec(&(&request,)).unwrap();
        self.enqueue_yield_request(
            method_names::RETURN_LLM_INFERENCE_AND_CLEAN_STATE_ON_SUCCESS,
            callback_args,
            callback_gas,
            move |this, id| this.add_llm_inference_request(request, id),
        );
    }

    #[handle_result]
    pub fn respond_llm_inference(
        &mut self,
        request: dtos::LlmInferenceRequest,
        response: dtos::LlmInferenceResponse,
    ) -> Result<(), Error> {
        let signer = Self::assert_caller_is_signer();

        log!(
            "respond_llm_inference: signer={}, request={:?}",
            &signer,
            &request
        );

        self.assert_caller_is_attested_participant_and_protocol_active();

        if !self.protocol_state.is_running_or_resharing() {
            return Err(InvalidState::ProtocolStateNotRunning.into());
        }

        if !self.accept_requests {
            return Err(TeeError::TeeValidationFailed.into());
        }

        let domain = request.domain_id;
        let public_key = self.public_key_extended(domain.0.into())?;

        let signature_is_valid = match (&response.signature, public_key) {
            (
                dtos::SignatureResponse::Secp256k1(signature_response),
                PublicKeyExtended::Secp256k1 { near_public_key },
            ) => {
                let payload_hash: [u8; 32] = response.payload_hash.0;

                near_mpc_signature_verifier::verify_ecdsa_signature(
                    signature_response,
                    &payload_hash,
                    &near_public_key,
                )
                .is_ok()
            }
            (signature_response, public_key_requested) => {
                return Err(RespondError::SignatureSchemeMismatch {
                    mpc_scheme: Box::new(signature_response.clone()),
                    user_scheme: Box::new(public_key_requested),
                }
                .into());
            }
        };

        if !signature_is_valid {
            return Err(RespondError::InvalidSignature.into());
        }

        pending_requests::resolve_yields_for(
            &mut self.pending_llm_inference_requests,
            &request,
            serde_json::to_vec(&response).unwrap(),
        )
    }

    /// Presence check for a pending LLM inference request, exposed as a view
    /// call. Only the `Some`/`None` distinction is meaningful; see
    /// [`Self::get_pending_verify_foreign_tx_request`].
    pub fn get_pending_llm_inference_request(
        &self,
        request: &dtos::LlmInferenceRequest,
    ) -> Option<dtos::YieldIndex> {
        self.pending_llm_inference_requests
            .get(request)
            .and_then(|q| q.first().cloned())
    }

    /// Yield-resume callback for a single queued LLM inference request.
    ///
    /// On success, returns the signed response to the original caller. On
    /// timeout, pops this yield's slot from the pending-request map and fires
    /// `fail_on_timeout`.
    #[private]
    pub fn return_llm_inference_and_clean_state_on_success(
        &mut self,
        request: dtos::LlmInferenceRequest,
        #[callback_result] response: Result<dtos::LlmInferenceResponse, PromiseError>,
    ) -> PromiseOrValue<dtos::LlmInferenceResponse> {
        match response {
            Ok(response) => PromiseOrValue::Value(response),
            Err(_) => {
                pending_requests::pop_oldest_pending_yield(
                    &mut self.pending_llm_inference_requests,
                    &request,
                );
                let fail_on_timeout_gas = Gas::from_tgas(self.config.fail_on_timeout_tera_gas);
                let promise = Promise::new(env::current_account_id()).function_call(
                    method_names::FAIL_ON_TIMEOUT.to_string(),
                    vec![],
                    NearToken::from_near(0),
                    fail_on_timeout_gas,
                );
                near_sdk::PromiseOrValue::Promise(promise.as_return())
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
#[expect(non_snake_case)]
mod tests {
    use super::*;
    use crate::api::test_utils::{SharedSecretKey, basic_setup_with_protocol};
    use assert_matches::assert_matches;
    use dtos::{DomainId, LlmSignPayload, Protocol};
    use k256::ecdsa::SigningKey;
    use k256::{Secp256k1, elliptic_curve};
    use near_sdk::test_utils::VMContextBuilder;
    use near_sdk::{AccountId, testing_env};
    use rand::SeedableRng;
    use std::str::FromStr;

    fn transfer_request_args() -> dtos::LlmInferenceRequestArgs {
        dtos::LlmInferenceRequestArgs {
            domain_id: DomainId::default().0.into(),
            model_id: "Qwen2.5-1.5B-Instruct-4bit".to_string(),
            prompt: "send 1 NEAR to alice.near".to_string(),
            schema: r#"{"action":"transfer|swap|deposit"}"#.to_string(),
        }
    }

    fn sign_llm_payload(
        secret_key: &k256::Scalar,
        payload: &LlmSignPayload,
    ) -> dtos::LlmInferenceResponse {
        let payload_hash = payload.compute_msg_hash().unwrap();
        let secret_key_ec: elliptic_curve::SecretKey<Secp256k1> =
            elliptic_curve::SecretKey::from_bytes(&secret_key.to_bytes()).unwrap();
        let signing_key = SigningKey::from_bytes(&secret_key_ec.to_bytes()).unwrap();
        let (signature, recovery_id) = signing_key
            .sign_prehash_recoverable(&payload_hash.0)
            .unwrap();
        dtos::LlmInferenceResponse {
            payload_hash,
            signature: dtos::SignatureResponse::Secp256k1(
                dtos::K256Signature::from_ecdsa_recoverable(&signature, recovery_id),
            ),
        }
    }

    fn llm_payload(request: &dtos::LlmInferenceRequest) -> LlmSignPayload {
        LlmSignPayload {
            request: request.clone(),
            model_id: request.model_id.clone(),
            output: r#"{"action":"transfer","to":"alice.near","amount":1}"#.to_string(),
        }
    }

    #[test]
    fn request_llm_inference__should_queue_duplicates_from_different_callers() {
        // Given: an Llm purpose domain.
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        let (context, mut contract, _) =
            basic_setup_with_protocol(Protocol::CaitSith, dtos::DomainPurpose::Llm, &mut rng);
        let request_args = transfer_request_args();
        let request = args_into_llm_inference_request(request_args.clone());

        // When: caller alice submits the request.
        let alice = AccountId::from_str("alice.near").unwrap();
        testing_env!(
            VMContextBuilder::new()
                .signer_account_id(alice.clone())
                .predecessor_account_id(alice)
                .current_account_id(context.current_account_id.clone())
                .attached_deposit(NearToken::from_yoctonear(1))
                .build()
        );
        contract.request_llm_inference(request_args.clone());

        // And: caller bob submits the identical request.
        let bob = AccountId::from_str("bob.near").unwrap();
        testing_env!(
            VMContextBuilder::new()
                .signer_account_id(bob.clone())
                .predecessor_account_id(bob)
                .current_account_id(context.current_account_id.clone())
                .attached_deposit(NearToken::from_yoctonear(1))
                .build()
        );
        contract.request_llm_inference(request_args);

        // Then: both yields are queued under the single (caller-agnostic) request key.
        assert_eq!(
            contract
                .pending_llm_inference_requests
                .get(&request)
                .map(|q| q.len()),
            Some(2),
            "duplicate LLM inference requests from different callers should fan out",
        );
    }

    #[test]
    fn respond_llm_inference__should_drain_pending_request_when_response_is_valid() {
        // Given
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        let (context, mut contract, secret_key) =
            basic_setup_with_protocol(Protocol::CaitSith, dtos::DomainPurpose::Llm, &mut rng);
        testing_env!(context.clone());
        let SharedSecretKey::Secp256k1(secret_key) = secret_key else {
            unreachable!();
        };
        let request_args = transfer_request_args();
        let request = args_into_llm_inference_request(request_args.clone());
        contract.request_llm_inference(request_args);
        assert!(
            contract
                .get_pending_llm_inference_request(&request)
                .is_some()
        );

        let payload = llm_payload(&request);
        let response = sign_llm_payload(&secret_key, &payload);
        crate::api::test_utils::with_active_participant_and_attested_context(&contract);

        // When
        contract
            .respond_llm_inference(request.clone(), response.clone())
            .expect("respond_llm_inference should succeed");

        // Then
        assert!(
            contract
                .get_pending_llm_inference_request(&request)
                .is_none()
        );
        contract
            .return_llm_inference_and_clean_state_on_success(request, Ok(response))
            .detach();
    }

    #[test]
    fn respond_llm_inference__should_reject_response_with_tampered_output() {
        // Given: a response signed over a different output than the queued request's payload.
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        let (context, mut contract, secret_key) =
            basic_setup_with_protocol(Protocol::CaitSith, dtos::DomainPurpose::Llm, &mut rng);
        testing_env!(context.clone());
        let SharedSecretKey::Secp256k1(secret_key) = secret_key else {
            unreachable!();
        };
        let request_args = transfer_request_args();
        let request = args_into_llm_inference_request(request_args.clone());
        contract.request_llm_inference(request_args);

        let payload = llm_payload(&request);
        let response = sign_llm_payload(&secret_key, &payload);
        let tampered_request = dtos::LlmInferenceRequest {
            prompt: "send 999 NEAR to evil.near".to_string(),
            ..request.clone()
        };
        crate::api::test_utils::with_active_participant_and_attested_context(&contract);

        // When
        let result = contract.respond_llm_inference(tampered_request, response);

        // Then
        assert_matches!(
            result.unwrap_err(),
            Error::InvalidParameters(crate::errors::InvalidParameters::RequestNotFound)
        );
        assert!(
            contract
                .get_pending_llm_inference_request(&request)
                .is_some(),
            "the pending request must remain unresolved",
        );
    }

    #[test]
    fn respond_llm_inference__should_reject_signature_over_wrong_payload_hash() {
        // Given
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        let (context, mut contract, secret_key) =
            basic_setup_with_protocol(Protocol::CaitSith, dtos::DomainPurpose::Llm, &mut rng);
        testing_env!(context.clone());
        let SharedSecretKey::Secp256k1(secret_key) = secret_key else {
            unreachable!();
        };
        let request_args = transfer_request_args();
        let request = args_into_llm_inference_request(request_args.clone());
        contract.request_llm_inference(request_args);

        let payload = llm_payload(&request);
        let mut response = sign_llm_payload(&secret_key, &payload);
        response.payload_hash = dtos::Hash256([7u8; 32]);
        crate::api::test_utils::with_active_participant_and_attested_context(&contract);

        // When
        let result = contract.respond_llm_inference(request.clone(), response);

        // Then
        assert_matches!(
            result.unwrap_err(),
            Error::Respond(RespondError::InvalidSignature)
        );
        assert!(
            contract
                .get_pending_llm_inference_request(&request)
                .is_some()
        );
    }

    #[test]
    fn llm_sign_payload__should_hash_deterministically() {
        // Given
        let request = args_into_llm_inference_request(transfer_request_args());

        // When
        let payload = llm_payload(&request);
        let hash_a = payload.compute_msg_hash().unwrap();
        let hash_b = payload.compute_msg_hash().unwrap();

        // Then
        assert_eq!(hash_a, hash_b);
    }

    #[rstest::rstest]
    #[case(Protocol::CaitSith, dtos::DomainPurpose::Sign)]
    #[case(Protocol::ConfidentialKeyDerivation, dtos::DomainPurpose::CKD)]
    #[should_panic(expected = "this method requires Llm")]
    fn request_llm_inference__should_reject_non_llm_domain(
        #[case] protocol: Protocol,
        #[case] purpose: dtos::DomainPurpose,
    ) {
        // Given
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        let (_context, mut contract, _sk) = basic_setup_with_protocol(protocol, purpose, &mut rng);

        // When
        contract.request_llm_inference(transfer_request_args());
    }

    #[test]
    fn test_llm_inference_timeout() {
        // Given
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        let (context, mut contract, _) =
            basic_setup_with_protocol(Protocol::CaitSith, dtos::DomainPurpose::Llm, &mut rng);
        testing_env!(
            VMContextBuilder::new()
                .signer_account_id(context.current_account_id.clone())
                .predecessor_account_id(context.current_account_id.clone())
                .current_account_id(context.current_account_id.clone())
                .attached_deposit(NearToken::from_yoctonear(1))
                .build()
        );
        let request_args = transfer_request_args();
        let request = args_into_llm_inference_request(request_args);

        // When
        contract.request_llm_inference(transfer_request_args());

        // Then
        // assert_matches! requires Debug, which PromiseOrValue doesn't implement
        assert!(matches!(
            contract.return_llm_inference_and_clean_state_on_success(
                request.clone(),
                Err(PromiseError::Failed)
            ),
            PromiseOrValue::Promise(_)
        ));
        assert!(
            contract
                .get_pending_llm_inference_request(&request)
                .is_none()
        );
    }
}

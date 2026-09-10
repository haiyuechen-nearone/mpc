use std::time::Duration;

use near_mpc_bounded_collections::BoundedVec;
use near_mpc_contract_interface::types::{
    ECDSA_PAYLOAD_SIZE_BYTES, LlmInferenceRequest, LlmSignPayload, Payload, Tweak,
};
use tokio::time::timeout;

use crate::network::NetworkTaskChannel;
use crate::primitives::UniqueId;
use crate::providers::EcdsaSignatureProvider;
use crate::providers::llm_inference::endpoint::LlmEndpoint;
use crate::providers::llm_inference::validate::validate_output;
use crate::types::{LlmInferenceRequest as NodeLlmRequest, SignatureRequest};

const PRESIGNATURE_TAKE_GRACE_PERIOD: Duration = Duration::from_secs(1);

pub(crate) struct LlmInferenceOutcome {
    pub payload: LlmSignPayload,
    pub signature: threshold_signatures::ecdsa::Signature,
    pub public_key: threshold_signatures::frost_secp256k1::VerifyingKey,
}

pub(crate) fn build_signature_request(
    request: &NodeLlmRequest,
    payload: &LlmSignPayload,
) -> anyhow::Result<SignatureRequest> {
    let msg_hash = payload.compute_msg_hash()?;
    let payload_hash: [u8; ECDSA_PAYLOAD_SIZE_BYTES] = msg_hash.into();
    let payload_bytes: BoundedVec<u8, ECDSA_PAYLOAD_SIZE_BYTES, ECDSA_PAYLOAD_SIZE_BYTES> =
        payload_hash.into();

    Ok(SignatureRequest {
        id: request.id,
        receipt_id: request.receipt_id,
        payload: Payload::Ecdsa(payload_bytes),
        tweak: Tweak::new([0u8; 32]),
        entropy: request.entropy,
        timestamp_nanosec: request.timestamp_nanosec,
        domain: request.request.domain_id,
    })
}

pub(crate) struct LlmInferenceCore<'a> {
    pub ecdsa_signature_provider: &'a EcdsaSignatureProvider,
    pub endpoint: &'a dyn LlmEndpoint,
    pub model_id: &'a str,
}

impl LlmInferenceCore<'_> {
    async fn run_inference(&self, request: &LlmInferenceRequest) -> anyhow::Result<String> {
        let output = self
            .endpoint
            .infer(&request.prompt, &request.schema)
            .await?;
        validate_output(&output, &request.schema)?;
        Ok(output)
    }

    fn build_payload(&self, request: &LlmInferenceRequest, output: String) -> LlmSignPayload {
        LlmSignPayload {
            request: request.clone(),
            model_id: self.model_id.to_string(),
            output,
        }
    }

    pub(crate) async fn make_leader(
        &self,
        request: &NodeLlmRequest,
    ) -> anyhow::Result<LlmInferenceOutcome> {
        tracing::info!(
            target: "mpc",
            request_id = ?request.id,
            "llm inference leader: querying the LLM endpoint"
        );
        let output = self.run_inference(&request.request).await?;
        tracing::info!(
            target: "mpc",
            request_id = ?request.id,
            output = %output,
            "llm inference leader: model output validated against the schema"
        );
        let payload = self.build_payload(&request.request, output);

        // Build and validate the request before the presignature is popped, so invalid
        // requests don't cost a presignature.
        let sign_request = build_signature_request(request, &payload)?;

        let keyshare = self
            .ecdsa_signature_provider
            .keyshare(request.request.domain_id)?;
        let (presignature_id, presignature) = await_with_slow_hook(
            PRESIGNATURE_TAKE_GRACE_PERIOD,
            keyshare.presignature_store.take_owned(),
            || {
                tracing::warn!(
                    domain = ?request.request.domain_id,
                    "no presignatures available, waiting"
                )
            },
        )
        .await;
        let participants = presignature.participants.clone();
        let channel = self.ecdsa_signature_provider.new_channel_for_task(
            crate::providers::llm_inference::LlmInferenceTaskId::LlmInference {
                id: request.id,
                presignature_id,
            },
            participants,
        )?;

        let (signature, public_key) = self
            .ecdsa_signature_provider
            .make_signature_leader_given_parameters(sign_request, presignature, channel)
            .await?;
        Ok(LlmInferenceOutcome {
            payload,
            signature,
            public_key,
        })
    }

    pub(crate) async fn make_follower(
        &self,
        channel: NetworkTaskChannel,
        request: &NodeLlmRequest,
        presignature_id: UniqueId,
    ) -> anyhow::Result<()> {
        tracing::info!(
            target: "mpc",
            request_id = ?request.id,
            "llm inference follower: querying the LLM endpoint"
        );
        let output = self.run_inference(&request.request).await?;
        tracing::info!(
            target: "mpc",
            request_id = ?request.id,
            output = %output,
            "llm inference follower: model output validated against the schema"
        );
        let payload = self.build_payload(&request.request, output);
        let sign_request = build_signature_request(request, &payload)?;

        self.ecdsa_signature_provider
            .make_signature_follower_given_request(channel, presignature_id, sign_request)
            .await
    }
}

// Awaits on the future for specified grace duration, and calls on_slow
// if grace period expires.
async fn await_with_slow_hook<F: Future>(
    grace: Duration,
    fut: F,
    on_slow: impl FnOnce(),
) -> F::Output {
    tokio::pin!(fut);
    match timeout(grace, &mut fut).await {
        Ok(output) => output,
        Err(_) => {
            on_slow();
            fut.await
        }
    }
}

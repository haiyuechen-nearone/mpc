mod endpoint;
mod sign;
mod validate;

use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

use borsh::{BorshDeserialize, BorshSerialize};
use mpc_node_config::ConfigFile;
use near_mpc_contract_interface::types as dtos;

use crate::network::NetworkTaskChannel;
use crate::primitives::{MpcTaskId, UniqueId};
use crate::providers::EcdsaSignatureProvider;
use crate::storage::LlmInferenceRequestStorage;
use crate::types::LlmInferenceId;

pub(crate) use endpoint::{LlmEndpoint, OpenAiCompatEndpoint};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, BorshSerialize, BorshDeserialize)]
pub enum LlmInferenceTaskId {
    LlmInference {
        id: LlmInferenceId,
        presignature_id: UniqueId,
    },
}

impl From<LlmInferenceTaskId> for MpcTaskId {
    fn from(val: LlmInferenceTaskId) -> Self {
        MpcTaskId::LlmInferenceTaskId(val)
    }
}

pub struct LlmInferenceProvider {
    config: Arc<ConfigFile>,
    ecdsa_signature_provider: Arc<EcdsaSignatureProvider>,
    llm_request_store: Arc<LlmInferenceRequestStorage>,
}

impl LlmInferenceProvider {
    pub fn new(
        config: Arc<ConfigFile>,
        ecdsa_signature_provider: Arc<EcdsaSignatureProvider>,
        llm_request_store: Arc<LlmInferenceRequestStorage>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            config,
            ecdsa_signature_provider,
            llm_request_store,
        })
    }

    fn endpoint(&self) -> impl LlmEndpoint + '_ {
        OpenAiCompatEndpoint::new(&self.config.llm)
    }

    pub(crate) async fn make_llm_inference_leader(
        &self,
        id: LlmInferenceId,
    ) -> anyhow::Result<(
        dtos::LlmSignPayload,
        threshold_signatures::ecdsa::Signature,
        threshold_signatures::frost_secp256k1::VerifyingKey,
    )> {
        let request = self.llm_request_store.get(id).await?;
        let endpoint = self.endpoint();
        let core = sign::LlmInferenceCore {
            ecdsa_signature_provider: &self.ecdsa_signature_provider,
            endpoint: &endpoint,
            model_id: &self.config.llm.model,
        };
        let outcome = core.make_leader(&request).await?;
        Ok((outcome.payload, outcome.signature, outcome.public_key))
    }

    pub async fn process_channel(&self, channel: NetworkTaskChannel) -> anyhow::Result<()> {
        match channel.task_id() {
            MpcTaskId::LlmInferenceTaskId(task) => match task {
                LlmInferenceTaskId::LlmInference {
                    id,
                    presignature_id,
                } => {
                    self.make_llm_inference_follower(channel, id, presignature_id)
                        .await?;
                }
            },
            _ => anyhow::bail!(
                "llm_inference task handler: received unexpected task id: {:?}",
                channel.task_id()
            ),
        }

        Ok(())
    }

    async fn make_llm_inference_follower(
        &self,
        channel: NetworkTaskChannel,
        id: LlmInferenceId,
        presignature_id: UniqueId,
    ) -> anyhow::Result<()> {
        let request = timeout(
            Duration::from_secs(self.config.signature.timeout_sec),
            self.llm_request_store.get(id),
        )
        .await??;

        let endpoint = self.endpoint();
        let core = sign::LlmInferenceCore {
            ecdsa_signature_provider: &self.ecdsa_signature_provider,
            endpoint: &endpoint,
            model_id: &self.config.llm.model,
        };
        core.make_follower(channel, &request, presignature_id).await
    }
}

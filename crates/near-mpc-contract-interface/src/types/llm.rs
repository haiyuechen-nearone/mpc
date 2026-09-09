//! LLM inference request/response DTOs: the natural-language prompt path
//! through the MPC contract, signed by an `Llm` purpose domain.

use crate::types::{DomainId, Hash256};
use borsh::{BorshDeserialize, BorshSerialize};
use near_mpc_crypto_types::SignatureResponse;
use serde::{Deserialize, Serialize};
use sha2::Digest;

#[derive(
    Debug,
    Clone,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Serialize,
    Deserialize,
    BorshSerialize,
    BorshDeserialize,
)]
#[cfg_attr(
    all(feature = "abi", not(target_arch = "wasm32")),
    derive(schemars::JsonSchema, borsh::BorshSchema)
)]
pub struct LlmInferenceRequestArgs {
    pub domain_id: DomainId,
    /// The model the caller expects, recorded in the signed payload for auditability.
    pub model_id: String,
    pub prompt: String,
    /// JSON Schema the output must satisfy. Stored and hashed opaquely; validation
    /// is node-side.
    pub schema: String,
}

#[derive(
    Debug,
    Clone,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Serialize,
    Deserialize,
    BorshSerialize,
    BorshDeserialize,
)]
#[cfg_attr(
    all(feature = "abi", not(target_arch = "wasm32")),
    derive(schemars::JsonSchema, borsh::BorshSchema)
)]
pub struct LlmInferenceRequest {
    pub domain_id: DomainId,
    pub model_id: String,
    pub prompt: String,
    pub schema: String,
}

/// Everything the MPC network attests: each participating node ran the same
/// prompt through `model_id` and produced byte-identical `output`.
#[derive(
    Debug,
    Clone,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Serialize,
    Deserialize,
    BorshSerialize,
    BorshDeserialize,
)]
#[cfg_attr(
    all(feature = "abi", not(target_arch = "wasm32")),
    derive(schemars::JsonSchema, borsh::BorshSchema)
)]
pub struct LlmSignPayload {
    pub request: LlmInferenceRequest,
    pub model_id: String,
    pub output: String,
}

impl LlmSignPayload {
    pub fn compute_msg_hash(&self) -> std::io::Result<Hash256> {
        let mut hasher = sha2::Sha256::new();
        borsh::BorshSerialize::serialize(self, &mut hasher)?;
        Ok(Hash256(hasher.finalize().into()))
    }
}

#[derive(
    Debug,
    Clone,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Serialize,
    Deserialize,
    BorshSerialize,
    BorshDeserialize,
)]
#[cfg_attr(
    all(feature = "abi", not(target_arch = "wasm32")),
    derive(schemars::JsonSchema, borsh::BorshSchema)
)]
pub struct LlmInferenceResponse {
    pub payload_hash: Hash256,
    pub signature: SignatureResponse,
}

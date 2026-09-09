use crate::indexer::participants::ContractState;
use crate::p2p::testing::port_seed;
use crate::tests::{
    DEFAULT_BLOCK_TIME, DEFAULT_MAX_PROTOCOL_WAIT_TIME, DEFAULT_MAX_SIGNATURE_WAIT_TIME,
    IntegrationTestSetup, request_llm_inference_and_await_response,
};
use crate::tracking::AutoAbortTask;
use httpmock::prelude::*;
use httpmock::{HttpMockRequest, HttpMockResponse, MockServer};
use mpc_node_config::LlmConfig;
use near_mpc_contract_interface::types::{
    DomainConfig, DomainId, DomainPurpose, Protocol, ReconstructionThreshold,
};
use near_time::Clock;

/// Serves an OpenAI-compatible chat completion with a fixed intent. The body
/// is constant: the test asserts on the agreement + signing flow, not on the
/// model's language skills.
fn must_start_llm_rpc_mock(output: String) -> MockServer {
    let server = MockServer::start();
    server.mock(move |when, then| {
        when.method(POST);
        then.respond_with(move |_req: &HttpMockRequest| {
            let body = serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": output,
                    }
                }]
            });
            HttpMockResponse::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(serde_json::to_string(&body).unwrap())
                .build()
        });
    });
    server
}

fn llm_only_config(llm_url: String) -> LlmConfig {
    LlmConfig {
        url: llm_url,
        model: "test-model".to_string(),
        timeout_sec: 10,
    }
}

// Full M2/M3 flow: three nodes index an llm inference request, each runs the
// prompt against the endpoint, the outputs agree, the network signs and the
// leader submits respond_llm_inference.
#[tokio::test]
#[test_log::test]
#[expect(non_snake_case)]
async fn llm_inference__should_be_served_when_all_nodes_agree() {
    const NUM_PARTICIPANTS: usize = 3;
    const THRESHOLD: usize = 2;
    const TXN_DELAY_BLOCKS: u64 = 1;

    // Given
    let llm_mock = must_start_llm_rpc_mock(
        r#"{"action":"transfer","to":"alice.near","amount":1}"#.to_string(),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    let mut setup: IntegrationTestSetup = IntegrationTestSetup::new(
        Clock::real(),
        temp_dir.path(),
        (0..NUM_PARTICIPANTS)
            .map(|i| format!("test{}", i).parse().unwrap())
            .collect(),
        THRESHOLD,
        TXN_DELAY_BLOCKS,
        port_seed::LLM_INFERENCE_TEST,
        DEFAULT_BLOCK_TIME,
    );
    for config in &mut setup.configs {
        config.config.llm = llm_only_config(llm_mock.base_url());
    }

    let llm_domain = DomainConfig {
        id: DomainId(0),
        protocol: Protocol::CaitSith,
        reconstruction_threshold: ReconstructionThreshold::new(THRESHOLD as u64),
        purpose: DomainPurpose::Llm,
    };
    {
        let mut contract = setup.indexer.contract_mut().await;
        contract.initialize(setup.participants.clone());
        contract.add_domains(vec![llm_domain.clone()]);
    }

    let _runs = setup
        .configs
        .into_iter()
        .map(|config| AutoAbortTask::from(tokio::spawn(config.run())))
        .collect::<Vec<_>>();

    setup
        .indexer
        .wait_for_contract_state(
            |state| matches!(state, ContractState::Running(_)),
            DEFAULT_MAX_PROTOCOL_WAIT_TIME,
        )
        .await
        .expect("timeout waiting for keygen to complete");

    // When, then
    assert!(
        request_llm_inference_and_await_response(
            &mut setup.indexer,
            "user0",
            &llm_domain,
            "send 1 NEAR to alice.near",
            DEFAULT_MAX_SIGNATURE_WAIT_TIME,
        )
        .await
        .is_some()
    );
}

use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use mpc_node_config::LlmConfig;
use serde_json::json;

/// A single inference endpoint. Implementations must be deterministic for a
/// fixed prompt: the MPC network only signs when every node produces
/// byte-identical output, so callers run greedy decoding (temperature 0).
#[async_trait]
pub trait LlmEndpoint: Send + Sync {
    async fn infer(&self, prompt: &str, schema: &str) -> anyhow::Result<String>;
}

/// OpenAI-compatible chat completions client (mlx_lm.server, llama.cpp
/// server, hosted APIs).
pub struct OpenAiCompatEndpoint {
    url: String,
    model: String,
    timeout: Duration,
}

impl OpenAiCompatEndpoint {
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            url: config.url.trim_end_matches('/').to_string(),
            model: config.model.clone(),
            timeout: Duration::from_secs(config.timeout_sec),
        }
    }
}

#[async_trait]
impl LlmEndpoint for OpenAiCompatEndpoint {
    async fn infer(&self, prompt: &str, schema: &str) -> anyhow::Result<String> {
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Extract the wallet intent as JSON matching the given schema. Output only the JSON object."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "intent",
                    "schema": serde_json::from_str::<serde_json::Value>(schema)
                        .context("request schema is not valid JSON")?,
                },
            },
        });

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/chat/completions", self.url))
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
            .context("LLM endpoint request failed")?;

        let status = response.status();
        let text = response.text().await.context("reading LLM endpoint body")?;
        anyhow::ensure!(
            status.is_success(),
            "LLM endpoint returned {status}: {text}"
        );

        let parsed: serde_json::Value =
            serde_json::from_str(&text).context("parsing LLM endpoint response")?;
        let content = parsed
            .pointer("/choices/0/message/content")
            .and_then(|content| content.as_str())
            .context("LLM endpoint response missing choices[0].message.content")?;
        Ok(content.to_string())
    }
}

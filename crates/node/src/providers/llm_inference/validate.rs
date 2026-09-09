use anyhow::Context;

/// Validates an inference output against the caller's JSON schema.
///
/// Mandatory before a verdict counts: grammar masking on the endpoint is a
/// liveness optimization, this is the correctness backstop. The schema
/// constrains shape, not magnitudes, so consumers must still bound field
/// values before acting on them.
pub(crate) fn validate_output(output: &str, schema: &str) -> anyhow::Result<()> {
    let instance: serde_json::Value =
        serde_json::from_str(output).context("inference output is not valid JSON")?;
    let schema: serde_json::Value =
        serde_json::from_str(schema).context("request schema is not valid JSON")?;
    let compiled = jsonschema::validator_for(&schema).context("compiling request schema")?;
    match compiled.iter_errors(&instance).next() {
        Some(error) => anyhow::bail!("inference output violates schema: {error}"),
        None => Ok(()),
    }
}

#[cfg(test)]
#[expect(non_snake_case)]
mod tests {
    use super::*;

    const TRANSFER_SCHEMA: &str = r#"{
        "type": "object",
        "properties": {
            "action": {"enum": ["transfer", "swap", "deposit"]},
            "to": {"type": "string"},
            "amount": {"type": "integer"}
        },
        "required": ["action", "to", "amount"],
        "additionalProperties": false
    }"#;

    #[test]
    fn validate_output__should_accept_schema_conforming_json() {
        // When
        let result = validate_output(
            r#"{"action":"transfer","to":"alice.near","amount":1}"#,
            TRANSFER_SCHEMA,
        );

        // Then
        result.unwrap();
    }

    #[test]
    fn validate_output__should_reject_action_outside_the_closed_enum() {
        // When
        let result = validate_output(
            r#"{"action":"drain","to":"alice.near","amount":1}"#,
            TRANSFER_SCHEMA,
        );

        // Then
        assert!(result.unwrap_err().to_string().contains("violates schema"));
    }

    #[test]
    fn validate_output__should_reject_non_json_output() {
        // When
        let result = validate_output("I cannot do that", TRANSFER_SCHEMA);

        // Then
        assert!(result.unwrap_err().to_string().contains("not valid JSON"));
    }

    #[test]
    fn validate_output__should_pass_huge_magnitude_because_schema_constrains_shape_only() {
        // Given: a prompt injection amount far beyond any real balance.
        let huge = r#"{"action":"transfer","to":"evil.near","amount":10000000000000000000000000000000000000000000000000}"#;

        // When
        let result = validate_output(huge, TRANSFER_SCHEMA);

        // Then: the schema cannot reject it; bounding magnitudes is the
        // consumer contract's job.
        result.unwrap();
    }

    #[test]
    fn validate_output__should_reject_missing_required_field() {
        // When
        let result = validate_output(
            r#"{"action":"transfer","to":"alice.near"}"#,
            TRANSFER_SCHEMA,
        );

        // Then
        assert!(result.unwrap_err().to_string().contains("violates schema"));
    }

    #[test]
    fn validate_output__should_reject_invalid_schema() {
        // When
        let result = validate_output(r#"{}"#, "not a schema");

        // Then
        assert!(result.is_err());
    }
}

package com.bafv.pcta.model;

public class ChallengeResponse {
    private String transaction_id;
    private String nonce;
    private String expected_digits;
    private String prompt_text;
    private Long expires_at_ms;
    private Integer validity_seconds;

    public ChallengeResponse() {}

    public ChallengeResponse(String transaction_id, String nonce, String expected_digits, String prompt_text, Long expires_at_ms, Integer validity_seconds) {
        this.transaction_id = transaction_id;
        this.nonce = nonce;
        this.expected_digits = expected_digits;
        this.prompt_text = prompt_text;
        this.expires_at_ms = expires_at_ms;
        this.validity_seconds = validity_seconds;
    }

    public String getTransaction_id() { return transaction_id; }
    public void setTransaction_id(String transaction_id) { this.transaction_id = transaction_id; }

    public String getNonce() { return nonce; }
    public void setNonce(String nonce) { this.nonce = nonce; }

    public String getExpected_digits() { return expected_digits; }
    public void setExpected_digits(String expected_digits) { this.expected_digits = expected_digits; }

    public String getPrompt_text() { return prompt_text; }
    public void setPrompt_text(String prompt_text) { this.prompt_text = prompt_text; }

    public Long getExpires_at_ms() { return expires_at_ms; }
    public void setExpires_at_ms(Long expires_at_ms) { this.expires_at_ms = expires_at_ms; }

    public Integer getValidity_seconds() { return validity_seconds; }
    public void setValidity_seconds(Integer validity_seconds) { this.validity_seconds = validity_seconds; }
}

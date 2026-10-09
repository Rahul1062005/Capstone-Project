package com.bafv.pcta.model;

public class ChallengeRequest {
    private String transaction_id;
    private Double amount_inr;

    public ChallengeRequest() {}

    public ChallengeRequest(String transaction_id, Double amount_inr) {
        this.transaction_id = transaction_id;
        this.amount_inr = amount_inr;
    }

    public String getTransaction_id() {
        return transaction_id;
    }

    public void setTransaction_id(String transaction_id) {
        this.transaction_id = transaction_id;
    }

    public Double getAmount_inr() {
        return amount_inr;
    }

    public void setAmount_inr(Double amount_inr) {
        this.amount_inr = amount_inr;
    }
}

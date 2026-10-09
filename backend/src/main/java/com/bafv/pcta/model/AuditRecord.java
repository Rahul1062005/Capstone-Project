package com.bafv.pcta.model;

import java.util.Map;

public class AuditRecord {
    private String record_id;
    private String prev_hash;
    private String record_hash;
    private String timestamp;
    private String transaction_id;
    private String event_type;
    private Map<String, Object> features;
    private Boolean deleted = false;

    public AuditRecord() {}

    public AuditRecord(String record_id, String prev_hash, String record_hash, String timestamp, String transaction_id, String event_type, Map<String, Object> features) {
        this.record_id = record_id;
        this.prev_hash = prev_hash;
        this.record_hash = record_hash;
        this.timestamp = timestamp;
        this.transaction_id = transaction_id;
        this.event_type = event_type;
        this.features = features;
        this.deleted = false;
    }

    public String getRecord_id() { return record_id; }
    public void setRecord_id(String record_id) { this.record_id = record_id; }

    public String getPrev_hash() { return prev_hash; }
    public void setPrev_hash(String prev_hash) { this.prev_hash = prev_hash; }

    public String getRecord_hash() { return record_hash; }
    public void setRecord_hash(String record_hash) { this.record_hash = record_hash; }

    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }

    public String getTransaction_id() { return transaction_id; }
    public void setTransaction_id(String transaction_id) { this.transaction_id = transaction_id; }

    public String getEvent_type() { return event_type; }
    public void setEvent_type(String event_type) { this.event_type = event_type; }

    public Map<String, Object> getFeatures() { return features; }
    public void setFeatures(Map<String, Object> features) { this.features = features; }

    public Boolean getDeleted() { return deleted; }
    public void setDeleted(Boolean deleted) { this.deleted = deleted; }
}

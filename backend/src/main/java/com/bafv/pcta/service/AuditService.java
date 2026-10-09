package com.bafv.pcta.service;

import com.bafv.pcta.model.AuditRecord;
import org.springframework.stereotype.Service;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.*;

@Service
public class AuditService {

    private final List<AuditRecord> auditChain = new ArrayList<>();
    private final String genesisHash = "0".repeat(64);

    public AuditService() {
        // Initialize Genesis Block
        AuditRecord genesis = new AuditRecord(
                "GENESIS-BLOCK",
                genesisHash,
                genesisHash,
                Instant.now().toString(),
                "SYS-ROOT",
                "SYSTEM_INITIALIZATION",
                Map.of("system", "Spring Boot BAFV-PCTA Audit Anchor")
        );
        auditChain.add(genesis);
    }

    public synchronized AuditRecord appendRecord(String eventType, String txnId, Map<String, Object> features) {
        AuditRecord prev = auditChain.get(auditChain.size() - 1);
        String prevHash = prev.getRecord_hash();
        String timestamp = Instant.now().toString();
        String recordId = UUID.randomUUID().toString();

        String rawPayload = String.format("%s|%s|%s|%s", prevHash, timestamp, eventType, features.toString());
        String recordHash = calculateSha256(rawPayload);

        AuditRecord newRecord = new AuditRecord(recordId, prevHash, recordHash, timestamp, txnId, eventType, features);
        auditChain.add(newRecord);
        return newRecord;
    }

    public List<AuditRecord> getAllRecords() {
        return new ArrayList<>(auditChain);
    }

    public synchronized boolean purgeRecordMetadata(String recordId) {
        for (AuditRecord r : auditChain) {
            if (r.getRecord_id().equals(recordId)) {
                r.setDeleted(true);
                r.setFeatures(Map.of("purged", true, "reason", "FR-11_ANALYST_PURGE_REQUEST"));
                return true;
            }
        }
        return false;
    }

    private String calculateSha256(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : hashBytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            throw new RuntimeException("SHA-256 calculation failed", e);
        }
    }
}

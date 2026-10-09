package com.bafv.pcta.controller;

import com.bafv.pcta.model.AuditRecord;
import com.bafv.pcta.model.ChallengeRequest;
import com.bafv.pcta.model.ChallengeResponse;
import com.bafv.pcta.service.AiProxyService;
import com.bafv.pcta.service.AuditService;
import com.bafv.pcta.service.ChallengeService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@CrossOrigin(origins = "*")
public class GatewayController {

    private final ChallengeService challengeService;
    private final AuditService auditService;
    private final AiProxyService aiProxyService;

    public GatewayController(ChallengeService challengeService, AuditService auditService, AiProxyService aiProxyService) {
        this.challengeService = challengeService;
        this.auditService = auditService;
        this.aiProxyService = aiProxyService;
    }

    @GetMapping("/")
    public Map<String, Object> rootStatus() {
        return Map.of(
                "project", "BAFV-PCTA (Voice-Clone Fraud Shield)",
                "framework", "Java 21 Spring Boot",
                "role", "Enterprise Gateway & Audit Ledger",
                "status", "ONLINE",
                "endpoints", Map.of(
                        "health", "/health",
                        "audit_records", "/api/audit/records",
                        "challenge_generate", "POST /api/challenge/generate",
                        "scan_analyze", "POST /api/scan/analyze"
                )
        );
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of(
                "status", "healthy",
                "service", "bafv-pcta-spring-gateway",
                "framework", "Spring Boot 3.4.3 (Java 21)",
                "timestamp", Instant.now().toString(),
                "audit_chain_length", auditService.getAllRecords().size()
        );
    }

    @PostMapping("/api/auth/login")
    public ResponseEntity<?> login(@RequestBody Map<String, String> credentials) {
        String username = credentials.get("username");
        String password = credentials.get("password");

        if ("analyst".equals(username) && "fraud2026".equals(password)) {
            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "token", "SPRING-JWT-SIMULATED-TOKEN-2026",
                    "analyst", Map.of("username", "analyst", "role", "ENTERPRISE_FRAUD_INVESTIGATOR")
            ));
        }
        return ResponseEntity.status(401).body(Map.of("error", "Invalid credentials. Default is analyst / fraud2026"));
    }

    @PostMapping("/api/challenge/generate")
    public ResponseEntity<ChallengeResponse> generateChallenge(@RequestBody ChallengeRequest request) {
        if (request.getTransaction_id() == null || request.getAmount_inr() == null) {
            return ResponseEntity.badRequest().build();
        }

        ChallengeResponse response = challengeService.generateChallenge(
                request.getTransaction_id(), request.getAmount_inr()
        );

        auditService.appendRecord("CHALLENGE_GENERATED", request.getTransaction_id(), Map.of(
                "nonce", response.getNonce(),
                "amount_inr", request.getAmount_inr(),
                "expires_in_sec", 60
        ));

        return ResponseEntity.ok(response);
    }

    @PostMapping("/api/scan/analyze")
    public ResponseEntity<?> analyzeAudio(
            @RequestParam("audio") MultipartFile file,
            @RequestParam(value = "transaction_id", defaultValue = "TXN-DEFAULT") String txnId,
            @RequestParam(value = "amount_inr", defaultValue = "25000") Double amountInr,
            @RequestParam(value = "is_new_beneficiary", defaultValue = "false") Boolean isNewBeneficiary,
            @RequestParam(value = "transfer_velocity_per_hour", defaultValue = "1") Integer velocity
    ) {
        try {
            // Forward audio to Python AI Microservice
            Map<String, Object> aiResult = aiProxyService.analyzeAudio(file);

            // Forward to Python Risk Engine
            Map<String, Object> baselineDetection = (Map<String, Object>) aiResult.get("baseline_detection");
            Map<String, Object> scores = (Map<String, Object>) baselineDetection.get("scores");
            Map<String, Object> channelMeta = (Map<String, Object>) aiResult.get("channel_metadata");
            Map<String, Object> biomechanics = (Map<String, Object>) aiResult.get("biomechanical_features");

            Map<String, Object> riskPayload = Map.of(
                    "audio_evidence", Map.of(
                            "voice_spoof_prob", scores.get("spoof_probability"),
                            "bafv_pcta_score", biomechanics.get("bafv_anomaly_score"),
                            "sustained_stability_risk", 0.2,
                            "channel_confidence", channelMeta.get("channel_confidence")
                    ),
                    "transaction_context", Map.of(
                            "transaction_id", txnId,
                            "amount_inr", amountInr,
                            "is_new_beneficiary", isNewBeneficiary,
                            "transfer_velocity_per_hour", velocity
                    ),
                    "previous_smoothed_score", 0.0
            );

            Map<String, Object> riskDecision = aiProxyService.evaluateRisk(riskPayload);

            // Log to cryptographic audit chain
            AuditRecord record = auditService.appendRecord("AUDIO_SCAN_ANALYZED", txnId, Map.of(
                    "file_name", file.getOriginalFilename() != null ? file.getOriginalFilename() : "scan.wav",
                    "file_size_bytes", file.getSize(),
                    "spoof_probability", scores.get("spoof_probability"),
                    "recommendation", riskDecision.get("recommended_action") != null ? riskDecision.get("recommended_action") : scores.get("recommended_action"),
                    "channel_confidence", channelMeta.get("channel_confidence")
            ));

            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("audit_record_id", record.getRecord_id());
            response.put("audit_hash", record.getRecord_hash());
            response.put("ai_result", aiResult);
            response.put("risk_decision", riskDecision);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/api/audit/records")
    public Map<String, Object> getAuditRecords() {
        List<AuditRecord> records = auditService.getAllRecords();
        return Map.of(
                "total_records", records.size(),
                "records", records
        );
    }

    @DeleteMapping("/api/audit/records/{id}")
    public ResponseEntity<?> purgeAuditRecord(@PathVariable("id") String id) {
        boolean success = auditService.purgeRecordMetadata(id);
        if (success) {
            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "message", "Record " + id + " metadata purged in compliance with FR-11."
            ));
        }
        return ResponseEntity.status(404).body(Map.of("error", "Record not found"));
    }
}

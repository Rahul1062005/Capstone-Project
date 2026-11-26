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
@CrossOrigin(originPatterns = "*")
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
            if (aiResult == null) {
                return ResponseEntity.status(502).body(Map.of("error", "AI microservice returned an empty response."));
            }

            // Forward to Python Risk Engine
            Map<String, Object> baselineDetection = (Map<String, Object>) aiResult.get("baseline_detection");
            Map<String, Object> scores = baselineDetection != null ? (Map<String, Object>) baselineDetection.get("scores") : null;
            Map<String, Object> channelMeta = (Map<String, Object>) aiResult.get("channel_metadata");
            Map<String, Object> telephonyConf = (Map<String, Object>) aiResult.get("telephony_confidence");
            Map<String, Object> biomechanics = (Map<String, Object>) aiResult.get("biomechanical_features");

            if (scores == null || channelMeta == null || biomechanics == null) {
                return ResponseEntity.status(502).body(Map.of("error", "AI microservice response is missing required detection fields."));
            }

            Object spoofProb = scores.get("spoof_probability");
            Object bafvScore = biomechanics.get("bafv_anomaly_score");
            Object channelConf = telephonyConf != null && telephonyConf.get("channel_confidence") != null
                    ? telephonyConf.get("channel_confidence")
                    : channelMeta.get("channel_confidence");

            Map<String, Object> audioEvidence = new HashMap<>();
            audioEvidence.put("voice_spoof_prob", spoofProb != null ? spoofProb : 0.0);
            audioEvidence.put("bafv_pcta_score", bafvScore != null ? bafvScore : 0.0);
            audioEvidence.put("sustained_stability_risk", 0.2);
            audioEvidence.put("channel_confidence", channelConf != null ? channelConf : 1.0);

            Map<String, Object> txnContext = new HashMap<>();
            txnContext.put("transaction_id", txnId);
            txnContext.put("amount_inr", amountInr);
            txnContext.put("is_new_beneficiary", isNewBeneficiary);
            txnContext.put("transfer_velocity_per_hour", velocity);

            Map<String, Object> riskPayload = new HashMap<>();
            riskPayload.put("audio_evidence", audioEvidence);
            riskPayload.put("transaction_context", txnContext);
            riskPayload.put("previous_smoothed_score", 0.0);

            Map<String, Object> riskDecision = aiProxyService.evaluateRisk(riskPayload);
            if (riskDecision == null) {
                return ResponseEntity.status(502).body(Map.of("error", "Risk engine returned an empty response."));
            }

            Object finalRecommendation = riskDecision.get("recommended_action") != null
                    ? riskDecision.get("recommended_action")
                    : scores.get("recommended_action");

            // Log to cryptographic audit chain (null-safe HashMap)
            Map<String, Object> auditFeatures = new HashMap<>();
            auditFeatures.put("file_name", file.getOriginalFilename() != null ? file.getOriginalFilename() : "scan.wav");
            auditFeatures.put("file_size_bytes", file.getSize());
            auditFeatures.put("spoof_probability", spoofProb != null ? spoofProb : 0.0);
            auditFeatures.put("recommendation", finalRecommendation != null ? finalRecommendation : "VERIFY");
            auditFeatures.put("channel_confidence", channelConf != null ? channelConf : 1.0);
            auditFeatures.put("transfer_amount", amountInr);

            AuditRecord record = auditService.appendRecord("AUDIO_SCAN_ANALYZED", txnId, auditFeatures);

            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("audit_record_id", record.getRecord_id());
            response.put("audit_hash", record.getRecord_hash());
            response.put("ai_result", aiResult);
            response.put("risk_decision", riskDecision);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            String message = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
            return ResponseEntity.status(502).body(Map.of("error", message));
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

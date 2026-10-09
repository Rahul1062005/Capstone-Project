package com.bafv.pcta.service;

import com.bafv.pcta.model.ChallengeResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

@Service
public class ChallengeService {

    @Value("${security.hmac.secret:BANK-CYBER-FRAUD-GUARD-2026}")
    private String secretKey;

    private static final String[] CODE_WORDS = {
            "alpha", "bravo", "delta", "echo", "foxtrot", "tango", "sierra", "victor"
    };

    private static final String[] DIGIT_WORDS = {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    };

    public ChallengeResponse generateChallenge(String transactionId, Double amountInr) {
        String nonce = UUID.randomUUID().toString().substring(0, 8).toUpperCase();
        long now = System.currentTimeMillis();
        long expiresAt = now + 60000; // 60 seconds

        String rawPayload = String.format("%s|%.2f|%s", transactionId, amountInr, nonce);
        String hmacHex = calculateHmacSha256(rawPayload, secretKey);

        // Derive 4 digits and 2 codewords
        StringBuilder digits = new StringBuilder();
        StringBuilder digitWords = new StringBuilder();

        for (int i = 0; i < 4; i++) {
            int d = Character.digit(hmacHex.charAt(i), 16) % 10;
            digits.append(d);
            digitWords.append(DIGIT_WORDS[d]).append(" ");
        }

        int w1Idx = Character.digit(hmacHex.charAt(4), 16) % CODE_WORDS.length;
        int w2Idx = Character.digit(hmacHex.charAt(5), 16) % CODE_WORDS.length;
        String w1 = CODE_WORDS[w1Idx];
        String w2 = CODE_WORDS[w2Idx];

        String prompt = String.format("Authorize transaction %s: say %sfollowed by %s",
                w1, digitWords.toString(), w2);

        return new ChallengeResponse(
                transactionId,
                nonce,
                digits.toString(),
                prompt,
                expiresAt,
                60
        );
    }

    private String calculateHmacSha256(String data, String key) {
        try {
            Mac sha256Hmac = Mac.getInstance("HmacSHA256");
            SecretKeySpec secretKeySpec = new SecretKeySpec(key.getBytes(StandardCharsets.UTF_8), "HmacSHA256");
            sha256Hmac.init(secretKeySpec);
            byte[] hmacBytes = sha256Hmac.doFinal(data.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : hmacBytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            throw new RuntimeException("HMAC computation failed", e);
        }
    }
}

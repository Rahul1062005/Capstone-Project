package com.bafv.pcta.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Service
public class AiProxyService {

    @Value("${python.ai.url:http://127.0.0.1:8000}")
    private String pythonAiUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    private String baseAiUrl() {
        String url = pythonAiUrl != null ? pythonAiUrl.trim() : "http://127.0.0.1:8000";
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            url = "http://" + url;
        }
        return url.endsWith("/") ? url.substring(0, url.length() - 1) : url;
    }

    public Map<String, Object> analyzeAudio(MultipartFile file) throws Exception {
        String url = baseAiUrl() + "/api/v1/detect/file";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        ByteArrayResource fileAsResource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename() != null ? file.getOriginalFilename() : "audio.wav";
            }
        };
        body.add("file", fileAsResource);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        ResponseEntity<Map> response = restTemplate.postForEntity(url, requestEntity, Map.class);

        return response.getBody();
    }

    public Map<String, Object> evaluateRisk(Map<String, Object> payload) {
        String url = baseAiUrl() + "/api/v1/risk/evaluate";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(payload, headers);
        ResponseEntity<Map> response = restTemplate.postForEntity(url, requestEntity, Map.class);
        return response.getBody();
    }

    public Map<String, Object> verifyChallenge(Map<String, Object> payload) {
        String url = baseAiUrl() + "/api/v1/tbpc/verify";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(payload, headers);
        ResponseEntity<Map> response = restTemplate.postForEntity(url, requestEntity, Map.class);
        return response.getBody();
    }

    public boolean pingAiHealth() {
        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(baseAiUrl() + "/health", Map.class);
            return response.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            return false;
        }
    }
}

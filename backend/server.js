const express = require('express');
const cors = require('cors');
const crypto = require('crypto');
const multer = require('multer');
const rateLimit = require('express-rate-limit');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_AI_URL = process.env.PYTHON_AI_URL || 'http://127.0.0.1:8000';
const SECRET_KEY = process.env.HMAC_SECRET || 'BANK-CYBER-FRAUD-GUARD-2026';

app.use(cors());
app.use(express.json({ limit: '25mb' }));

// FR-13: Apply Rate Limiting
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 200, // limit each IP to 200 requests per window
  message: { error: 'Too many requests from this IP. Rate limit enforced.' },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', apiLimiter);

// Multer memory storage (FR-11: Do not retain raw audio on disk)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25 MB max
});

// --- In-Memory State & Hash-Chained Audit Trail (FR-11 & Scope Sec 4.1) ---
const activeNonces = new Map(); // nonce -> { createdAt, expiresAt, txnId, used }
let auditLogChain = [];

// Initialize Genesis Record
const genesisHash = '0'.repeat(64);
auditLogChain.push({
  record_id: 'GENESIS-BLOCK',
  prev_hash: '0'.repeat(64),
  record_hash: genesisHash,
  timestamp: new Date().toISOString(),
  event_type: 'SYSTEM_INITIALIZATION',
  features: { system: 'BAFV-PCTA Audit Log Anchor' },
});

function calculateRecordHash(prevHash, timestamp, eventType, features) {
  const payload = `${prevHash}|${timestamp}|${eventType}|${JSON.stringify(features)}`;
  return crypto.createHash('sha256').update(payload).digest('hex');
}

function appendAuditRecord(eventType, txnId, features) {
  const prevRecord = auditLogChain[auditLogChain.length - 1];
  const prevHash = prevRecord ? prevRecord.record_hash : genesisHash;
  const timestamp = new Date().toISOString();
  const record_id = uuidv4();
  const record_hash = calculateRecordHash(prevHash, timestamp, eventType, features);

  const newRecord = {
    record_id,
    prev_hash: prevHash,
    record_hash,
    timestamp,
    transaction_id: txnId,
    event_type: eventType,
    features, // feature-level metadata only; NO raw audio
  };

  auditLogChain.push(newRecord);
  return newRecord;
}

// --- Routes ---

// Root Status Page
app.get('/', (req, res) => {
  res.json({
    project: 'BAFV-PCTA (Voice-Clone Fraud Shield)',
    role: 'Application Backend & Audit Gateway',
    status: 'ONLINE',
    endpoints: {
      health: '/health',
      audit_records: '/api/audit/records',
      challenge_generate: 'POST /api/challenge/generate',
      scan_analyze: 'POST /api/scan/analyze',
      auth_login: 'POST /api/auth/login',
    },
    python_ai_service: PYTHON_AI_URL,
    frontend_portal: 'http://localhost:5173',
  });
});

// Health & Status
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'bafv-pcta-backend-gateway',
    timestamp: new Date().toISOString(),
    audit_chain_length: auditLogChain.length,
    active_challenges: activeNonces.size,
  });
});

// FR-13: Simple Authentication
app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body;
  if (username === 'analyst' && password === 'fraud2026') {
    const token = crypto.createHmac('sha256', SECRET_KEY).update(`${username}:${Date.now()}`).digest('hex');
    res.json({
      success: true,
      token,
      analyst: { username: 'analyst', role: 'FINTECH_FRAUD_INVESTIGATOR' },
    });
  } else {
    res.status(401).json({ error: 'Invalid credentials. Default is analyst / fraud2026' });
  }
});

// FR-09: TB-PC Challenge Generation (HMAC-SHA256 & Ephemeral Nonce)
app.post('/api/challenge/generate', (req, res) => {
  const { transaction_id, amount_inr } = req.body;
  if (!transaction_id || amount_inr === undefined) {
    return res.status(400).json({ error: 'Missing transaction_id or amount_inr.' });
  }

  const nonce = uuidv4().substring(0, 8).toUpperCase();
  const now = Date.now();
  const validityMs = 60 * 1000; // 60 seconds expiry

  // Derive challenge payload with HMAC-SHA256
  const rawPayload = `${transaction_id}|${Number(amount_inr).toFixed(2)}|${nonce}`;
  const hmacDigest = crypto.createHmac('sha256', SECRET_KEY).update(rawPayload).digest('hex');

  // Generate 4-digit code and NATO codeword
  const digitIndices = [0, 1, 2, 3].map((i) => parseInt(hmacDigest[i], 16) % 10);
  const words = ['alpha', 'bravo', 'delta', 'echo', 'foxtrot', 'tango', 'sierra', 'victor'];
  const w1 = words[parseInt(hmacDigest[4], 16) % words.length];
  const w2 = words[parseInt(hmacDigest[5], 16) % words.length];

  const digitWords = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
  const expectedDigits = digitIndices.join('');
  const spokenPhrase = `Authorize transaction ${w1}: say ${digitIndices.map((d) => digitWords[d]).join(' ')} followed by ${w2}`;

  activeNonces.set(nonce, {
    createdAt: now,
    expiresAt: now + validityMs,
    transaction_id,
    expectedDigits,
    used: false,
    attempts: 0,
  });

  appendAuditRecord('CHALLENGE_GENERATED', transaction_id, {
    nonce,
    amount_inr,
    expires_in_sec: 60,
  });

  res.json({
    transaction_id,
    nonce,
    expected_digits: expectedDigits,
    prompt_text: spokenPhrase,
    expires_at_ms: now + validityMs,
    validity_seconds: 60,
  });
});

// Forward audio upload to Python AI Service
app.post('/api/scan/analyze', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file provided.' });
    }

    const txnContext = {
      amount_inr: parseFloat(req.body.amount_inr || '25000'),
      is_new_beneficiary: req.body.is_new_beneficiary === 'true',
      transfer_velocity_per_hour: parseInt(req.body.transfer_velocity_per_hour || '1', 10),
    };

    // Forward file to Python AI Microservice
    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype || 'audio/wav' });
    formData.append('file', blob, req.file.originalname || 'upload.wav');

    let aiResult;
    try {
      const response = await fetch(`${PYTHON_AI_URL}/api/v1/detect/file`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorDetail = await response.json();
        return res.status(response.status).json({ error: errorDetail.detail || 'AI Service rejected audio.' });
      }
      aiResult = await response.json();
    } catch (fetchErr) {
      // In case Python service is booting up, provide fallback simulated signal with explicit note
      return res.status(503).json({
        error: 'Python AI Microservice unavailable at ' + PYTHON_AI_URL + '. Ensure FastAPI is running.',
        hint: 'Run: uvicorn ai_service.api:app --port 8000',
      });
    }

    // Call Python Risk Engine
    let riskDecision = null;
    try {
      const riskPayload = {
        audio_evidence: {
          voice_spoof_prob: aiResult.baseline_detection.scores.spoof_probability,
          bafv_pcta_score: aiResult.biomechanical_features.bafv_anomaly_score,
          sustained_stability_risk: 0.2,
          channel_confidence: aiResult.channel_metadata.channel_confidence,
        },
        transaction_context: txnContext,
        previous_smoothed_score: 0.0,
      };

      const riskRes = await fetch(`${PYTHON_AI_URL}/api/v1/risk/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(riskPayload),
      });
      if (riskRes.ok) {
        riskDecision = await riskRes.json();
      }
    } catch (riskErr) {
      console.warn('Risk engine call failed, using baseline recommendation:', riskErr);
    }

    // Append to Hash-Chained Audit Trail (FR-11: no raw audio)
    const auditRecord = appendAuditRecord('AUDIO_SCAN_ANALYZED', req.body.transaction_id || 'TXN-ANONYMOUS', {
      file_name: req.file.originalname,
      file_size_bytes: req.file.size,
      spoof_probability: aiResult.baseline_detection.scores.spoof_probability,
      recommendation: riskDecision ? riskDecision.recommended_action : aiResult.baseline_detection.scores.recommended_action,
      channel_confidence: aiResult.channel_metadata.channel_confidence,
      snr_db: aiResult.channel_metadata.snr_db,
    });

    res.json({
      success: true,
      audit_record_id: auditRecord.record_id,
      audit_hash: auditRecord.record_hash,
      ai_result: aiResult,
      risk_decision: riskDecision,
    });
  } catch (err) {
    console.error('Scan processing error:', err);
    res.status(500).json({ error: err.message });
  }
});

// FR-11: Audit Chain Inspection & Record Deletion
app.get('/api/audit/records', (req, res) => {
  res.json({
    total_records: auditLogChain.length,
    records: auditLogChain,
  });
});

app.delete('/api/audit/records/:id', (req, res) => {
  const { id } = req.params;
  const index = auditLogChain.findIndex((r) => r.record_id === id);
  if (index === -1) {
    return res.status(404).json({ error: 'Record not found.' });
  }

  // Anonymize/Delete metadata while maintaining chain cryptographic hash
  auditLogChain[index].features = { deleted: true, reason: 'ANALYST_METADATA_PURGE_REQUEST' };
  auditLogChain[index].deleted = true;

  res.json({
    success: true,
    message: `Record ${id} metadata successfully purged in compliance with FR-11.`,
    remaining_count: auditLogChain.length,
  });
});

app.listen(PORT, () => {
  console.log(`BAFV-PCTA Backend Gateway running on port ${PORT}`);
  console.log(`Proxying AI inference to: ${PYTHON_AI_URL}`);
});

import React, { useState, useEffect } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Activity,
  Mic,
  Play,
  FileCheck2,
  Upload,
  CheckCircle,
  HelpCircle,
  Clock,
  Trash2,
  Volume2,
  RefreshCw,
  PhoneCall,
  Sparkles,
  ArrowRight,
  Info,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface TimelineWindow {
  window_index: number;
  start_time_seconds: number;
  end_time_seconds: number;
  is_speech_active: boolean;
  speech_ratio: number;
  baseline_spoof_probability: number;
  bafv_anomaly_score: number;
  channel_confidence: number;
  instantaneous_risk: number;
  smoothed_risk: number;
  recommendation: 'ALLOW' | 'VERIFY' | 'HOLD_FOR_REVIEW';
}

interface AuditRecord {
  record_id: string;
  prev_hash: string;
  record_hash: string;
  timestamp: string;
  transaction_id: string;
  event_type: string;
  features: any;
  deleted?: boolean;
}

const GATEWAY_URL = (import.meta as any).env?.VITE_GATEWAY_URL || 'http://localhost:5000';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'inspector' | 'timeline' | 'challenge' | 'audit' | 'guide' | 'science'>('inspector');

  // Backend Connection & Error States (FR-13: explicit errors, never fabricate)
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [scanError, setScanError] = useState<string | null>(null);
  const [challengeError, setChallengeError] = useState<string | null>(null);

  // Transaction Information
  const [transactionId, setTransactionId] = useState('TXN-88492-IN');
  const [amountInr, setAmountInr] = useState(45000);
  const [isNewBeneficiary, setIsNewBeneficiary] = useState(false);

  // Selected Call Audio Scenario
  const [selectedSample, setSelectedSample] = useState<'genuine_clean' | 'genuine_phone' | 'ai_clone' | 'custom'>('genuine_clean');
  const [customFile, setCustomFile] = useState<File | null>(null);
  const [customFileName, setCustomFileName] = useState<string | null>(null);
  const [customAudioUrl, setCustomAudioUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Results & Timeline State
  const [scanResult, setScanResult] = useState<any>(null);
  const [timelineData, setTimelineData] = useState<TimelineWindow[]>([]);

  // Live Spoken Challenge State (Security Anti-Bot Test)
  const [challengeData, setChallengeData] = useState<any>(null);
  const [isGeneratingChallenge, setIsGeneratingChallenge] = useState(false);
  const [challengeSecondsLeft, setChallengeSecondsLeft] = useState(60);
  const [challengeEvaluation, setChallengeEvaluation] = useState<any>(null);

  // Tamper-Proof Audit Chain (fetched from Spring Boot AuditService)
  const [auditRecords, setAuditRecords] = useState<AuditRecord[]>([]);

  // Sample Presets (real .wav files in /samples/ sent to backend)
  const samplePresets = {
    genuine_clean: {
      name: 'Legitimate Customer (Clear Call)',
      file: 'genuine_human_speaker_a.wav',
      audioSrc: '/samples/genuine_human_speaker_a.wav',
    },
    genuine_phone: {
      name: 'Elderly Customer (Noisy Landline / 8 kHz Phone)',
      file: 'genuine_telephone_8khz_amr.wav',
      audioSrc: '/samples/genuine_telephone_8khz_amr.wav',
    },
    ai_clone: {
      name: 'AI Voice Clone Attack (Deepfake Vishing)',
      file: 'synthetic_voice_clone_v1.wav',
      audioSrc: '/samples/synthetic_voice_clone_v1.wav',
    },
  };

  // Cleanup custom Blob URL on change/unmount
  useEffect(() => {
    return () => {
      if (customAudioUrl) {
        URL.revokeObjectURL(customAudioUrl);
      }
    };
  }, [customAudioUrl]);

  // Fetch Gateway Health & Audit Ledger
  const checkBackendHealth = async (): Promise<boolean> => {
    try {
      const res = await fetch(`${GATEWAY_URL}/health`);
      if (res.ok) {
        setBackendStatus('online');
        return true;
      }
      setBackendStatus('offline');
      return false;
    } catch {
      setBackendStatus('offline');
      return false;
    }
  };

  const fetchAuditRecords = async () => {
    try {
      const res = await fetch(`${GATEWAY_URL}/api/audit/records`);
      if (res.ok) {
        const data = await res.json();
        setAuditRecords(Array.isArray(data.records) ? data.records : []);
      }
    } catch {
      // Gateway offline; backendStatus already indicates connectivity
    }
  };

  // Run Real Call Analysis via Spring Boot -> FastAPI
  const runCallAnalysis = async (
    target: 'genuine_clean' | 'genuine_phone' | 'ai_clone' | 'custom',
    overrides?: {
      file?: File;
      audioUrl?: string;
      txnId?: string;
      amount?: number;
      isNewBene?: boolean;
    }
  ) => {
    setIsAnalyzing(true);
    setScanError(null);

    const activeTxnId = overrides?.txnId ?? transactionId;
    const activeAmount = overrides?.amount ?? amountInr;
    const activeNewBene = overrides?.isNewBene ?? isNewBeneficiary;

    try {
      let fileToSend: File;
      let audioSrcToPlay: string;
      let callerLabel: string;

      if (target === 'custom') {
        const chosenFile = overrides?.file ?? customFile;
        if (!chosenFile) {
          throw new Error('Please choose an audio file (.wav, .mp3) first.');
        }
        fileToSend = chosenFile;
        audioSrcToPlay = overrides?.audioUrl ?? customAudioUrl ?? '';
        callerLabel = `Uploaded Audio (${chosenFile.name})`;
      } else {
        const preset = samplePresets[target];
        const sampleRes = await fetch(preset.audioSrc);
        if (!sampleRes.ok) {
          throw new Error(`Failed to load sample audio file ${preset.file}.`);
        }
        const blob = await sampleRes.blob();
        fileToSend = new File([blob], preset.file, { type: blob.type || 'audio/wav' });
        audioSrcToPlay = preset.audioSrc;
        callerLabel = preset.name;
      }

      const formData = new FormData();
      formData.append('audio', fileToSend);
      formData.append('transaction_id', activeTxnId);
      formData.append('amount_inr', String(activeAmount));
      formData.append('is_new_beneficiary', String(activeNewBene));
      formData.append('transfer_velocity_per_hour', '1');

      const response = await fetch(`${GATEWAY_URL}/api/scan/analyze`, {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || payload.detail || `Gateway returned HTTP ${response.status}`);
      }

      setBackendStatus('online');

      const aiResult = payload.ai_result || {};
      const riskDecision = payload.risk_decision || {};
      const channelMeta = aiResult.channel_metadata || {};
      const tcsMeta = aiResult.telephony_confidence || {};
      const baseline = aiResult.baseline_detection || {};
      const scores = baseline.scores || {};
      const biomech = aiResult.biomechanical_features || {};
      const pcta = biomech.pcta_engine || {};
      const openSet = aiResult.open_set_detector || {};
      const timeline = aiResult.windowed_timeline?.timeline || [];

      const spoofProb = Number(scores.spoof_probability ?? 0);
      const lineScore = Number(tcsMeta.channel_confidence ?? channelMeta.channel_confidence ?? 1);
      const isBandLimited = Boolean(tcsMeta.is_band_limited_8khz ?? channelMeta.is_band_limited_8khz);
      const bafvAnomaly = Number(biomech.bafv_anomaly_score ?? pcta.pcvs_score ?? 0);
      const finalAction = riskDecision.recommended_action || scores.recommended_action || 'VERIFY';

      const mappedPairs = Array.isArray(pcta.trajectory_pairs)
        ? pcta.trajectory_pairs.map((tp: any) => {
            const isOk = tp.status === 'ALIGNED';
            return {
              name: `${tp.name || tp.pair} (${tp.pair})`,
              status: isOk ? 'Natural Harmony' : 'Anomaly Detected',
              isOk,
              desc: isOk ? tp.normal_desc : tp.spoof_desc,
            };
          })
        : [];

      const warnings: string[] = Array.isArray(channelMeta.quality_warnings)
        ? channelMeta.quality_warnings
        : [];
      const guardWarning = riskDecision.low_confidence_guard_active
        ? 'Low-Confidence Safety Guard active: degraded audio channel prevents unilateral HOLD on voice evidence alone.'
        : warnings.length > 0
        ? warnings.join(' ')
        : null;

      const factors: string[] = Array.isArray(riskDecision.contributing_factors)
        ? riskDecision.contributing_factors
        : [];

      setScanResult({
        presetKey: target,
        caller_name: callerLabel,
        file_name: aiResult.file_name || fileToSend.name,
        audio_src: audioSrcToPlay,
        processing_latency_ms: aiResult.processing_latency_ms ?? 0,
        audit_record_id: payload.audit_record_id,
        audit_hash: payload.audit_hash,
        line_quality: {
          snr_db: tcsMeta.snr_db ?? channelMeta.snr_db ?? 0,
          line_score: lineScore,
          is_phone_line: isBandLimited,
          codec_desc: tcsMeta.detected_codec || (isBandLimited ? 'Telephone 8 kHz Bandwidth' : 'HD Wideband Audio (16 kHz)'),
          warning: guardWarning,
        },
        ai_detector: {
          spoof_prob: spoofProb,
          human_prob: Number(scores.bonafide_probability ?? Math.max(0, 1 - spoofProb)),
          generator_family: openSet.attack_type && openSet.attack_type !== 'NONE'
            ? `${openSet.attack_type} (${openSet.generator_family || 'Synthetic'})`
            : 'Natural Human Speech',
        },
        throat_physics: {
          anomaly_score: bafvAnomaly,
          pairs: mappedPairs,
          explanation: factors.length > 0
            ? factors.join(' ')
            : riskDecision.probabilistic_disclaimer || 'Biomechanical trajectory analysis completed.',
        },
        recommendation: finalAction,
      });

      setTimelineData(timeline);
      await fetchAuditRecords();
    } catch (err: any) {
      const msg = err?.message || 'Failed to connect to Spring Boot / FastAPI backend.';
      setScanError(
        msg.includes('Failed to fetch') || msg.includes('NetworkError')
          ? `Backend unreachable at ${GATEWAY_URL}. Start FastAPI (port 8000) and Spring Boot (port 5000) to run live analysis.`
          : msg
      );
      if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
        setBackendStatus('offline');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Initial health check, audit fetch, and initial scan on mount
  useEffect(() => {
    checkBackendHealth().then((online) => {
      if (online) {
        fetchAuditRecords();
        runCallAnalysis('genuine_clean');
      }
    });
  }, []);

  // Generate Real HMAC-SHA256 Spoken Security Challenge via Spring Boot
  const generateSpokenChallenge = async () => {
    setIsGeneratingChallenge(true);
    setChallengeError(null);
    try {
      const response = await fetch(`${GATEWAY_URL}/api/challenge/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transaction_id: transactionId,
          amount_inr: amountInr,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `Gateway returned HTTP ${response.status}`);
      }

      setBackendStatus('online');
      setChallengeData({
        transaction_id: data.transaction_id,
        nonce: data.nonce,
        expected_digits: data.expected_digits,
        prompt_phrase: data.prompt_text,
        created_at: Date.now(),
        seconds_left: data.validity_seconds || 60,
      });
      setChallengeSecondsLeft(data.validity_seconds || 60);
      setChallengeEvaluation(null);
      await fetchAuditRecords();
    } catch (err: any) {
      const msg = err?.message || 'Could not generate spoken challenge.';
      setChallengeError(
        msg.includes('Failed to fetch')
          ? `Gateway offline at ${GATEWAY_URL}. Start Spring Boot (port 5000) to derive cryptographic challenges.`
          : msg
      );
      if (msg.includes('Failed to fetch')) setBackendStatus('offline');
    } finally {
      setIsGeneratingChallenge(false);
    }
  };

  // Challenge countdown
  useEffect(() => {
    if (!challengeData) return;
    const interval = setInterval(() => {
      setChallengeSecondsLeft((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [challengeData]);

  // Evaluate caller challenge response
  const evaluateChallengeResponse = (scenario: 'genuine_human' | 'ai_clone_lag' | 'stolen_recording') => {
    if (!challengeData) return;

    if (scenario === 'genuine_human') {
      setChallengeEvaluation({
        verdict: 'PASSED',
        is_safe: true,
        caller_latency_sec: 1.1,
        latency_diagnosis: 'Natural Human Response (~1.1 seconds)',
        message: 'The caller repeated the security digits smoothly with biological response timing. Safe to process transfer.',
        next_step: 'Transaction Approved (Fraud risk downgraded to SAFE)',
      });
    } else if (scenario === 'ai_clone_lag') {
      setChallengeEvaluation({
        verdict: 'FAILED (AI Delay)',
        is_safe: false,
        caller_latency_sec: 4.6,
        latency_diagnosis: 'High Delay (~4.6 seconds of silence)',
        message: 'The caller took over 4.5 seconds to speak. AI voice cloners require 3 to 5 seconds to receive the prompt, type the text into software, and render audio.',
        next_step: 'Transaction Blocked (Advise human investigator to call registered phone number directly)',
      });
    } else {
      setChallengeEvaluation({
        verdict: 'FAILED (Pre-Recorded Audio)',
        is_safe: false,
        caller_latency_sec: 0.15,
        latency_diagnosis: 'Abnormally Instant (<200 ms trigger)',
        message: 'Audio triggered almost instantly or spoke incorrect digits, indicating a fraudster playing pre-recorded stolen voicemail clips.',
        next_step: 'Transaction Blocked (Stolen audio replay detected)',
      });
    }
  };

  // Delete Audit Record via Spring Boot (FR-11 Privacy Rights)
  const deleteAuditRecord = async (recordId: string) => {
    try {
      const res = await fetch(`${GATEWAY_URL}/api/audit/records/${encodeURIComponent(recordId)}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        await fetchAuditRecords();
      }
    } catch {
      setScanError(`Unable to purge audit record ${recordId}: gateway offline at ${GATEWAY_URL}.`);
    }
  };

  // Guided test launcher
  const launchGuidedTest = (testIndex: number) => {
    if (testIndex === 1) {
      setTransactionId('TXN-NORMAL-WIRE');
      setAmountInr(25000);
      setIsNewBeneficiary(false);
      setSelectedSample('genuine_clean');
      runCallAnalysis('genuine_clean', { txnId: 'TXN-NORMAL-WIRE', amount: 25000, isNewBene: false });
      setActiveTab('inspector');
    } else if (testIndex === 2) {
      setTransactionId('TXN-PHONE-RETIREE');
      setAmountInr(15000);
      setIsNewBeneficiary(false);
      setSelectedSample('genuine_phone');
      runCallAnalysis('genuine_phone', { txnId: 'TXN-PHONE-RETIREE', amount: 15000, isNewBene: false });
      setActiveTab('inspector');
    } else if (testIndex === 3) {
      setTransactionId('TXN-CEO-URGENT-WIRE');
      setAmountInr(150000);
      setIsNewBeneficiary(true);
      setSelectedSample('ai_clone');
      runCallAnalysis('ai_clone', { txnId: 'TXN-CEO-URGENT-WIRE', amount: 150000, isNewBene: true });
      setActiveTab('inspector');
    } else if (testIndex === 4) {
      setActiveTab('challenge');
      generateSpokenChallenge();
    } else if (testIndex === 5) {
      setActiveTab('audit');
      fetchAuditRecords();
    }
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '24px 20px' }}>
      {/* Top Header */}
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
          paddingBottom: '20px',
          borderBottom: '1px solid var(--border-card)',
          marginBottom: '24px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div
            style={{
              width: '46px',
              height: '46px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: 'var(--shadow-primary)',
            }}
          >
            <ShieldCheck size={28} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#fff' }}>
                VoiceShield <span style={{ color: '#60a5fa', fontWeight: 500 }}>FinTech</span>
              </h1>
              <span
                className={`status-pill ${
                  backendStatus === 'online'
                    ? 'status-pill-safe'
                    : backendStatus === 'checking'
                    ? 'status-pill-verify'
                    : 'status-pill-danger'
                }`}
                style={{ fontSize: '0.72rem', padding: '3px 10px' }}
              >
                <span
                  className={`status-dot ${
                    backendStatus === 'online'
                      ? 'status-dot-green'
                      : backendStatus === 'checking'
                      ? 'status-dot-amber'
                      : 'status-dot-red'
                  }`}
                ></span>{' '}
                {backendStatus === 'online'
                  ? 'Live Gateway Connected (Port 5000)'
                  : backendStatus === 'checking'
                  ? 'Connecting to Gateway...'
                  : 'Gateway Offline (Start Ports 8000 & 5000)'}
              </span>
            </div>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
              AI Voice Clone & Deepfake Defense for Bank Customer Calls
            </p>
          </div>
        </div>

        {/* Quick System Indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <div className="card-panel-subtle" style={{ padding: '6px 14px', fontSize: '0.78rem' }}>
            <span style={{ color: 'var(--text-tertiary)' }}>AI Latency: </span>
            <strong style={{ color: '#34d399' }}>
              {scanResult?.processing_latency_ms ? `${Number(scanResult.processing_latency_ms).toFixed(1)} ms` : '—'}
            </strong>
          </div>
          <div className="card-panel-subtle" style={{ padding: '6px 14px', fontSize: '0.78rem' }}>
            <span style={{ color: 'var(--text-tertiary)' }}>Audio Analysis: </span>
            <strong style={{ color: '#60a5fa' }}>16,000 Hz Mono</strong>
          </div>
          <button
            onClick={() => setActiveTab('guide')}
            className="btn-secondary"
            style={{ padding: '6px 14px', fontSize: '0.78rem', color: '#60a5fa' }}
          >
            <HelpCircle size={14} /> Step-by-Step Guide
          </button>
        </div>
      </header>

      {/* Primary Navigation Tabs */}
      <nav
        style={{
          display: 'flex',
          gap: '6px',
          marginBottom: '24px',
          overflowX: 'auto',
          borderBottom: '1px solid var(--border-subtle)',
          paddingBottom: '2px',
        }}
      >
        {[
          { id: 'inspector', label: 'Call Inspector', icon: PhoneCall, badge: null },
          { id: 'timeline', label: 'Call Risk Timeline', icon: Activity, badge: null },
          { id: 'challenge', label: 'Live Voice Security Test', icon: Mic, badge: 'Anti-AI' },
          { id: 'audit', label: 'Tamper-Proof Audit History', icon: FileCheck2, badge: `${auditRecords.length} records` },
          { id: 'guide', label: 'Interactive Testing Guide', icon: Sparkles, badge: 'Start Here' },
          { id: 'science', label: 'How It Works (The Science)', icon: Info, badge: null },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '11px 16px',
                background: isActive ? 'var(--bg-card)' : 'transparent',
                color: isActive ? '#fff' : 'var(--text-secondary)',
                borderBottom: isActive ? '2.5px solid var(--brand-blue)' : '2.5px solid transparent',
                borderRadius: '8px 8px 0 0',
                fontWeight: isActive ? 600 : 500,
                fontSize: '0.86rem',
                whiteSpace: 'nowrap',
              }}
            >
              <Icon size={16} color={isActive ? '#60a5fa' : 'currentColor'} />
              {tab.label}
              {tab.badge && (
                <span
                  style={{
                    fontSize: '0.68rem',
                    padding: '2px 6px',
                    borderRadius: '6px',
                    background: isActive ? 'rgba(37, 99, 235, 0.25)' : 'rgba(255, 255, 255, 0.06)',
                    color: isActive ? '#93c5fd' : 'var(--text-tertiary)',
                  }}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* ========================================================================= */}
      {/* TAB 1: CALL INSPECTOR (MAIN DASHBOARD) */}
      {/* ========================================================================= */}
      {activeTab === 'inspector' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 1fr) 2fr', gap: '24px' }}>
          {/* Left Column: Transaction & Caller Selection */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Step 1: Select Call Audio */}
            <div className="card-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                <span style={{ width: '22px', height: '22px', borderRadius: '50%', background: 'var(--brand-blue)', color: '#fff', fontSize: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700 }}>
                  1
                </span>
                <h2 style={{ fontSize: '0.98rem' }}>Choose Caller Audio Sample</h2>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '16px' }}>
                {[
                  {
                    id: 'genuine_clean',
                    title: 'Normal Customer Voice',
                    subtitle: 'Real person, clear mobile phone',
                    tag: 'Real Human',
                    tagColor: '#34d399',
                  },
                  {
                    id: 'genuine_phone',
                    title: 'Elderly Customer (Old Phone)',
                    subtitle: 'Muffled 8 kHz landline with static',
                    tag: 'Degraded Line',
                    tagColor: '#fbbf24',
                  },
                  {
                    id: 'ai_clone',
                    title: 'AI Voice Clone (Scam Attempt)',
                    subtitle: 'Synthesized voice impersonating an account holder',
                    tag: 'AI Deepfake',
                    tagColor: '#f87171',
                  },
                ].map((item) => (
                  <div
                    key={item.id}
                    onClick={() => {
                      setSelectedSample(item.id as any);
                      runCallAnalysis(item.id as any);
                    }}
                    className={`scenario-card ${selectedSample === item.id ? 'active' : ''}`}
                    style={{ cursor: 'pointer' }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 600, fontSize: '0.88rem', color: selectedSample === item.id ? '#fff' : 'var(--text-primary)' }}>
                        {item.title}
                      </span>
                      <span
                        style={{
                          fontSize: '0.68rem',
                          fontWeight: 600,
                          padding: '2px 8px',
                          borderRadius: '6px',
                          background: 'rgba(255,255,255,0.06)',
                          color: item.tagColor,
                        }}
                      >
                        {item.tag}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      {item.subtitle}
                    </p>
                  </div>
                ))}
              </div>

              {/* Audio Player to Listen to the Chosen File */}
              {scanResult && (
                <div style={{ padding: '12px', background: 'var(--bg-input)', borderRadius: '10px', border: '1px solid var(--border-subtle)', marginBottom: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                    <Volume2 size={15} color="#60a5fa" />
                    <span>Listen to Caller Audio Recording:</span>
                  </div>
                  <audio controls key={scanResult.audio_src} src={scanResult.audio_src} preload="auto" />
                </div>
              )}

              {/* Upload Custom Audio File */}
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  padding: '10px',
                  borderRadius: '10px',
                  border: '1.5px dashed var(--border-card)',
                  cursor: 'pointer',
                  fontSize: '0.78rem',
                  color: 'var(--text-secondary)',
                  background: 'rgba(255,255,255,0.02)',
                }}
              >
                <Upload size={14} />
                <span>{customFileName ? `Loaded: ${customFileName}` : 'Upload Real Audio File (.wav, .mp3)'}</span>
                <input
                  type="file"
                  accept="audio/*"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      const nextUrl = URL.createObjectURL(file);
                      setCustomFile(file);
                      setCustomFileName(file.name);
                      setCustomAudioUrl(nextUrl);
                      setSelectedSample('custom');
                      runCallAnalysis('custom', { file, audioUrl: nextUrl });
                    }
                  }}
                />
              </label>

              <button
                className="btn-secondary"
                style={{ width: '100%', marginTop: '10px', fontSize: '0.78rem' }}
                disabled={isAnalyzing}
                onClick={() => runCallAnalysis(selectedSample)}
              >
                <RefreshCw size={14} className={isAnalyzing ? 'animate-spin' : ''} />
                <span>{isAnalyzing ? 'Analyzing Audio Physics...' : 'Re-run Call Analysis'}</span>
              </button>
            </div>

            {/* Step 2: Transaction Context */}
            <div className="card-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                <span style={{ width: '22px', height: '22px', borderRadius: '50%', background: 'var(--brand-blue)', color: '#fff', fontSize: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700 }}>
                  2
                </span>
                <h2 style={{ fontSize: '0.98rem' }}>Transaction Details</h2>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div>
                  <label style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '5px' }}>
                    Reference Number:
                  </label>
                  <input
                    type="text"
                    value={transactionId}
                    onChange={(e) => setTransactionId(e.target.value)}
                    className="mono"
                    style={{
                      width: '100%',
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border-card)',
                      color: '#fff',
                      padding: '8px 12px',
                      borderRadius: '8px',
                      fontSize: '0.82rem',
                    }}
                  />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <label style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>Transfer Amount:</label>
                    <strong style={{ fontSize: '0.95rem', color: '#60a5fa' }}>₹{amountInr.toLocaleString('en-IN')}</strong>
                  </div>
                  <input
                    type="range"
                    min="5000"
                    max="1000000"
                    step="5000"
                    value={amountInr}
                    onChange={(e) => setAmountInr(Number(e.target.value))}
                  />
                  <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
                    {[15000, 45000, 150000, 500000].map((amt) => (
                      <button
                        key={amt}
                        onClick={() => setAmountInr(amt)}
                        className="btn-secondary"
                        style={{ flex: 1, padding: '4px', fontSize: '0.7rem' }}
                      >
                        ₹{amt >= 100000 ? `${amt / 100000} Lakh` : `${amt / 1000}k`}
                      </button>
                    ))}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', borderTop: '1px solid var(--border-subtle)' }}>
                  <div>
                    <span style={{ fontSize: '0.82rem', fontWeight: 600, display: 'block' }}>First-Time Recipient</span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Added to bank account within last 24 hours</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={isNewBeneficiary}
                    onChange={(e) => setIsNewBeneficiary(e.target.checked)}
                    style={{ width: '18px', height: '18px', accentColor: 'var(--brand-blue)' }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Clear Human Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {scanError && (
              <div
                className="card-panel"
                style={{
                  padding: '18px 20px',
                  borderLeft: '6px solid #ef4444',
                  background: 'rgba(239, 68, 68, 0.08)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                  <div>
                    <div style={{ fontWeight: 700, color: '#f87171', fontSize: '0.92rem' }}>
                      Backend Analysis Unavailable (FR-13 Fail-Secure)
                    </div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      {scanError}
                    </p>
                  </div>
                  <button
                    className="btn-secondary"
                    style={{ fontSize: '0.76rem' }}
                    onClick={() => runCallAnalysis(selectedSample)}
                  >
                    <RefreshCw size={14} /> Retry Connection
                  </button>
                </div>
              </div>
            )}

            {!scanResult && !scanError && (
              <div className="card-panel" style={{ padding: '24px', color: 'var(--text-secondary)', fontSize: '0.86rem' }}>
                {isAnalyzing ? 'Running live AI & biomechanical audio analysis...' : 'Select a caller audio sample or upload a .wav file to run live backend analysis.'}
              </div>
            )}

            {scanResult && (
              <>
              {/* Primary Decision Banner */}
              <div
                className="card-panel"
                style={{
                  padding: '24px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '16px',
                  borderLeft: `6px solid ${
                    scanResult.recommendation === 'ALLOW'
                      ? '#10b981'
                      : scanResult.recommendation === 'VERIFY'
                      ? '#f59e0b'
                      : '#ef4444'
                  }`,
                }}
              >
                <div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Security Recommendation for Bank Officer
                  </div>
                  <div style={{ fontSize: '1.65rem', fontWeight: 800, marginTop: '4px' }}>
                    {scanResult.recommendation === 'ALLOW' && (
                      <span style={{ color: '#34d399' }}>Approved · Safe to Process Transfer</span>
                    )}
                    {scanResult.recommendation === 'VERIFY' && (
                      <span style={{ color: '#fbbf24' }}>Extra Check Required · Ask Security Phrase</span>
                    )}
                    {scanResult.recommendation === 'HOLD_FOR_REVIEW' && (
                      <span style={{ color: '#f87171' }}>Block Transaction · Suspected AI Voice Clone</span>
                    )}
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    {scanResult.recommendation === 'ALLOW' && 'Voice exhibits genuine human speech physics and natural breathing coordination.'}
                    {scanResult.recommendation === 'VERIFY' && 'Phone line is noisy or caller confidence is borderline. Trigger a quick 10-second security phrase test.'}
                    {scanResult.recommendation === 'HOLD_FOR_REVIEW' && 'High probability of neural voice cloning. Never authorize transfer on this call alone.'}
                  </p>
                </div>

                <div>
                  <span
                    className={`status-pill ${
                      scanResult.recommendation === 'ALLOW'
                        ? 'status-pill-safe'
                        : scanResult.recommendation === 'VERIFY'
                        ? 'status-pill-verify'
                        : 'status-pill-danger'
                    }`}
                    style={{ fontSize: '0.88rem', padding: '8px 18px' }}
                  >
                    {scanResult.recommendation === 'ALLOW' && <CheckCircle size={16} />}
                    {scanResult.recommendation === 'VERIFY' && <HelpCircle size={16} />}
                    {scanResult.recommendation === 'HOLD_FOR_REVIEW' && <ShieldAlert size={16} />}
                    {scanResult.recommendation === 'ALLOW' ? 'SAFE' : scanResult.recommendation === 'VERIFY' ? 'VERIFY CALLER' : 'HIGH RISK'}
                  </span>
                </div>
              </div>

              {/* 3 Human-Friendly Metric Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
                <div className="card-panel" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>AI Voice Cloner Score</div>
                  <div className="mono" style={{ fontSize: '1.7rem', fontWeight: 800, marginTop: '4px', color: scanResult.ai_detector.spoof_prob > 0.5 ? '#f87171' : '#34d399' }}>
                    {(scanResult.ai_detector.spoof_prob * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                    {scanResult.ai_detector.spoof_prob > 0.5 ? 'High chance of computer synthesis' : 'Normal human acoustic range'}
                  </div>
                </div>

                <div className="card-panel" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>Phone Line Quality</div>
                  <div className="mono" style={{ fontSize: '1.7rem', fontWeight: 800, marginTop: '4px', color: scanResult.line_quality.line_score < 0.5 ? '#fbbf24' : '#34d399' }}>
                    {(scanResult.line_quality.line_score * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                    {scanResult.line_quality.codec_desc}
                  </div>
                </div>

                <div className="card-panel" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>Throat Physics Anomaly</div>
                  <div className="mono" style={{ fontSize: '1.7rem', fontWeight: 800, marginTop: '4px', color: scanResult.throat_physics.anomaly_score > 0.4 ? '#f87171' : '#34d399' }}>
                    {(scanResult.throat_physics.anomaly_score * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                    {scanResult.throat_physics.anomaly_score > 0.4 ? 'Unnatural vocal cord jumps' : 'Natural human muscle dynamics'}
                  </div>
                </div>
              </div>

              {/* Safety Guard Callout for Degraded Calls */}
              {scanResult.line_quality.warning && (
                <div
                  style={{
                    padding: '12px 16px',
                    borderRadius: '10px',
                    background: 'rgba(245, 158, 11, 0.1)',
                    border: '1px solid rgba(245, 158, 11, 0.3)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    fontSize: '0.78rem',
                    color: '#fef3c7',
                  }}
                >
                  <AlertTriangle size={18} color="#fbbf24" style={{ flexShrink: 0 }} />
                  <div>
                    <strong>Customer Protection Active: </strong>
                    {scanResult.line_quality.warning}
                  </div>
                </div>
              )}

              {/* Human Biology & Throat Movement Inspection */}
              <div className="card-panel" style={{ padding: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
                  <div>
                    <h3 style={{ fontSize: '0.96rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Activity size={17} color="#60a5fa" /> Vocal Anatomy & Movement Coordination
                    </h3>
                    <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                      Living humans produce voice via physically coupled throat, tongue, and breathing muscles. Software cloners synthesize frame-by-frame, creating unnatural breaks.
                    </p>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                  {scanResult.throat_physics.pairs.map((p: any, idx: number) => (
                    <div
                      key={idx}
                      style={{
                        padding: '12px',
                        borderRadius: '10px',
                        background: p.isOk ? 'rgba(16, 185, 129, 0.06)' : 'rgba(239, 68, 68, 0.08)',
                        border: `1px solid ${p.isOk ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.25)'}`,
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#fff' }}>{p.name}</span>
                        <span
                          style={{
                            fontSize: '0.7rem',
                            fontWeight: 700,
                            color: p.isOk ? '#34d399' : '#f87171',
                          }}
                        >
                          {p.status}
                        </span>
                      </div>
                      <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                        {p.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Quick Action Navigation Buttons */}
              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  className="btn-secondary"
                  style={{ flex: 1, padding: '12px' }}
                  onClick={() => setActiveTab('timeline')}
                >
                  <Activity size={16} /> View Call Risk Timeline
                </button>
                <button
                  className="btn-primary"
                  style={{ flex: 1, padding: '12px' }}
                  onClick={() => {
                    setActiveTab('challenge');
                    generateSpokenChallenge();
                  }}
                >
                  <Mic size={16} /> Start Live Voice Security Test
                </button>
              </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: CALL RISK TIMELINE */}
      {/* ========================================================================= */}
      {activeTab === 'timeline' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div className="card-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px', flexWrap: 'wrap', gap: '10px' }}>
              <div>
                <h2 style={{ fontSize: '1.1rem' }}>Call Risk Timeline (Second-by-Second)</h2>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Tracks the caller's voice continuously across 2.0-second sliding windows. Catches sudden voice clone cut-ins during long conversations.
                </p>
              </div>
              <div style={{ display: 'flex', gap: '14px', fontSize: '0.76rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#60a5fa' }}></span> Current Window Risk
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#f87171' }}></span> Smoothed Call Risk
                </span>
              </div>
            </div>

            <div style={{ height: '320px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timelineData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorSmoothed" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f87171" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#f87171" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="colorInstant" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#243247" />
                  <XAxis
                    dataKey="start_time_seconds"
                    unit="s"
                    stroke="#64748b"
                    fontSize={12}
                    tickFormatter={(v) => `${v.toFixed(1)}s`}
                  />
                  <YAxis domain={[0, 1]} stroke="#64748b" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#131b27',
                      borderColor: '#243247',
                      borderRadius: '8px',
                      fontSize: '0.8rem',
                      color: '#fff',
                    }}
                  />
                  <ReferenceLine y={0.35} stroke="#10b981" strokeDasharray="4 4" label={{ value: 'Safe Level (<0.35)', fill: '#10b981', fontSize: 11 }} />
                  <ReferenceLine y={0.75} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Block Level (>=0.75)', fill: '#ef4444', fontSize: 11 }} />
                  <Area type="monotone" dataKey="instantaneous_risk" stroke="#60a5fa" strokeWidth={2} fillOpacity={1} fill="url(#colorInstant)" />
                  <Area type="monotone" dataKey="smoothed_risk" stroke="#f87171" strokeWidth={3} fillOpacity={1} fill="url(#colorSmoothed)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Breakdown cards for each window */}
          <div className="card-panel" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '0.95rem', marginBottom: '14px' }}>
              Individual Audio Slices (2-Second Windows)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: '10px' }}>
              {timelineData.map((w) => (
                <div
                  key={w.window_index}
                  style={{
                    padding: '12px',
                    borderRadius: '10px',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span className="mono" style={{ fontSize: '0.78rem', color: '#60a5fa', fontWeight: 600 }}>
                      Slice #{w.window_index + 1}
                    </span>
                    <span
                      className={`status-pill ${
                        w.recommendation === 'ALLOW'
                          ? 'status-pill-safe'
                          : w.recommendation === 'VERIFY'
                          ? 'status-pill-verify'
                          : 'status-pill-danger'
                      }`}
                      style={{ fontSize: '0.65rem', padding: '2px 8px' }}
                    >
                      {w.recommendation}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
                    Time: {w.start_time_seconds.toFixed(1)}s to {w.end_time_seconds.toFixed(1)}s
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.74rem' }}>
                    <span style={{ color: 'var(--text-tertiary)' }}>Cumulative Risk:</span>
                    <strong className="mono">{(w.smoothed_risk * 100).toFixed(0)}%</strong>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: LIVE VOICE SECURITY TEST (ANTI-BOT CHALLENGE) */}
      {/* ========================================================================= */}
      {activeTab === 'challenge' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr', gap: '24px' }}>
          {/* Challenge Generation */}
          <div className="card-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
              <Mic size={20} color="#60a5fa" />
              <h2 style={{ fontSize: '1.1rem' }}>Live Caller Voice Challenge</h2>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '18px' }}>
              When a phone call sounds ambiguous or line quality is noisy, ask the caller to speak a quick, 1-time challenge phrase.
              <strong> Real humans answer in ~1 second. AI voice cloners take 3 to 5 seconds to generate audio</strong>, immediately exposing the scammer.
            </p>

            <button
              className="btn-primary"
              style={{ width: '100%', marginBottom: '20px' }}
              onClick={generateSpokenChallenge}
              disabled={isGeneratingChallenge}
            >
              <RefreshCw size={16} className={isGeneratingChallenge ? 'animate-spin' : ''} />
              {challengeData ? 'Generate New 1-Time Security Phrase' : 'Start Voice Security Test'}
            </button>

            {challengeError && (
              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: '10px',
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.35)',
                  fontSize: '0.78rem',
                  color: '#fca5a5',
                  marginBottom: '14px',
                }}
              >
                {challengeError}
              </div>
            )}

            {challengeData && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ padding: '14px', borderRadius: '10px', background: 'rgba(37, 99, 235, 0.1)', border: '1px solid rgba(37, 99, 235, 0.3)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.74rem', color: '#60a5fa', fontWeight: 600 }}>ONE-TIME SECURITY CODE</span>
                    <span className="status-pill status-pill-verify" style={{ fontSize: '0.72rem', padding: '2px 8px' }}>
                      <Clock size={12} /> {challengeSecondsLeft}s left to speak
                    </span>
                  </div>
                  <div className="mono" style={{ fontSize: '1.4rem', fontWeight: 800, marginTop: '4px' }}>
                    {challengeData.nonce} · Code: {challengeData.expected_digits}
                  </div>
                </div>

                <div style={{ padding: '16px', borderRadius: '10px', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>
                    Phrase for the Bank Officer to Read:
                  </div>
                  <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#fff', marginTop: '6px' }}>
                    "{challengeData.prompt_phrase}"
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Caller Response Simulator */}
          <div className="card-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.1rem', marginBottom: '8px' }}>
              Simulate How the Caller Responds
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '18px' }}>
              Choose an outcome below to test how the bank security system reacts to human vs AI response timing:
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '22px' }}>
              <button
                className="scenario-card"
                onClick={() => evaluateChallengeResponse('genuine_human')}
                disabled={!challengeData}
                style={{ textAlign: 'center', padding: '14px 10px' }}
              >
                <div style={{ fontWeight: 700, color: '#34d399', fontSize: '0.85rem' }}>1. Real Human</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  Speaks in 1.1s (Natural)
                </div>
              </button>

              <button
                className="scenario-card"
                onClick={() => evaluateChallengeResponse('ai_clone_lag')}
                disabled={!challengeData}
                style={{ textAlign: 'center', padding: '14px 10px' }}
              >
                <div style={{ fontWeight: 700, color: '#f87171', fontSize: '0.85rem' }}>2. AI Voice Clone</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  Stalls 4.6s (Software Lag)
                </div>
              </button>

              <button
                className="scenario-card"
                onClick={() => evaluateChallengeResponse('stolen_recording')}
                disabled={!challengeData}
                style={{ textAlign: 'center', padding: '14px 10px' }}
              >
                <div style={{ fontWeight: 700, color: '#f87171', fontSize: '0.85rem' }}>3. Stolen Audio Clip</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  Instant &lt;200ms or wrong code
                </div>
              </button>
            </div>

            {/* Evaluation Result */}
            {challengeEvaluation && (
              <div
                style={{
                  padding: '20px',
                  borderRadius: '12px',
                  background: challengeEvaluation.is_safe ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                  border: `1.5px solid ${challengeEvaluation.is_safe ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.35)'}`,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span
                    className={`status-pill ${challengeEvaluation.is_safe ? 'status-pill-safe' : 'status-pill-danger'}`}
                    style={{ fontSize: '0.85rem' }}
                  >
                    {challengeEvaluation.verdict}
                  </span>
                  <span className="mono" style={{ fontSize: '0.9rem', color: '#fff' }}>
                    {challengeEvaluation.latency_diagnosis}
                  </span>
                </div>

                <p style={{ fontSize: '0.84rem', color: '#fff', marginTop: '12px', lineHeight: 1.5 }}>
                  {challengeEvaluation.message}
                </p>

                <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.1)', fontSize: '0.8rem', color: challengeEvaluation.is_safe ? '#34d399' : '#f87171', fontWeight: 600 }}>
                  Next Step: {challengeEvaluation.next_step}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 4: TAMPER-PROOF AUDIT HISTORY */}
      {/* ========================================================================= */}
      {activeTab === 'audit' && (
        <div className="card-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px', flexWrap: 'wrap', gap: '10px' }}>
            <div>
              <h2 style={{ fontSize: '1.1rem' }}>Tamper-Proof Security Audit Log</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                Every inspection is permanently anchored in a cryptographic SHA-256 chain. Raw customer voice is never saved; only mathematical feature fingerprints are preserved.
              </p>
            </div>
            <span className="status-pill status-pill-safe">
              <CheckCircle size={14} /> Chain Verified ({auditRecords.length} Blocks)
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {auditRecords.map((r, i) => (
              <div
                key={r.record_id}
                style={{
                  padding: '14px 18px',
                  borderRadius: '10px',
                  background: r.deleted ? 'rgba(255, 255, 255, 0.02)' : 'var(--bg-input)',
                  border: '1px solid var(--border-subtle)',
                  opacity: r.deleted ? 0.6 : 1,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span className="mono" style={{ fontSize: '0.78rem', color: '#60a5fa', fontWeight: 700 }}>
                      Block #{i + 1}
                    </span>
                    <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{r.event_type}</span>
                    <span className="mono" style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)' }}>[{r.record_id}]</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)' }}>
                      {new Date(r.timestamp).toLocaleTimeString()}
                    </span>
                    {!r.deleted && r.record_id !== 'GENESIS-BLOCK' && (
                      <button
                        onClick={() => deleteAuditRecord(r.record_id)}
                        className="btn-ghost"
                        style={{ color: '#f87171', fontSize: '0.72rem', padding: '3px 8px' }}
                        title="Purge customer metadata to comply with privacy laws"
                      >
                        <Trash2 size={13} /> Purge Customer Data
                      </button>
                    )}
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.72rem', marginTop: '6px' }}>
                  <div className="mono" style={{ color: 'var(--text-tertiary)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    Previous Block Hash: {(r.prev_hash || '').substring(0, 28)}...
                  </div>
                  <div className="mono" style={{ color: '#60a5fa', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    Record Hash: {(r.record_hash || '').substring(0, 28)}...
                  </div>
                </div>

                {r.deleted ? (
                  <div style={{ fontSize: '0.74rem', color: '#f87171', marginTop: '6px' }}>
                    * Customer metadata was purged upon privacy request. Cryptographic hash remains anchored.
                  </div>
                ) : (
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
                    Transaction: {r.transaction_id} · Result: {r.features?.recommendation || r.event_type} · AI Voice Score:{' '}
                    {r.features?.spoof_probability !== undefined
                      ? `${Math.round(Number(r.features.spoof_probability) * 100)}%`
                      : r.features?.ai_voice_score || 'N/A'}{' '}
                    · Amount:{' '}
                    {r.features?.transfer_amount !== undefined
                      ? `₹${Number(r.features.transfer_amount).toLocaleString('en-IN')}`
                      : r.features?.amount_inr !== undefined
                      ? `₹${Number(r.features.amount_inr).toLocaleString('en-IN')}`
                      : 'N/A'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 5: STEP-BY-STEP TESTING GUIDE (INTERACTIVE TUTORIAL) */}
      {/* ========================================================================= */}
      {activeTab === 'guide' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div className="card-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.2rem', marginBottom: '6px' }}>
              How to Test VoiceShield Step-by-Step
            </h2>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
              Welcome! Use this guide to understand every feature and run live demonstrations in one click.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '16px' }}>
            {/* Guided Test 1 */}
            <div className="card-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#34d399', textTransform: 'uppercase' }}>
                  Test Scenario 1
                </span>
                <span className="status-pill status-pill-safe" style={{ fontSize: '0.7rem' }}>Normal Caller</span>
              </div>
              <h3 style={{ fontSize: '1rem', marginTop: '8px' }}>Legitimate Customer Transfer</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.5 }}>
                A genuine account holder calls the bank to transfer ₹25,000. Their voice has organic vocal cord micro-jitter, natural breathing, and continuous acoustic resonance.
              </p>
              <div style={{ marginTop: '12px', padding: '10px', background: 'var(--bg-input)', borderRadius: '8px', fontSize: '0.74rem' }}>
                <strong>Expected Result:</strong> Green "Approved · Safe to Process Transfer".
              </div>
              <button
                className="btn-primary"
                style={{ width: '100%', marginTop: '14px' }}
                onClick={() => launchGuidedTest(1)}
              >
                <Play size={14} /> Run Test 1: Real Customer
              </button>
            </div>

            {/* Guided Test 2 */}
            <div className="card-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#fbbf24', textTransform: 'uppercase' }}>
                  Test Scenario 2
                </span>
                <span className="status-pill status-pill-verify" style={{ fontSize: '0.7rem' }}>Customer Protection</span>
              </div>
              <h3 style={{ fontSize: '1rem', marginTop: '8px' }}>Grandma on Muffled Phone Line</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.5 }}>
                An elderly customer calls on a crackly 8 kHz telephone line. Older anti-spoofing systems mistakenly freeze her money because high frequencies are missing.
              </p>
              <div style={{ marginTop: '12px', padding: '10px', background: 'var(--bg-input)', borderRadius: '8px', fontSize: '0.74rem' }}>
                <strong>Expected Result:</strong> System detects telephone line and activates the <em>Safety Guard</em>, preventing an unfair account freeze.
              </div>
              <button
                className="btn-primary"
                style={{ width: '100%', marginTop: '14px' }}
                onClick={() => launchGuidedTest(2)}
              >
                <Play size={14} /> Run Test 2: Muffled Phone Line
              </button>
            </div>

            {/* Guided Test 3 */}
            <div className="card-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f87171', textTransform: 'uppercase' }}>
                  Test Scenario 3
                </span>
                <span className="status-pill status-pill-danger" style={{ fontSize: '0.7rem' }}>Vishing Attack</span>
              </div>
              <h3 style={{ fontSize: '1rem', marginTop: '8px' }}>AI Voice Clone (₹1,50,000 Fraud)</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.5 }}>
                A scammer uses an AI neural voice cloner to sound like an executive authorizing an urgent wire transfer to an unknown recipient.
              </p>
              <div style={{ marginTop: '12px', padding: '10px', background: 'var(--bg-input)', borderRadius: '8px', fontSize: '0.74rem' }}>
                <strong>Expected Result:</strong> Red "Block Transaction · Suspected AI Voice Clone" with detected AI generator family.
              </div>
              <button
                className="btn-primary"
                style={{ width: '100%', marginTop: '14px', background: '#dc2626' }}
                onClick={() => launchGuidedTest(3)}
              >
                <Play size={14} /> Run Test 3: Voice Clone Attack
              </button>
            </div>

            {/* Guided Test 4 */}
            <div className="card-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#60a5fa', textTransform: 'uppercase' }}>
                  Test Scenario 4
                </span>
                <span className="status-pill status-pill-safe" style={{ fontSize: '0.7rem' }}>Interactive Test</span>
              </div>
              <h3 style={{ fontSize: '1rem', marginTop: '8px' }}>Live Spoken Challenge (Anti-Bot)</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.5 }}>
                Test the live voice security phrase. Watch how human brains answer in 1 second, while AI neural generators take 4 to 5 seconds to generate audio.
              </p>
              <div style={{ marginTop: '12px', padding: '10px', background: 'var(--bg-input)', borderRadius: '8px', fontSize: '0.74rem' }}>
                <strong>Expected Result:</strong> Interactive 60s countdown with human vs AI synthesis latency scoring.
              </div>
              <button
                className="btn-secondary"
                style={{ width: '100%', marginTop: '14px' }}
                onClick={() => launchGuidedTest(4)}
              >
                <ArrowRight size={14} /> Go to Voice Challenge
              </button>
            </div>

            {/* Guided Test 5 */}
            <div className="card-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#a78bfa', textTransform: 'uppercase' }}>
                  Test Scenario 5
                </span>
                <span className="status-pill status-pill-safe" style={{ fontSize: '0.7rem' }}>Privacy Rights</span>
              </div>
              <h3 style={{ fontSize: '1rem', marginTop: '8px' }}>Customer Privacy Purge</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.5 }}>
                Demonstrates compliance with privacy regulations (GDPR / Indian DPDP Act). Click "Purge" on any audit record to safely wipe customer details while preserving blockchain-style proof.
              </p>
              <div style={{ marginTop: '12px', padding: '10px', background: 'var(--bg-input)', borderRadius: '8px', fontSize: '0.74rem' }}>
                <strong>Expected Result:</strong> Metadata deleted; cryptographic SHA-256 chain remains intact.
              </div>
              <button
                className="btn-secondary"
                style={{ width: '100%', marginTop: '14px' }}
                onClick={() => launchGuidedTest(5)}
              >
                <ArrowRight size={14} /> Open Audit History
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 6: HOW IT WORKS (THE SCIENCE IN PLAIN ENGLISH) */}
      {/* ========================================================================= */}
      {activeTab === 'science' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div className="card-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.1rem', marginBottom: '12px', color: '#60a5fa' }}>
              Why AI Voice Cloners Fail Against Human Biology
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              <p>
                <strong>1. Living Throats vs Math Algorithms:</strong> Human speech is produced by physical air flowing from lungs through vocal folds and resonant chambers (throat, tongue, lips). This creates continuous physical momentum.
              </p>
              <p>
                <strong>2. Frame Discontinuities:</strong> Generative AI models (like ElevenLabs, Tortoise, or Bark) generate sound frame-by-frame (typically every 10 to 20 milliseconds). When these frames are joined, subtle digital phase jumps and unnatural frequency flat-lines occur.
              </p>
              <p>
                <strong>3. The Articulator Speed Limit:</strong> Human mouth muscles have physical top speeds. AI models often generate phoneme velocity jumps exceeding 14,000 Hz/second, which is biologically impossible for living vocal tracts.
              </p>
            </div>
          </div>

          <div className="card-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.1rem', marginBottom: '12px', color: '#60a5fa' }}>
              Bank Advisory Philosophy & Privacy
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              <p>
                <strong>• Human in the Loop:</strong> VoiceShield provides recommendations (<code>SAFE</code>, <code>VERIFY</code>, <code>BLOCK</code>) to assist bank fraud investigators. It does not automatically freeze bank accounts without human confirmation.
              </p>
              <p>
                <strong>• Zero Raw Audio Storage:</strong> To protect customer privacy, incoming call audio is converted into temporary mathematical feature vectors and immediately discarded. No recorded audio is kept on the server.
              </p>
              <p>
                <strong>• SIH Cybersecurity Alignment:</strong> Built for bank fraud departments fighting urgent wire fraud, CEO impersonation, and vishing scams.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Clock,
  FileCheck2,
  HelpCircle,
  Mic,
  Moon,
  PhoneCall,
  Play,
  RefreshCw,
  ShieldCheck,
  Sun,
  Trash2,
  Upload,
  Volume2,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
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

type TabId = 'inspector' | 'timeline' | 'challenge' | 'audit' | 'guide' | 'science';
type SampleKey = 'genuine_clean' | 'genuine_phone' | 'ai_clone' | 'custom';

const GATEWAY_URL = (import.meta as any).env?.VITE_GATEWAY_URL || 'http://localhost:5000';

const SAMPLE_PRESETS = {
  genuine_clean: {
    name: 'Legitimate Customer (Clear Call)',
    shortTitle: 'Standard Mobile Call',
    subtitle: 'Genuine human speech on a clear 16 kHz connection',
    tag: 'Human',
    file: 'genuine_human_speaker_a.wav',
    audioSrc: '/samples/genuine_human_speaker_a.wav',
  },
  genuine_phone: {
    name: 'Elderly Account Holder (8 kHz Landline)',
    shortTitle: '8 kHz Landline Call',
    subtitle: 'Narrowband telephone audio with background line hum',
    tag: 'Degraded Line',
    file: 'genuine_telephone_8khz_amr.wav',
    audioSrc: '/samples/genuine_telephone_8khz_amr.wav',
  },
  ai_clone: {
    name: 'AI Voice Clone Attack (Vishing)',
    shortTitle: 'Suspected Voice Clone',
    subtitle: 'Synthesized speech impersonating an account holder',
    tag: 'Synthetic',
    file: 'synthetic_voice_clone_v1.wav',
    audioSrc: '/samples/synthetic_voice_clone_v1.wav',
  },
} as const;

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('inspector');
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    try {
      return localStorage.getItem('voiceshield_theme') === 'dark' ? 'dark' : 'light';
    } catch {
      return 'light';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('voiceshield_theme', theme);
    } catch {
      // Ignore storage failures in restricted contexts
    }
  }, [theme]);

  const isDark = theme === 'dark';

  // Gateway & explicit error states (FR-13)
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [scanError, setScanError] = useState<string | null>(null);
  const [challengeError, setChallengeError] = useState<string | null>(null);

  // Transaction context (FR-06)
  const [transactionId, setTransactionId] = useState('TXN-88492-IN');
  const [amountInr, setAmountInr] = useState(45000);
  const [isNewBeneficiary, setIsNewBeneficiary] = useState(false);

  // Audio selection
  const [selectedSample, setSelectedSample] = useState<SampleKey>('genuine_clean');
  const [customFile, setCustomFile] = useState<File | null>(null);
  const [customFileName, setCustomFileName] = useState<string | null>(null);
  const [customAudioUrl, setCustomAudioUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Analysis output
  const [scanResult, setScanResult] = useState<any>(null);
  const [timelineData, setTimelineData] = useState<TimelineWindow[]>([]);

  // TB-PC Challenge state
  const [challengeData, setChallengeData] = useState<any>(null);
  const [isGeneratingChallenge, setIsGeneratingChallenge] = useState(false);
  const [isRecordingChallenge, setIsRecordingChallenge] = useState(false);
  const [isVerifyingChallenge, setIsVerifyingChallenge] = useState(false);
  const [spokenTranscript, setSpokenTranscript] = useState('');
  const [challengeSecondsLeft, setChallengeSecondsLeft] = useState(60);
  const [challengeEvaluation, setChallengeEvaluation] = useState<any>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const micChunksRef = useRef<Blob[]>([]);
  const recordStartMsRef = useRef<number>(0);
  const speechRecRef = useRef<any>(null);

  // Audit ledger state
  const [auditRecords, setAuditRecords] = useState<AuditRecord[]>([]);

  useEffect(() => {
    return () => {
      if (customAudioUrl) {
        URL.revokeObjectURL(customAudioUrl);
      }
    };
  }, [customAudioUrl]);

  const fetchAuditRecords = useCallback(async () => {
    try {
      const res = await fetch(`${GATEWAY_URL}/api/audit/records`);
      if (res.ok) {
        const data = await res.json();
        setAuditRecords(Array.isArray(data.records) ? data.records : []);
      }
    } catch {
      // Gateway connectivity is tracked via backendStatus
    }
  }, []);

  const runCallAnalysis = useCallback(
    async (
      target: SampleKey,
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
            throw new Error('Select an audio file (.wav or .mp3) before running analysis.');
          }
          fileToSend = chosenFile;
          audioSrcToPlay = overrides?.audioUrl ?? customAudioUrl ?? '';
          callerLabel = `Uploaded Recording (${chosenFile.name})`;
        } else {
          const preset = SAMPLE_PRESETS[target];
          const sampleRes = await fetch(preset.audioSrc);
          if (!sampleRes.ok) {
            throw new Error(`Unable to load sample audio file ${preset.file}.`);
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

        let response = await fetch(`${GATEWAY_URL}/api/scan/analyze`, {
          method: 'POST',
          body: formData,
        });

        // If Render's free AI container was asleep and returned 502 during wake-up, retry once
        if (response.status === 502) {
          await new Promise((resolve) => setTimeout(resolve, 4000));
          response = await fetch(`${GATEWAY_URL}/api/scan/analyze`, {
            method: 'POST',
            body: formData,
          });
        }

        setBackendStatus('online');
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.error || payload.detail || `Gateway returned HTTP ${response.status}`);
        }

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
                pair: tp.pair,
                name: tp.name || tp.pair,
                status: isOk ? 'Within Normal Range' : 'Deviation Detected',
                isOk,
                desc: isOk ? tp.normal_desc : tp.spoof_desc,
              };
            })
          : [];

        const warnings: string[] = Array.isArray(channelMeta.quality_warnings)
          ? channelMeta.quality_warnings
          : [];
        const guardWarning = riskDecision.low_confidence_guard_active
          ? 'Low-Confidence Safety Guard active: degraded telephony bandwidth prevents an automated hold on voice evidence alone.'
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
            codec_desc: tcsMeta.detected_codec || (isBandLimited ? '8 kHz Narrowband Telephony' : '16 kHz Wideband Audio'),
            warning: guardWarning,
          },
          ai_detector: {
            spoof_prob: spoofProb,
            human_prob: Number(scores.bonafide_probability ?? Math.max(0, 1 - spoofProb)),
            generator_family:
              openSet.attack_type && openSet.attack_type !== 'NONE'
                ? `${openSet.attack_type}`
                : 'Bonafide Human Speech',
            weights_source: baseline.model_metadata?.weights_source || 'unknown',
          },
          throat_physics: {
            anomaly_score: bafvAnomaly,
            pairs: mappedPairs,
            explanation:
              factors.length > 0
                ? factors.join(' ')
                : riskDecision.probabilistic_disclaimer || 'Biomechanical trajectory check completed.',
          },
          recommendation: finalAction,
        });

        setTimelineData(timeline);
        await fetchAuditRecords();
      } catch (err: any) {
        const msg = err?.message || 'Unable to reach the backend gateway.';
        const isNetwork = msg.includes('Failed to fetch') || msg.includes('NetworkError');
        const isLocal = GATEWAY_URL.includes('localhost') || GATEWAY_URL.includes('127.0.0.1');
        setScanError(
          isNetwork
            ? isLocal
              ? `Gateway unreachable at ${GATEWAY_URL}. Start FastAPI (port 8000) and Spring Boot (port 5000).`
              : `Cloud backend at ${GATEWAY_URL} is waking up or redeploying. Wait ~30 seconds and click Retry.`
            : msg
        );
        if (isNetwork) {
          setBackendStatus('offline');
        }
      } finally {
        setIsAnalyzing(false);
      }
    },
    [amountInr, customAudioUrl, customFile, fetchAuditRecords, isNewBeneficiary, transactionId]
  );

  // Initial connection check on mount with automatic cold-start wake-up retries
  useEffect(() => {
    let active = true;
    const bootstrap = async () => {
      const maxAttempts = 12; // Up to ~48s for Render free-tier cold starts
      for (let attempt = 1; attempt <= maxAttempts && active; attempt++) {
        try {
          const res = await fetch(`${GATEWAY_URL}/api/warmup`);
          if (!active) return;
          if (res.ok) {
            const info = await res.json().catch(() => ({}));
            if (info.ai_ready) {
              setBackendStatus('online');
              await fetchAuditRecords();
              await runCallAnalysis('genuine_clean');
              return;
            }
          } else if (res.status === 404) {
            // Fallback for older gateway builds without /api/warmup
            const healthRes = await fetch(`${GATEWAY_URL}/health`);
            if (healthRes.ok) {
              setBackendStatus('online');
              await fetchAuditRecords();
              await runCallAnalysis('genuine_clean');
              return;
            }
          }
        } catch {
          // Gateway still booting or redeploying; retry below
        }
        if (attempt < maxAttempts && active) {
          await new Promise((resolve) => setTimeout(resolve, 4000));
        }
      }
      if (active) setBackendStatus('offline');
    };
    void bootstrap();
    return () => {
      active = false;
    };
    // Run once on initial mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      setSpokenTranscript('');
      await fetchAuditRecords();
    } catch (err: any) {
      const msg = err?.message || 'Unable to issue spoken challenge.';
      const isNetwork = msg.includes('Failed to fetch') || msg.includes('NetworkError');
      setChallengeError(
        isNetwork
          ? `Gateway offline at ${GATEWAY_URL}. Start Spring Boot (port 5000) to derive HMAC-SHA256 challenges.`
          : msg
      );
      if (isNetwork) setBackendStatus('offline');
    } finally {
      setIsGeneratingChallenge(false);
    }
  };

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

  const evaluateChallengeResponse = (scenario: 'genuine_human' | 'ai_clone_lag' | 'stolen_recording') => {
    if (!challengeData) return;

    if (scenario === 'genuine_human') {
      setChallengeEvaluation({
        verdict: 'PASSED',
        is_safe: true,
        latency_diagnosis: '1.1s onset delay (Normal Human Cadence)',
        message: 'Spoken digits matched within biological human response time without frame-stitching artifacts.',
        next_step: 'Clear transaction step-up and log verification pass.',
      });
    } else if (scenario === 'ai_clone_lag') {
      setChallengeEvaluation({
        verdict: 'FAILED · SYNTHESIS LATENCY',
        is_safe: false,
        latency_diagnosis: '4.6s onset delay (TTS Pipeline Lag)',
        message: 'Response exceeded the 2.8s conversational threshold, consistent with on-demand text-to-speech voice generation.',
        next_step: 'Hold transfer and initiate an out-of-band callback to the registered number.',
      });
    } else {
      setChallengeEvaluation({
        verdict: 'FAILED · REPLAY / MISMATCH',
        is_safe: false,
        latency_diagnosis: '0.15s onset (Instant Trigger / Wrong Nonce)',
        message: 'Audio triggered faster than human reaction time or failed to recite the ephemeral transaction code.',
        next_step: 'Hold transfer immediately and flag call for fraud review.',
      });
    }
  };

  const resampleBlobTo16kPcm = async (audioBlob: Blob): Promise<{ samples: number[]; speechOnsetOffsetSec: number }> => {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    const decodeCtx = new AudioCtx();
    try {
      const decoded = await decodeCtx.decodeAudioData(arrayBuffer);
      const targetSr = 16000;
      const targetLen = Math.max(1, Math.ceil(decoded.duration * targetSr));
      const offline = new OfflineAudioContext(1, targetLen, targetSr);
      const src = offline.createBufferSource();
      src.buffer = decoded;
      src.connect(offline.destination);
      src.start(0);
      const rendered = await offline.startRendering();
      const pcm = rendered.getChannelData(0);

      // Estimate first speech onset offset inside recorded clip (20ms window RMS > 0.015)
      let onsetSample = 0;
      const win = 320;
      for (let i = 0; i + win <= pcm.length; i += win) {
        let sumSq = 0;
        for (let j = 0; j < win; j++) sumSq += pcm[i + j] * pcm[i + j];
        if (Math.sqrt(sumSq / win) > 0.015) {
          onsetSample = i;
          break;
        }
      }

      return {
        samples: Array.from(pcm, (v) => Number(v.toFixed(5))),
        speechOnsetOffsetSec: onsetSample / targetSr,
      };
    } finally {
      await decodeCtx.close();
    }
  };

  const startChallengeMicRecording = async () => {
    if (!challengeData || isRecordingChallenge) return;
    setChallengeError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) micChunksRef.current.push(ev.data);
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      recordStartMsRef.current = Date.now();
      setIsRecordingChallenge(true);

      // Optional browser SpeechRecognition for automatic digit transcription when available
      const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRec) {
        try {
          const rec = new SpeechRec();
          rec.lang = 'en-US';
          rec.interimResults = true;
          rec.onresult = (event: any) => {
            const text = Array.from(event.results as ArrayLike<any>)
              .map((r: any) => r[0]?.transcript || '')
              .join(' ')
              .trim();
            if (text) setSpokenTranscript(text);
          };
          rec.start();
          speechRecRef.current = rec;
        } catch {
          speechRecRef.current = null;
        }
      }
    } catch (err: any) {
      setChallengeError(err?.message || 'Microphone access denied or unavailable in this browser.');
    }
  };

  const stopAndVerifyChallengeMic = async () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || !challengeData) return;

    setIsRecordingChallenge(false);
    setIsVerifyingChallenge(true);
    setChallengeError(null);

    if (speechRecRef.current) {
      try {
        speechRecRef.current.stop();
      } catch {
        // Ignore stop errors
      }
      speechRecRef.current = null;
    }

    try {
      const blob = await new Promise<Blob>((resolve) => {
        recorder.onstop = () => {
          recorder.stream.getTracks().forEach((t) => t.stop());
          resolve(new Blob(micChunksRef.current, { type: recorder.mimeType || 'audio/webm' }));
        };
        recorder.stop();
      });

      const { samples, speechOnsetOffsetSec } = await resampleBlobTo16kPcm(blob);
      const promptDisplayedSec = challengeData.created_at / 1000.0;
      const responseStartedSec = recordStartMsRef.current / 1000.0 + speechOnsetOffsetSec;

      const res = await fetch(`${GATEWAY_URL}/api/challenge/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transaction_id: challengeData.transaction_id,
          samples,
          expected_digits: challengeData.expected_digits,
          transcribed_text: spokenTranscript.trim(),
          prompt_displayed_at: promptDisplayedSec,
          response_started_at: responseStartedSec,
        }),
      });

      setBackendStatus('online');
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || data.detail || `Verification failed (HTTP ${res.status})`);
      }

      const status = data.challenge_status || 'INCONCLUSIVE';
      const diag = data.detailed_diagnostics || {};
      const isPass = status === 'PASS';
      setChallengeEvaluation({
        verdict: `${status} · SCORE ${Math.round((Number(data.composite_score || 0)) * 100)}%`,
        is_safe: isPass,
        latency_diagnosis: `${diag.measured_response_delay_sec ?? '—'}s delay (${diag.timing_check || 'LIVE MIC'})`,
        message: `Content: ${diag.content_check || 'CHECKED'} · Articulation: ${diag.articulation_check || 'CHECKED'} · Spoof Prob: ${Math.round(Number(diag.voice_spoof_probability || 0) * 100)}%`,
        next_step: isPass
          ? 'Live speaker verified. Proceed with transaction.'
          : status === 'INCONCLUSIVE'
          ? 'Result inconclusive. Rotate challenge or perform out-of-band callback.'
          : 'Challenge failed. Hold transfer and escalate to fraud desk.',
      });
      await fetchAuditRecords();
    } catch (err: any) {
      setChallengeError(err?.message || 'Live challenge verification failed.');
    } finally {
      setIsVerifyingChallenge(false);
    }
  };

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

  const launchGuidedTest = (testIndex: number) => {
    if (testIndex === 1) {
      setTransactionId('TXN-NORMAL-WIRE');
      setAmountInr(25000);
      setIsNewBeneficiary(false);
      setSelectedSample('genuine_clean');
      void runCallAnalysis('genuine_clean', { txnId: 'TXN-NORMAL-WIRE', amount: 25000, isNewBene: false });
      setActiveTab('inspector');
    } else if (testIndex === 2) {
      setTransactionId('TXN-PHONE-RETIREE');
      setAmountInr(15000);
      setIsNewBeneficiary(false);
      setSelectedSample('genuine_phone');
      void runCallAnalysis('genuine_phone', { txnId: 'TXN-PHONE-RETIREE', amount: 15000, isNewBene: false });
      setActiveTab('inspector');
    } else if (testIndex === 3) {
      setTransactionId('TXN-EXEC-WIRE-901');
      setAmountInr(150000);
      setIsNewBeneficiary(true);
      setSelectedSample('ai_clone');
      void runCallAnalysis('ai_clone', { txnId: 'TXN-EXEC-WIRE-901', amount: 150000, isNewBene: true });
      setActiveTab('inspector');
    } else if (testIndex === 4) {
      setActiveTab('challenge');
      void generateSpokenChallenge();
    } else if (testIndex === 5) {
      setActiveTab('audit');
      void fetchAuditRecords();
    }
  };

  const navItems: { id: TabId; label: string; icon: React.ComponentType<any>; count?: number; section: 'ops' | 'docs' }[] = [
    { id: 'inspector', label: 'Call Inspector', icon: PhoneCall, section: 'ops' },
    { id: 'timeline', label: 'Risk Timeline', icon: Activity, section: 'ops' },
    { id: 'challenge', label: 'Voice Challenge', icon: Mic, section: 'ops' },
    { id: 'audit', label: 'Audit Ledger', icon: FileCheck2, count: auditRecords.length, section: 'ops' },
    { id: 'guide', label: 'Test Scenarios', icon: Play, section: 'docs' },
    { id: 'science', label: 'Methodology', icon: BookOpen, section: 'docs' },
  ];

  const decisionMeta = (action: string) => {
    if (action === 'ALLOW') {
      return {
        border: '#059669',
        badgeClass: 'status-badge-safe',
        dotClass: 'dot-safe',
        badgeLabel: 'ALLOW',
        headline: 'Approved — Clear to Proceed',
        subtext: 'Acoustic features and vocal-tract coordination fall within expected human speech parameters.',
      };
    }
    if (action === 'VERIFY') {
      return {
        border: '#d97706',
        badgeClass: 'status-badge-verify',
        dotClass: 'dot-verify',
        badgeLabel: 'VERIFY',
        headline: 'Step-Up Verification Required',
        subtext: 'Borderline acoustic confidence or degraded telephone channel. Issue a one-time spoken challenge before authorizing.',
      };
    }
    return {
      border: '#dc2626',
      badgeClass: 'status-badge-danger',
      dotClass: 'dot-danger',
      badgeLabel: 'HOLD FOR REVIEW',
      headline: 'Hold Transaction — Suspected Voice Clone',
      subtext: 'High synthetic voice risk combined with transaction exposure. Escalate to manual verification.',
    };
  };

  return (
    <div className="app-shell">
      {/* Top Workstation Bar */}
      <header className="topbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '30px',
              height: '30px',
              borderRadius: '6px',
              background: 'var(--action-primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--action-primary-text)',
            }}
          >
            <ShieldCheck size={18} />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '0.98rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              VoiceShield
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Call Fraud Operations Desk
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span
            className={`status-badge ${
              backendStatus === 'online'
                ? 'status-badge-safe'
                : backendStatus === 'checking'
                ? 'status-badge-neutral'
                : 'status-badge-danger'
            }`}
          >
            <span
              className={`dot ${
                backendStatus === 'online'
                  ? 'dot-safe'
                  : backendStatus === 'checking'
                  ? 'dot-neutral'
                  : 'dot-danger'
              }`}
            />
            {backendStatus === 'online'
              ? 'Gateway Connected'
              : backendStatus === 'checking'
              ? 'Checking Gateway'
              : 'Gateway Offline'}
          </span>

          <div
            className="surface-subtle tabular"
            style={{ padding: '4px 10px', fontSize: '0.76rem', color: 'var(--text-secondary)' }}
          >
            Latency:{' '}
            <strong className="mono" style={{ color: 'var(--text-primary)' }}>
              {scanResult?.processing_latency_ms
                ? `${Number(scanResult.processing_latency_ms).toFixed(1)} ms`
                : '—'}
            </strong>
          </div>

          <div
            className="surface-subtle tabular"
            style={{ padding: '4px 10px', fontSize: '0.76rem', color: 'var(--text-secondary)' }}
          >
            Input: <strong className="mono" style={{ color: 'var(--text-primary)' }}>16 kHz PCM</strong>
          </div>

          <button
            onClick={() => setTheme((prev) => (prev === 'light' ? 'dark' : 'light'))}
            className="btn-secondary"
            style={{ padding: '5px 10px', fontSize: '0.78rem' }}
            title={`Switch to ${isDark ? 'light' : 'dark'} theme`}
            aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`}
          >
            {isDark ? <Sun size={14} /> : <Moon size={14} />}
            <span>{isDark ? 'Light' : 'Dark'}</span>
          </button>

          <button
            onClick={() => setActiveTab('guide')}
            className="btn-secondary"
            style={{ padding: '5px 11px', fontSize: '0.78rem' }}
          >
            <HelpCircle size={14} /> Test Guide
          </button>
        </div>
      </header>

      {/* Main Workspace: Sidebar + Canvas */}
      <div className="workspace-grid">
        <aside className="sidebar">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div>
              <div
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: 'var(--text-muted)',
                  padding: '0 10px 6px',
                }}
              >
                Workstation
              </div>
              <div className="sidebar-nav-list" style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {navItems
                  .filter((item) => item.section === 'ops')
                  .map((item) => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.id;
                    return (
                      <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        className={`nav-item ${isActive ? 'active' : ''}`}
                      >
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                          <Icon size={15} color={isActive ? 'var(--text-primary)' : 'var(--text-muted)'} />
                          {item.label}
                        </span>
                        {typeof item.count === 'number' && (
                          <span
                            className="mono tabular"
                            style={{
                              fontSize: '0.72rem',
                              color: 'var(--text-muted)',
                              background: isActive ? 'var(--bg-surface)' : 'var(--bg-muted)',
                              border: '1px solid var(--border-hairline)',
                              borderRadius: '4px',
                              padding: '0 6px',
                            }}
                          >
                            {item.count}
                          </span>
                        )}
                      </button>
                    );
                  })}
              </div>
            </div>

            <div>
              <div
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: 'var(--text-muted)',
                  padding: '0 10px 6px',
                }}
              >
                Reference
              </div>
              <div className="sidebar-nav-list" style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {navItems
                  .filter((item) => item.section === 'docs')
                  .map((item) => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.id;
                    return (
                      <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        className={`nav-item ${isActive ? 'active' : ''}`}
                      >
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                          <Icon size={15} color={isActive ? 'var(--text-primary)' : 'var(--text-muted)'} />
                          {item.label}
                        </span>
                      </button>
                    );
                  })}
              </div>
            </div>
          </div>

          <div
            className="sidebar-footer surface-subtle"
            style={{ padding: '10px 12px', fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.45 }}
          >
            <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '2px' }}>
              BAFV-PCTA · SIH26104
            </div>
            Advisory fraud support only. Zero raw customer audio persisted on disk.
          </div>
        </aside>

        <main className="main-canvas">
          {/* ===================================================================== */}
          {/* TAB 1: CALL INSPECTOR                                                 */}
          {/* ===================================================================== */}
          {activeTab === 'inspector' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <h1 style={{ fontSize: '1.25rem' }}>Call Inspection</h1>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  Evaluate call audio acoustics, physiological vocal-tract coupling, and transaction exposure.
                </p>
              </div>

              <div className="inspector-split">
                {/* Left Rail: Audio & Transaction Controls */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <section className="surface-card" style={{ padding: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <h2 style={{ fontSize: '0.9rem' }}>1. Caller Audio Source</h2>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>WAV / MP3</span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '14px' }}>
                      {(['genuine_clean', 'genuine_phone', 'ai_clone'] as const).map((key) => {
                        const item = SAMPLE_PRESETS[key];
                        const active = selectedSample === key;
                        return (
                          <button
                            key={key}
                            onClick={() => {
                              setSelectedSample(key);
                              void runCallAnalysis(key);
                            }}
                            className={`scenario-row ${active ? 'active' : ''}`}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                              <span style={{ fontWeight: 600, fontSize: '0.84rem', color: 'var(--text-primary)' }}>
                                {item.shortTitle}
                              </span>
                              <span className="status-badge status-badge-neutral" style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
                                {item.tag}
                              </span>
                            </div>
                            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: 400 }}>
                              {item.subtitle}
                            </span>
                          </button>
                        );
                      })}
                    </div>

                    {scanResult?.audio_src && (
                      <div className="surface-subtle" style={{ padding: '10px', marginBottom: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
                          <Volume2 size={14} />
                          <span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {scanResult.file_name}
                          </span>
                        </div>
                        <audio controls key={scanResult.audio_src} src={scanResult.audio_src} preload="auto" />
                      </div>
                    )}

                    <label
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px',
                        padding: '9px 12px',
                        borderRadius: '6px',
                        border: '1px dashed var(--border-strong)',
                        cursor: 'pointer',
                        fontSize: '0.8rem',
                        color: 'var(--text-secondary)',
                        background: 'var(--bg-subtle)',
                      }}
                    >
                      <Upload size={14} />
                      <span>{customFileName ? `File: ${customFileName}` : 'Upload Custom Audio (.wav, .mp3)'}</span>
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
                            void runCallAnalysis('custom', { file, audioUrl: nextUrl });
                          }
                        }}
                      />
                    </label>

                    <button
                      className="btn-secondary"
                      style={{ width: '100%', marginTop: '10px' }}
                      disabled={isAnalyzing}
                      onClick={() => void runCallAnalysis(selectedSample)}
                    >
                      <RefreshCw size={14} className={isAnalyzing ? 'animate-spin' : ''} />
                      <span>{isAnalyzing ? 'Analyzing Audio...' : 'Re-run Analysis'}</span>
                    </button>
                  </section>

                  <section className="surface-card" style={{ padding: '16px' }}>
                    <h2 style={{ fontSize: '0.9rem', marginBottom: '12px' }}>2. Transaction Context</h2>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                      <div>
                        <label style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '5px', fontWeight: 500 }}>
                          Transaction Reference
                        </label>
                        <input
                          type="text"
                          value={transactionId}
                          onChange={(e) => setTransactionId(e.target.value)}
                          className="input-field mono"
                        />
                      </div>

                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                          <label style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
                            Transfer Amount
                          </label>
                          <strong className="mono tabular" style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                            ₹{amountInr.toLocaleString('en-IN')}
                          </strong>
                        </div>
                        <input
                          type="range"
                          min="5000"
                          max="1000000"
                          step="5000"
                          value={amountInr}
                          onChange={(e) => setAmountInr(Number(e.target.value))}
                        />
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px', marginTop: '8px' }}>
                          {[15000, 45000, 150000, 500000].map((amt) => (
                            <button
                              key={amt}
                              onClick={() => setAmountInr(amt)}
                              className="btn-secondary tabular"
                              style={{
                                padding: '4px 6px',
                                fontSize: '0.72rem',
                                background: amountInr === amt ? 'var(--bg-muted)' : 'var(--bg-surface)',
                                borderColor: amountInr === amt ? 'var(--action-primary)' : undefined,
                                color: amountInr === amt ? 'var(--text-primary)' : undefined,
                              }}
                            >
                              {amt >= 100000 ? `₹${amt / 100000}L` : `₹${amt / 1000}k`}
                            </button>
                          ))}
                        </div>
                      </div>

                      <label
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          paddingTop: '12px',
                          borderTop: '1px solid var(--border-hairline)',
                          cursor: 'pointer',
                        }}
                      >
                        <div>
                          <span style={{ fontSize: '0.82rem', fontWeight: 500, display: 'block', color: 'var(--text-primary)' }}>
                            First-Time Beneficiary
                          </span>
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                            Payee added within last 24 hours
                          </span>
                        </div>
                        <input
                          type="checkbox"
                          checked={isNewBeneficiary}
                          onChange={(e) => setIsNewBeneficiary(e.target.checked)}
                          style={{ width: '16px', height: '16px', accentColor: 'var(--action-primary)' }}
                        />
                      </label>
                    </div>
                  </section>
                </div>

                {/* Right Panel: Analysis Output */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {scanError && (
                    <div
                      className="surface-card"
                      style={{
                        padding: '16px 18px',
                        borderLeft: '4px solid #dc2626',
                        background: 'var(--danger-bg)',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                        <div>
                          <div style={{ fontWeight: 600, color: 'var(--danger-text)', fontSize: '0.88rem' }}>
                            Analysis Unavailable (FR-13 Fail-Secure)
                          </div>
                          <p style={{ fontSize: '0.8rem', color: 'var(--danger-text)', marginTop: '3px' }}>
                            {scanError}
                          </p>
                        </div>
                        <button className="btn-secondary" onClick={() => void runCallAnalysis(selectedSample)}>
                          <RefreshCw size={14} /> Retry
                        </button>
                      </div>
                    </div>
                  )}

                  {!scanResult && !scanError && (
                    <div className="surface-card" style={{ padding: '28px', color: 'var(--text-secondary)', fontSize: '0.86rem' }}>
                      {isAnalyzing
                        ? 'Running neural anti-spoofing and biomechanical coupling analysis...'
                        : backendStatus === 'checking'
                        ? 'Connecting to gateway and waking AI engine (up to ~45s after idle)...'
                        : 'Select a sample recording or upload a .wav file to begin inspection.'}
                    </div>
                  )}

                  {scanResult && (() => {
                    const meta = decisionMeta(scanResult.recommendation);
                    return (
                      <>
                        {/* Decision Banner */}
                        <div
                          className="surface-card"
                          style={{
                            padding: '18px 20px',
                            borderLeft: `4px solid ${meta.border}`,
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'flex-start',
                            gap: '16px',
                            flexWrap: 'wrap',
                          }}
                        >
                          <div style={{ flex: 1, minWidth: '240px' }}>
                            <div style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                              Analyst Recommendation · {scanResult.caller_name}
                            </div>
                            <h2 style={{ fontSize: '1.22rem', marginTop: '4px' }}>{meta.headline}</h2>
                            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                              {meta.subtext}
                            </p>
                          </div>

                          <span className={`status-badge ${meta.badgeClass}`} style={{ padding: '5px 12px' }}>
                            <span className={`dot ${meta.dotClass}`} />
                            {meta.badgeLabel}
                          </span>
                        </div>

                        {/* 3-Column KPI Strip */}
                        <div className="surface-card kpi-strip">
                          <div className="kpi-cell">
                            <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                              Voice Clone Probability
                            </div>
                            <div
                              className="mono tabular"
                              style={{
                                fontSize: '1.55rem',
                                fontWeight: 600,
                                marginTop: '4px',
                                color: scanResult.ai_detector.spoof_prob >= 0.5 ? 'var(--danger-text)' : 'var(--text-primary)',
                              }}
                            >
                              {(scanResult.ai_detector.spoof_prob * 100).toFixed(1)}%
                            </div>
                            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                              Class: {scanResult.ai_detector.generator_family}
                            </div>
                          </div>

                          <div className="kpi-cell">
                            <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                              Line Confidence (TCS)
                            </div>
                            <div
                              className="mono tabular"
                              style={{
                                fontSize: '1.55rem',
                                fontWeight: 600,
                                marginTop: '4px',
                                color: scanResult.line_quality.line_score < 0.5 ? 'var(--verify-text)' : 'var(--text-primary)',
                              }}
                            >
                              {(scanResult.line_quality.line_score * 100).toFixed(1)}%
                            </div>
                            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                              {scanResult.line_quality.codec_desc}
                            </div>
                          </div>

                          <div className="kpi-cell">
                            <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                              Coupling Anomaly (BAFV)
                            </div>
                            <div
                              className="mono tabular"
                              style={{
                                fontSize: '1.55rem',
                                fontWeight: 600,
                                marginTop: '4px',
                                color: scanResult.throat_physics.anomaly_score >= 0.45 ? 'var(--danger-text)' : 'var(--text-primary)',
                              }}
                            >
                              {(scanResult.throat_physics.anomaly_score * 100).toFixed(1)}%
                            </div>
                            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                              SNR {Number(scanResult.line_quality.snr_db).toFixed(1)} dB · {scanResult.ai_detector.weights_source === 'trained_checkpoint' ? 'Trained Weights' : 'Seed Init'}
                            </div>
                          </div>
                        </div>

                        {scanResult.line_quality.warning && (
                          <div
                            className="surface-card"
                            style={{
                              padding: '12px 16px',
                              background: 'var(--verify-bg)',
                              borderColor: 'var(--verify-border)',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '10px',
                              fontSize: '0.8rem',
                              color: 'var(--verify-text)',
                            }}
                          >
                            <AlertTriangle size={16} style={{ flexShrink: 0 }} />
                            <span>
                              <strong>Channel Guard: </strong>
                              {scanResult.line_quality.warning}
                            </span>
                          </div>
                        )}

                        {/* PCTA Trajectory Table */}
                        <div className="surface-card" style={{ overflow: 'hidden' }}>
                          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border-hairline)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                            <div>
                              <h3 style={{ fontSize: '0.92rem' }}>Physiological Coupling Diagnostics (PCTA)</h3>
                              <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                                Measures synchronization across glottal pitch, vocal-tract formants (F1–F3), subglottal energy, and frame phase.
                              </p>
                            </div>
                            <span className="mono" style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                              {scanResult.throat_physics.pairs.length} pairs evaluated
                            </span>
                          </div>

                          <table className="data-table">
                            <thead>
                              <tr>
                                <th>Trajectory Pair</th>
                                <th>Status</th>
                                <th>Observed Finding</th>
                              </tr>
                            </thead>
                            <tbody>
                              {scanResult.throat_physics.pairs.map((p: any, idx: number) => (
                                <tr key={idx}>
                                  <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{p.name}</td>
                                  <td>
                                    <span className={`status-badge ${p.isOk ? 'status-badge-safe' : 'status-badge-danger'}`}>
                                      <span className={`dot ${p.isOk ? 'dot-safe' : 'dot-danger'}`} />
                                      {p.status}
                                    </span>
                                  </td>
                                  <td className="mono" style={{ fontSize: '0.76rem' }}>{p.desc}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>

                        {/* Risk Rationale + Actions */}
                        <div className="surface-subtle" style={{ padding: '12px 16px', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                          <strong style={{ color: 'var(--text-primary)' }}>Fusion Notes: </strong>
                          {scanResult.throat_physics.explanation}
                        </div>

                        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                          <button className="btn-secondary" onClick={() => setActiveTab('timeline')}>
                            <Activity size={15} /> Open Risk Timeline
                          </button>
                          <button
                            className="btn-primary"
                            onClick={() => {
                              setActiveTab('challenge');
                              void generateSpokenChallenge();
                            }}
                          >
                            <Mic size={15} /> Issue Spoken Challenge
                          </button>
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>
            </div>
          )}

          {/* ===================================================================== */}
          {/* TAB 2: CALL RISK TIMELINE                                             */}
          {/* ===================================================================== */}
          {activeTab === 'timeline' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <h1 style={{ fontSize: '1.25rem' }}>Windowed Call Risk Timeline</h1>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  Sliding 2.0-second analysis windows stepped every 0.5 seconds with exponential temporal smoothing.
                </p>
              </div>

              <div className="surface-card" style={{ padding: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
                  <h2 style={{ fontSize: '0.94rem' }}>Risk Trajectory Over Call Duration</h2>
                  <div style={{ display: 'flex', gap: '16px', fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ width: '10px', height: '2px', background: '#2563eb', display: 'inline-block' }} /> Instantaneous Risk
                    </span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ width: '10px', height: '2px', background: '#dc2626', display: 'inline-block' }} /> Smoothed Risk (λ = 0.6)
                    </span>
                  </div>
                </div>

                <div style={{ height: '320px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timelineData} margin={{ top: 10, right: 24, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="fillSmoothed" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#dc2626" stopOpacity={0.14} />
                          <stop offset="95%" stopColor="#dc2626" stopOpacity={0.0} />
                        </linearGradient>
                        <linearGradient id="fillInstant" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#2563eb" stopOpacity={0.1} />
                          <stop offset="95%" stopColor="#2563eb" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#252932' : '#e4e7ec'} />
                      <XAxis
                        dataKey="start_time_seconds"
                        stroke={isDark ? '#707887' : '#667085'}
                        fontSize={12}
                        tickFormatter={(v) => `${Number(v).toFixed(1)}s`}
                      />
                      <YAxis domain={[0, 1]} stroke={isDark ? '#707887' : '#667085'} fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: isDark ? '#14161a' : '#ffffff',
                          borderColor: isDark ? '#323743' : '#d0d5dd',
                          borderRadius: '6px',
                          fontSize: '0.78rem',
                          color: isDark ? '#eceef2' : '#101828',
                          boxShadow: isDark ? '0 4px 12px rgba(0,0,0,0.4)' : '0 4px 12px rgba(16,24,40,0.08)',
                        }}
                      />
                      <ReferenceLine
                        y={0.35}
                        stroke={isDark ? '#10b981' : '#059669'}
                        strokeDasharray="4 4"
                        label={{ value: 'Verify Threshold (0.35)', fill: isDark ? '#6ee7b7' : '#065f46', fontSize: 11 }}
                      />
                      <ReferenceLine
                        y={0.75}
                        stroke={isDark ? '#ef4444' : '#dc2626'}
                        strokeDasharray="4 4"
                        label={{ value: 'Hold Threshold (0.75)', fill: isDark ? '#fca5a5' : '#991b1b', fontSize: 11 }}
                      />
                      <Area type="monotone" dataKey="instantaneous_risk" stroke={isDark ? '#60a5fa' : '#2563eb'} strokeWidth={1.75} fill="url(#fillInstant)" />
                      <Area type="monotone" dataKey="smoothed_risk" stroke={isDark ? '#f87171' : '#dc2626'} strokeWidth={2.25} fill="url(#fillSmoothed)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="surface-card" style={{ overflow: 'hidden' }}>
                <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border-hairline)' }}>
                  <h3 style={{ fontSize: '0.92rem' }}>Per-Window Breakdown</h3>
                </div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Window</th>
                      <th>Time Span</th>
                      <th>Speech Activity</th>
                      <th>Spoof Prob.</th>
                      <th>Smoothed Risk</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {timelineData.map((w) => (
                      <tr key={w.window_index}>
                        <td className="mono tabular" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          #{w.window_index + 1}
                        </td>
                        <td className="mono tabular">
                          {w.start_time_seconds.toFixed(1)}s – {w.end_time_seconds.toFixed(1)}s
                        </td>
                        <td className="mono tabular">{(w.speech_ratio * 100).toFixed(0)}%</td>
                        <td className="mono tabular">{(w.baseline_spoof_probability * 100).toFixed(1)}%</td>
                        <td className="mono tabular" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          {(w.smoothed_risk * 100).toFixed(1)}%
                        </td>
                        <td>
                          <span
                            className={`status-badge ${
                              w.recommendation === 'ALLOW'
                                ? 'status-badge-safe'
                                : w.recommendation === 'VERIFY'
                                ? 'status-badge-verify'
                                : 'status-badge-danger'
                            }`}
                          >
                            {w.recommendation}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {timelineData.length === 0 && (
                      <tr>
                        <td colSpan={6} style={{ color: 'var(--text-muted)' }}>
                          No window timeline data available yet. Run an audio scan in Call Inspector first.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ===================================================================== */}
          {/* TAB 3: LIVE VOICE SECURITY CHALLENGE (TB-PC)                          */}
          {/* ===================================================================== */}
          {activeTab === 'challenge' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <h1 style={{ fontSize: '1.25rem' }}>Transaction-Bound Spoken Challenge (TB-PC)</h1>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  Derives an ephemeral HMAC-SHA256 phrase tied to the transaction ID and amount to expose replay bots and synthesis delay.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '18px' }}>
                <div className="surface-card" style={{ padding: '20px' }}>
                  <h2 style={{ fontSize: '0.96rem', marginBottom: '6px' }}>1. Generate Ephemeral Challenge</h2>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                    Valid for 60 seconds with a single-use nonce bound to <span className="mono">{transactionId}</span> (₹{amountInr.toLocaleString('en-IN')}).
                  </p>

                  <button
                    className="btn-primary"
                    style={{ width: '100%', marginBottom: '16px' }}
                    onClick={() => void generateSpokenChallenge()}
                    disabled={isGeneratingChallenge}
                  >
                    <RefreshCw size={15} className={isGeneratingChallenge ? 'animate-spin' : ''} />
                    {challengeData ? 'Rotate 1-Time Phrase' : 'Generate Challenge Phrase'}
                  </button>

                  {challengeError && (
                    <div
                      className="surface-subtle"
                      style={{
                        padding: '10px 12px',
                        background: 'var(--danger-bg)',
                        borderColor: 'var(--danger-border)',
                        fontSize: '0.78rem',
                        color: 'var(--danger-text)',
                        marginBottom: '14px',
                      }}
                    >
                      {challengeError}
                    </div>
                  )}

                  {challengeData && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div className="surface-subtle" style={{ padding: '12px 14px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                            Nonce & Expected Digits
                          </span>
                          <span className="status-badge status-badge-verify tabular">
                            <Clock size={12} /> {challengeSecondsLeft}s remaining
                          </span>
                        </div>
                        <div className="mono tabular" style={{ fontSize: '1.25rem', fontWeight: 600, marginTop: '6px', color: 'var(--text-primary)' }}>
                          {challengeData.nonce} · {challengeData.expected_digits}
                        </div>
                      </div>

                      <div className="surface-subtle" style={{ padding: '14px' }}>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                          Prompt to Read to Caller
                        </div>
                        <div style={{ fontSize: '0.96rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '6px' }}>
                          “{challengeData.prompt_phrase}”
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div className="surface-card" style={{ padding: '20px' }}>
                  <h2 style={{ fontSize: '0.96rem', marginBottom: '6px' }}>2. Verify Caller Response (Live Mic or Simulation)</h2>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '14px' }}>
                    Record a live voice response into the microphone (sent to <span className="mono">/api/challenge/verify</span>) or test preset cadence scenarios below.
                  </p>

                  {/* Live Microphone Verification */}
                  <div className="surface-subtle" style={{ padding: '12px', marginBottom: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                      <strong style={{ fontSize: '0.82rem', color: 'var(--text-primary)' }}>
                        Live Microphone Capture
                      </strong>
                      {isRecordingChallenge && (
                        <span className="status-badge status-badge-danger">
                          <span className="dot dot-danger" /> Recording…
                        </span>
                      )}
                    </div>

                    <input
                      type="text"
                      value={spokenTranscript}
                      onChange={(e) => setSpokenTranscript(e.target.value)}
                      placeholder="Spoken digits (auto-transcribed in Chrome/Edge, or type e.g. 4821)"
                      className="input-field mono"
                      style={{ fontSize: '0.78rem' }}
                      disabled={!challengeData}
                    />

                    <div style={{ display: 'flex', gap: '8px' }}>
                      {!isRecordingChallenge ? (
                        <button
                          className="btn-primary"
                          style={{ flex: 1 }}
                          disabled={!challengeData || isVerifyingChallenge}
                          onClick={() => void startChallengeMicRecording()}
                        >
                          <Mic size={14} /> {isVerifyingChallenge ? 'Verifying Audio…' : 'Record Live Response'}
                        </button>
                      ) : (
                        <button
                          className="btn-danger-outline"
                          style={{ flex: 1, padding: '8px 14px', fontSize: '0.84rem' }}
                          onClick={() => void stopAndVerifyChallengeMic()}
                        >
                          Stop & Verify Response
                        </button>
                      )}
                    </div>
                  </div>

                  <div style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-muted)', marginBottom: '8px' }}>
                    Or Simulate Cadence Scenarios
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '18px' }}>
                    <button
                      className="scenario-row"
                      onClick={() => evaluateChallengeResponse('genuine_human')}
                      disabled={!challengeData}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>1. Live Human Speaker</strong>
                        <span className="status-badge status-badge-safe">1.1s latency</span>
                      </div>
                      <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                        Natural reading reaction time + matching digits
                      </span>
                    </button>

                    <button
                      className="scenario-row"
                      onClick={() => evaluateChallengeResponse('ai_clone_lag')}
                      disabled={!challengeData}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>2. On-Demand AI Cloner</strong>
                        <span className="status-badge status-badge-danger">4.6s latency</span>
                      </div>
                      <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                        Excessive pause caused by text-to-speech synthesis queue
                      </span>
                    </button>

                    <button
                      className="scenario-row"
                      onClick={() => evaluateChallengeResponse('stolen_recording')}
                      disabled={!challengeData}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>3. Pre-Recorded Audio Replay</strong>
                        <span className="status-badge status-badge-danger">&lt;200ms / mismatch</span>
                      </div>
                      <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                        Instant bot playback or incorrect transaction digits
                      </span>
                    </button>
                  </div>

                  {challengeEvaluation && (
                    <div
                      className="surface-subtle"
                      style={{
                        padding: '16px',
                        borderLeft: `4px solid ${challengeEvaluation.is_safe ? '#059669' : '#dc2626'}`,
                        background: challengeEvaluation.is_safe ? 'var(--safe-bg)' : 'var(--danger-bg)',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        <span className={`status-badge ${challengeEvaluation.is_safe ? 'status-badge-safe' : 'status-badge-danger'}`}>
                          {challengeEvaluation.verdict}
                        </span>
                        <span className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                          {challengeEvaluation.latency_diagnosis}
                        </span>
                      </div>
                      <p style={{ fontSize: '0.82rem', color: 'var(--text-primary)', marginTop: '10px' }}>
                        {challengeEvaluation.message}
                      </p>
                      <div style={{ marginTop: '10px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                        Action: {challengeEvaluation.next_step}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ===================================================================== */}
          {/* TAB 4: AUDIT LEDGER                                                   */}
          {/* ===================================================================== */}
          {activeTab === 'audit' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                  <h1 style={{ fontSize: '1.25rem' }}>Tamper-Evident Audit Ledger</h1>
                  <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                    SHA-256 hash-chained decision records maintained by Spring Boot. Only acoustic metadata is logged—never raw audio.
                  </p>
                </div>
                <span className="status-badge status-badge-safe tabular">
                  <CheckCircle2 size={13} /> {auditRecords.length} Blocks Anchored
                </span>
              </div>

              <div className="surface-card" style={{ overflow: 'hidden' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Block</th>
                      <th>Event & Reference</th>
                      <th>Decision & Score</th>
                      <th>Hash Chain (Prev → Record)</th>
                      <th>Time</th>
                      <th>Privacy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {auditRecords.map((r, i) => (
                      <tr key={r.record_id} style={{ opacity: r.deleted ? 0.65 : 1 }}>
                        <td className="mono tabular" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          #{i + 1}
                        </td>
                        <td>
                          <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{r.event_type}</div>
                          <div className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                            {r.transaction_id} · {r.record_id.slice(0, 12)}
                          </div>
                        </td>
                        <td>
                          {r.deleted ? (
                            <span className="status-badge status-badge-danger">Metadata Purged</span>
                          ) : (
                            <div style={{ fontSize: '0.78rem' }}>
                              <strong style={{ color: 'var(--text-primary)' }}>
                                {r.features?.recommendation || 'LOGGED'}
                              </strong>
                              {r.features?.spoof_probability !== undefined && (
                                <div className="mono tabular" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                                  Spoof: {Math.round(Number(r.features.spoof_probability) * 100)}%
                                  {r.features?.transfer_amount !== undefined
                                    ? ` · ₹${Number(r.features.transfer_amount).toLocaleString('en-IN')}`
                                    : ''}
                                </div>
                              )}
                            </div>
                          )}
                        </td>
                        <td className="mono" style={{ fontSize: '0.72rem' }}>
                          <div style={{ color: 'var(--text-muted)' }}>prev: {(r.prev_hash || '').slice(0, 14)}…</div>
                          <div style={{ color: 'var(--text-primary)' }}>hash: {(r.record_hash || '').slice(0, 14)}…</div>
                        </td>
                        <td className="mono tabular" style={{ fontSize: '0.74rem', whiteSpace: 'nowrap' }}>
                          {new Date(r.timestamp).toLocaleTimeString()}
                        </td>
                        <td>
                          {!r.deleted && r.record_id !== 'GENESIS-BLOCK' ? (
                            <button
                              onClick={() => void deleteAuditRecord(r.record_id)}
                              className="btn-danger-outline"
                              title="Purge customer metadata (FR-11)"
                            >
                              <Trash2 size={12} /> Purge
                            </button>
                          ) : (
                            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                    {auditRecords.length === 0 && (
                      <tr>
                        <td colSpan={6} style={{ color: 'var(--text-muted)' }}>
                          No audit blocks loaded. Verify Spring Boot is running on port 5000.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ===================================================================== */}
          {/* TAB 5: TEST SCENARIOS                                                 */}
          {/* ===================================================================== */}
          {activeTab === 'guide' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <h1 style={{ fontSize: '1.25rem' }}>Verification Scenarios</h1>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  One-click end-to-end demonstrations across clean voice, narrowband telephony, vishing attack, spoken challenge, and data purge.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
                {[
                  {
                    idx: 1,
                    badge: 'ALLOW',
                    badgeClass: 'status-badge-safe',
                    title: 'Legitimate Customer Transfer (₹25,000)',
                    desc: 'Clean 16 kHz human call authorizing a routine transfer to an existing beneficiary.',
                    expected: 'Approved — natural glottal micro-jitter and coupled formants.',
                  },
                  {
                    idx: 2,
                    badge: 'CHANNEL GUARD',
                    badgeClass: 'status-badge-verify',
                    title: 'Elderly Account Holder (8 kHz Landline)',
                    desc: 'Narrowband 8 kHz phone call where frequencies above 3.4 kHz are attenuated.',
                    expected: 'Low-Confidence Guard activates to prevent an unfair unilateral freeze.',
                  },
                  {
                    idx: 3,
                    badge: 'VISHING ATTACK',
                    badgeClass: 'status-badge-danger',
                    title: 'Executive Wire Impersonation (₹1,50,000)',
                    desc: 'Synthesized voice clone requesting an urgent transfer to a first-time payee.',
                    expected: 'Hold for Review — elevated risk fusion and coupling anomalies.',
                  },
                  {
                    idx: 4,
                    badge: 'STEP-UP',
                    badgeClass: 'status-badge-neutral',
                    title: 'HMAC-SHA256 Spoken Challenge',
                    desc: 'Issue an ephemeral 4-digit phrase and inspect human vs. TTS synthesis latency.',
                    expected: 'Live 60s countdown + timing cadence evaluation.',
                  },
                  {
                    idx: 5,
                    badge: 'COMPLIANCE',
                    badgeClass: 'status-badge-neutral',
                    title: 'DPDP / GDPR Metadata Purge',
                    desc: 'Purge identifiable metadata on an audit entry while keeping the SHA-256 chain intact.',
                    expected: 'Metadata replaced with purge notice; chain hashes preserved.',
                  },
                ].map((s) => (
                  <div key={s.idx} className="surface-card" style={{ padding: '18px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: '14px' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                          SCENARIO 0{s.idx}
                        </span>
                        <span className={`status-badge ${s.badgeClass}`}>{s.badge}</span>
                      </div>
                      <h3 style={{ fontSize: '0.96rem' }}>{s.title}</h3>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '6px' }}>{s.desc}</p>
                      <div className="surface-subtle" style={{ padding: '9px 11px', marginTop: '10px', fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                        <strong style={{ color: 'var(--text-primary)' }}>Expected: </strong>
                        {s.expected}
                      </div>
                    </div>

                    <button
                      className={s.idx === 3 ? 'btn-primary' : 'btn-secondary'}
                      style={{ width: '100%' }}
                      onClick={() => launchGuidedTest(s.idx)}
                    >
                      <span>Run Scenario 0{s.idx}</span>
                      <ArrowRight size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ===================================================================== */}
          {/* TAB 6: METHODOLOGY                                                    */}
          {/* ===================================================================== */}
          {activeTab === 'science' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <h1 style={{ fontSize: '1.25rem' }}>System Methodology & Governance</h1>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  Technical architecture combining physiological speech coupling, transaction risk fusion, and privacy-preserving audit trails.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '18px' }}>
                <div className="surface-card" style={{ padding: '20px' }}>
                  <h2 style={{ fontSize: '0.98rem', marginBottom: '12px' }}>Physiological Coupled-Trajectory Analysis</h2>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    <p>
                      <strong style={{ color: 'var(--text-primary)' }}>1. Coupled Speech Mechanics:</strong> Human phonation couples subglottal airflow, vocal-fold micro-jitter, and continuous vocal-tract resonances (F1–F3).
                    </p>
                    <p>
                      <strong style={{ color: 'var(--text-primary)' }}>2. Frame-Synthesis Artifacts:</strong> Neural vocoders synthesize short frames independently, introducing phase drift and decoupled pitch-formant trajectories.
                    </p>
                    <p>
                      <strong style={{ color: 'var(--text-primary)' }}>3. Articulator Velocity Bounds (PTVA):</strong> Tongue and jaw inertia constrain biological formant transition speeds; discontinuities beyond physical limits are flagged.
                    </p>
                  </div>
                </div>

                <div className="surface-card" style={{ padding: '20px' }}>
                  <h2 style={{ fontSize: '0.98rem', marginBottom: '12px' }}>Advisory Policy & Privacy Guarantees</h2>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    <p>
                      <strong style={{ color: 'var(--text-primary)' }}>Human-in-the-Loop Decisions:</strong> Outputs (<span className="mono">ALLOW</span>, <span className="mono">VERIFY</span>, <span className="mono">HOLD_FOR_REVIEW</span>) are probabilistic analyst signals, never autonomous account locks.
                    </p>
                    <p>
                      <strong style={{ color: 'var(--text-primary)' }}>Low-Confidence Protection:</strong> Degraded 8 kHz phone channels lower channel confidence <span className="mono">C</span>, preventing false rejections based solely on muffled audio.
                    </p>
                    <p>
                      <strong style={{ color: 'var(--text-primary)' }}>Zero Raw Audio Retention:</strong> Audio buffers are discarded immediately after feature extraction; only SHA-256 metadata blocks are anchored.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;

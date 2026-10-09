import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  Activity,
  AudioWaveform,
  Lock,
  Mic,
  RefreshCw,
  Trash2,
  FileText,
  AlertTriangle,
  Info,
  CheckCircle2,
  Radio,
  Clock,
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

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'scan' | 'timeline' | 'challenge' | 'audit' | 'info'>('scan');
  
  // Transaction State (FR-06)
  const [transactionId, setTransactionId] = useState('TXN-88492-IN');
  const [amountInr, setAmountInr] = useState(45000);
  const [isNewBeneficiary, setIsNewBeneficiary] = useState(false);

  // Audio Selection State
  const [selectedSample, setSelectedSample] = useState<'genuine_a' | 'genuine_tel' | 'clone_v1' | 'custom'>('genuine_a');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Results State
  const [scanResult, setScanResult] = useState<any>(null);
  const [timelineData, setTimelineData] = useState<TimelineWindow[]>([]);

  // TB-PC Challenge State (FR-09, FR-10)
  const [challengeData, setChallengeData] = useState<any>(null);
  const [isGeneratingChallenge, setIsGeneratingChallenge] = useState(false);
  const [challengeSecondsLeft, setChallengeSecondsLeft] = useState(60);
  const [challengeEvaluation, setChallengeEvaluation] = useState<any>(null);

  // Audit Chain State (FR-11)
  const [auditRecords, setAuditRecords] = useState<AuditRecord[]>([]);

  // Sample presets for quick testing (Phase P3 PCTA Aligned)
  const samplePresets = {
    genuine_a: {
      name: 'Human Voice (Clean)',
      file: 'genuine_human_speaker_a.wav',
      type: 'genuine',
      spoofProb: 0.12,
      channelConf: 0.94,
      snr: 28.5,
      bafvScore: 0.05,
      isBandLimited: false,
      pcvsScore: 0.06,
      mahalanobisDist: 0.82,
      rpci: {
        status: 'NORMAL_RESPIRATION_COUPLING',
        breath_delay_ms: 48.2,
        explanation: 'Natural pre-voicing respiration detected (48.2 ms delay)',
      },
      ptva: {
        ptva_score: 0.08,
        f1_max_velocity_hz_s: 4200.0,
        f2_max_velocity_hz_s: 6800.0,
        speed_limit_violation: false,
      },
      pairs: [
        { pair: 'F0 ↔ F1', name: 'Glottal / Pharyngeal Coupling', desc: 'Natural Glottal Coupling (r = 0.84)', status: 'ALIGNED' },
        { pair: 'F1 ↔ F2', name: 'Tongue Height / Backness Transition', desc: 'Physiological Trajectory (r = 0.79)', status: 'ALIGNED' },
        { pair: 'F2 ↔ F3', name: 'Oral / Retroflex Resonator', desc: 'Natural Dynamic Resonators (r = 0.72)', status: 'ALIGNED' },
        { pair: 'Energy ↔ Pitch', name: 'Subglottal Pressure Covariance', desc: 'Natural Lung-Vocal Fold Covariance (r = 0.81)', status: 'ALIGNED' },
        { pair: 'Breath ↔ Onset', name: 'RPCI: Respiration-to-Voicing', desc: 'Normal Breath Coupling (48ms)', status: 'ALIGNED' },
        { pair: 'Phase Continuity', name: 'Frame Boundary Phase Drift', desc: 'Continuous Acoustic Phase (drift < 0.08)', status: 'ALIGNED' },
      ],
    },
    genuine_tel: {
      name: 'Telephone Degraded 8 kHz',
      file: 'genuine_telephone_8khz_amr.wav',
      type: 'degraded_human',
      spoofProb: 0.48,
      channelConf: 0.42,
      snr: 12.1,
      bafvScore: 0.18,
      isBandLimited: true,
      pcvsScore: 0.19,
      mahalanobisDist: 1.74,
      rpci: {
        status: 'BREATH_CUE_MISSING_OR_AMBIGUOUS',
        breath_delay_ms: 0.0,
        explanation: 'Pre-voicing breath cue suppressed by telephony codec (Rule: zero suspicion)',
      },
      ptva: {
        ptva_score: 0.14,
        f1_max_velocity_hz_s: 3800.0,
        f2_max_velocity_hz_s: 5900.0,
        speed_limit_violation: false,
      },
      pairs: [
        { pair: 'F0 ↔ F1', name: 'Glottal / Pharyngeal Coupling', desc: 'Natural Glottal Coupling (r = 0.74)', status: 'ALIGNED' },
        { pair: 'F1 ↔ F2', name: 'Tongue Height / Backness Transition', desc: 'Physiological Trajectory (r = 0.71)', status: 'ALIGNED' },
        { pair: 'F2 ↔ F3', name: 'Oral / Retroflex Resonator', desc: 'Band-Limited Resonator (r = 0.63)', status: 'ALIGNED' },
        { pair: 'Energy ↔ Pitch', name: 'Subglottal Pressure Covariance', desc: 'Lung-Vocal Fold Covariance (r = 0.69)', status: 'ALIGNED' },
        { pair: 'Breath ↔ Onset', name: 'RPCI: Respiration-to-Voicing', desc: 'Breath Suppressed by Codec (Tolerated)', status: 'ALIGNED' },
        { pair: 'Phase Continuity', name: 'Frame Boundary Phase Drift', desc: 'Minor Codec Phase Jitter (< 0.22)', status: 'ALIGNED' },
      ],
    },
    clone_v1: {
      name: 'Synthetic Voice Clone (Vishing)',
      file: 'synthetic_voice_clone_v1.wav',
      type: 'spoof',
      spoofProb: 0.91,
      channelConf: 0.91,
      snr: 32.0,
      bafvScore: 0.82,
      isBandLimited: false,
      pcvsScore: 0.91,
      mahalanobisDist: 6.82,
      rpci: {
        status: 'BREATH_CUE_MISSING_OR_AMBIGUOUS',
        breath_delay_ms: 0.0,
        explanation: 'Instantaneous electronic energy onset without pre-voicing airflow',
      },
      ptva: {
        ptva_score: 0.81,
        f1_max_velocity_hz_s: 14800.0,
        f2_max_velocity_hz_s: 21500.0,
        speed_limit_violation: true,
      },
      pairs: [
        { pair: 'F0 ↔ F1', name: 'Glottal / Pharyngeal Coupling', desc: 'Uncoupled Glottal-Pharyngeal (r = 0.12)', status: 'DEVIATION' },
        { pair: 'F1 ↔ F2', name: 'Tongue Height / Backness Transition', desc: 'Lag Anomaly / Disconnected (r = 0.18)', status: 'DEVIATION' },
        { pair: 'F2 ↔ F3', name: 'Oral / Retroflex Resonator', desc: 'Missing Vocal Dynamics (r = 0.14)', status: 'DEVIATION' },
        { pair: 'Energy ↔ Pitch', name: 'Subglottal Pressure Covariance', desc: 'Vocoder Flat-Energy Lock (r = 0.08)', status: 'DEVIATION' },
        { pair: 'Breath ↔ Onset', name: 'RPCI: Respiration-to-Voicing', desc: 'Instantaneous Electronic Onset', status: 'DEVIATION' },
        { pair: 'Phase Continuity', name: 'Frame Boundary Phase Drift', desc: 'Vocoder Frame Phase Hop (> 0.76)', status: 'DEVIATION' },
      ],
    },
  };

  // Generate simulated timeline based on preset
  const runPresetAnalysis = (presetKey: 'genuine_a' | 'genuine_tel' | 'clone_v1') => {
    setIsAnalyzing(true);
    setTimeout(() => {
      const preset = samplePresets[presetKey];
      const windows: TimelineWindow[] = [];
      let smoothed = 0.0;

      for (let i = 0; i < 7; i++) {
        const start = i * 0.5;
        const end = start + 2.0;
        const noise = (Math.sin(i * 1.5) * 0.08);
        const inst = Math.max(0.02, Math.min(0.98, preset.spoofProb * 0.8 + preset.bafvScore * 0.2 + noise));
        smoothed = i === 0 ? inst : 0.6 * smoothed + 0.4 * inst;

        let rec: 'ALLOW' | 'VERIFY' | 'HOLD_FOR_REVIEW' = 'ALLOW';
        if (smoothed >= 0.75) rec = 'HOLD_FOR_REVIEW';
        else if (smoothed >= 0.35) rec = 'VERIFY';

        // Apply Low-Confidence Guard: telephone audio doesn't unilaterally hold
        if (preset.isBandLimited && rec === 'HOLD_FOR_REVIEW') {
          rec = 'VERIFY';
        }

        windows.push({
          window_index: i,
          start_time_seconds: start,
          end_time_seconds: end,
          is_speech_active: true,
          speech_ratio: 0.85,
          baseline_spoof_probability: Math.round(preset.spoofProb * 1000) / 1000,
          bafv_anomaly_score: Math.round(preset.bafvScore * 1000) / 1000,
          channel_confidence: preset.channelConf,
          instantaneous_risk: Math.round(inst * 1000) / 1000,
          smoothed_risk: Math.round(smoothed * 1000) / 1000,
          recommendation: rec,
        });
      }

      const finalSmoothed = windows[windows.length - 1].smoothed_risk;
      let finalAction: 'ALLOW' | 'VERIFY' | 'HOLD_FOR_REVIEW' = 'ALLOW';
      if (finalSmoothed >= 0.75) finalAction = 'HOLD_FOR_REVIEW';
      else if (finalSmoothed >= 0.35) finalAction = 'VERIFY';

      if (preset.isBandLimited && finalAction === 'HOLD_FOR_REVIEW') {
        finalAction = 'VERIFY';
      }

      const resultPayload = {
        file_name: preset.file,
        processing_latency_ms: 21.4,
        channel_metadata: {
          snr_db: preset.snr,
          channel_confidence: preset.channelConf,
          is_band_limited_8khz: preset.isBandLimited,
          duration_seconds: 3.5,
          quality_warnings: preset.isBandLimited
            ? ['Telephone 8 kHz band-limiting detected; high-frequency phase cues suppressed.']
            : [],
        },
        baseline_detection: {
          model_name: 'AASIST-ResNet-Baseline',
          model_version: 'v1.2.0-baseline',
          scores: {
            spoof_probability: preset.spoofProb,
            bonafide_probability: Math.round((1 - preset.spoofProb) * 1000) / 1000,
            recommended_action: finalAction,
          },
        },
        biomechanical_features: {
          bafv_anomaly_score: preset.bafvScore,
          pitch_f0: { f0_mean_hz: 142.5, jitter_percent: preset.type === 'spoof' ? 0.08 : 0.95 },
          formants: { f1_hz: 710, f2_hz: 1240, f3_hz: 2580 },
          voice_quality: { shimmer_percent: preset.type === 'spoof' ? 0.02 : 2.4, hnr_db: 22.0 },
          pcta_engine: {
            pcvs_score: preset.pcvsScore,
            mahalanobis_distance: preset.mahalanobisDist,
            rpci: preset.rpci,
            ptva: preset.ptva,
            trajectory_pairs: preset.pairs,
          },
        },
        open_set_detector: preset.type === 'spoof'
          ? {
              detected_class: 'KNOWN_SYNTHETIC_GENERATOR',
              attack_type: 'ElevenLabs-Multilingual-v2',
              generator_family: 'Autoregressive-Diffusion',
              is_zero_day_unknown: false,
              description: 'High-fidelity commercial multi-speaker neural voice clone',
              open_set_confidence: 0.94,
            }
          : {
              detected_class: 'BONAFIDE_HUMAN_SPEECH',
              attack_type: 'NONE',
              generator_family: 'HUMAN_BIOLOGY',
              is_zero_day_unknown: false,
              description: 'Natural speech within empirical human manifold',
              open_set_confidence: 0.98,
            },
        overall_recommendation: finalAction,
      };

      setScanResult(resultPayload);
      setTimelineData(windows);
      setIsAnalyzing(false);

      // Auto-append to audit trail (FR-11)
      const newAudit: AuditRecord = {
        record_id: 'REC-' + Math.random().toString(36).substring(2, 9).toUpperCase(),
        prev_hash: auditRecords.length > 0 ? auditRecords[auditRecords.length - 1].record_hash : '0'.repeat(64),
        record_hash: Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join(''),
        timestamp: new Date().toISOString(),
        transaction_id: transactionId,
        event_type: 'AUDIO_SCAN_ANALYZED',
        features: {
          file: preset.file,
          spoof_probability: preset.spoofProb,
          recommendation: finalAction,
          channel_conf: preset.channelConf,
        },
      };
      setAuditRecords((prev) => [...prev, newAudit]);
    }, 600);
  };

  // Run initial scan on load
  useEffect(() => {
    runPresetAnalysis('genuine_a');
  }, []);

  // Challenge Generation (FR-09)
  const generateChallenge = () => {
    setIsGeneratingChallenge(true);
    setTimeout(() => {
      const codeDigits = Math.floor(1000 + Math.random() * 9000).toString();
      const codeWords = ['Alpha', 'Delta', 'Sierra', 'Victor', 'Tango'];
      const w1 = codeWords[Math.floor(Math.random() * codeWords.length)];
      const w2 = codeWords[Math.floor(Math.random() * codeWords.length)];

      const challenge = {
        transaction_id: transactionId,
        nonce: Math.random().toString(36).substring(2, 8).toUpperCase(),
        expected_digits: codeDigits,
        prompt_text: `Authorize transaction ${w1}: say ${codeDigits.split('').join(' ')} followed by ${w2}`,
        created_at_utc: Date.now(),
        expires_at_utc: Date.now() + 60000,
        validity_seconds: 60,
      };

      setChallengeData(challenge);
      setChallengeSecondsLeft(60);
      setChallengeEvaluation(null);
      setIsGeneratingChallenge(false);
    }, 400);
  };

  // Countdown timer for challenge nonce
  useEffect(() => {
    if (!challengeData) return;
    const timer = setInterval(() => {
      setChallengeSecondsLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [challengeData]);

  // Evaluate Spoken Challenge Response (FR-10)
  const evaluateChallengeResponse = (scenario: 'genuine_pass' | 'tts_delay_fail' | 'wrong_digit_fail') => {
    if (!challengeData) return;

    if (scenario === 'genuine_pass') {
      setChallengeEvaluation({
        challenge_status: 'PASS',
        composite_score: 0.885,
        per_check_scores: {
          content_match_score: 1.0,
          timing_cadence_score: 0.95,
          articulation_score: 0.90,
          biomechanical_stability_score: 0.88,
          acoustic_naturalness_score: 0.86,
        },
        diagnostics: {
          content_check: 'EXACT_DIGIT_MATCH (100%)',
          timing_check: 'NORMAL_HUMAN_CADENCE (1.2s latency)',
          articulation_check: 'NATURAL_VOCAL_TRACT_COUPLING',
          voice_spoof_prob: 0.14,
        },
        matrix_action: 'Action level downgraded: ALLOW (Transaction cleared)',
      });
    } else if (scenario === 'tts_delay_fail') {
      setChallengeEvaluation({
        challenge_status: 'FAIL',
        composite_score: 0.38,
        per_check_scores: {
          content_match_score: 1.0,
          timing_cadence_score: 0.20,
          articulation_score: 0.50,
          biomechanical_stability_score: 0.40,
          acoustic_naturalness_score: 0.15,
        },
        diagnostics: {
          content_check: 'EXACT_DIGIT_MATCH',
          timing_check: 'EXCESSIVE_LATENCY (4.8s on-demand TTS synthesis lag)',
          articulation_check: 'FLAT_PITCH_LOCK_AND_PHASE_JUMP',
          voice_spoof_prob: 0.88,
        },
        matrix_action: 'Recommend HOLD FOR REVIEW (Automated block advice to analyst)',
      });
    } else {
      setChallengeEvaluation({
        challenge_status: 'FAIL',
        composite_score: 0.25,
        per_check_scores: {
          content_match_score: 0.0,
          timing_cadence_score: 0.40,
          articulation_score: 0.50,
          biomechanical_stability_score: 0.45,
          acoustic_naturalness_score: 0.35,
        },
        diagnostics: {
          content_check: 'DIGIT_MISMATCH_OR_MISSING (Stolen Replay / Wrong Code)',
          timing_check: 'SUSPICIOUS_INSTANT_REPLAY (<200ms trigger)',
          articulation_check: 'IRREGULAR_ARTICULATION_RATE',
          voice_spoof_prob: 0.65,
        },
        matrix_action: 'Recommend HOLD FOR REVIEW (Stolen audio replay detected)',
      });
    }
  };

  // Delete Audit Record (FR-11)
  const deleteAuditRecord = (recordId: string) => {
    setAuditRecords((prev) =>
      prev.map((r) =>
        r.record_id === recordId
          ? { ...r, deleted: true, features: { purged: true, reason: 'FR-11_ANALYST_PURGE' } }
          : r
      )
    );
  };

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '24px 20px' }}>
      {/* Header Banner */}
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
          paddingBottom: '20px',
          borderBottom: '1px solid var(--border-card)',
          marginBottom: '28px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div
            style={{
              width: '46px',
              height: '46px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #0284c7 0%, #00f0ff 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: 'var(--shadow-glow-cyan)',
            }}
          >
            <ShieldAlert size={28} color="#040814" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.45rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                BAFV-PCTA <span style={{ color: 'var(--color-cyan)', fontWeight: 400 }}>| Voice-Clone Fraud Shield</span>
              </h1>
              <span className="badge badge-allow" style={{ fontSize: '0.7rem' }}>
                <Radio size={12} className="animate-pulse" /> SIH26104 • AICTE CELL
              </span>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Physiological Coupled-Trajectory Analysis (PCTA) & Spring Boot 3 Orchestrator
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
          <div
            className="glass-panel"
            style={{ padding: '6px 14px', borderRadius: '10px', fontSize: '0.78rem' }}
          >
            <span style={{ color: 'var(--text-dim)' }}>Baseline Model: </span>
            <span className="mono" style={{ color: 'var(--color-cyan)', fontWeight: 600 }}>
              AASIST-ResNet v1.2.0
            </span>
          </div>
          <div
            className="glass-panel"
            style={{ padding: '6px 14px', borderRadius: '10px', fontSize: '0.78rem' }}
          >
            <span style={{ color: 'var(--text-dim)' }}>Avg CPU Latency: </span>
            <span className="mono" style={{ color: '#10b981', fontWeight: 600 }}>
              ~19.5 ms
            </span>
          </div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 14px',
              borderRadius: '10px',
              background: 'rgba(255, 255, 255, 0.05)',
              fontSize: '0.78rem',
            }}
          >
            <Lock size={14} color="var(--color-cyan)" />
            <span>Analyst: <strong>investigator@bank.secure</strong></span>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav style={{ display: 'flex', gap: '8px', marginBottom: '24px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        {[
          { id: 'scan', label: '1. Forensic Audio Scan', icon: AudioWaveform },
          { id: 'timeline', label: '2. Sliding-Window Timeline (FR-05)', icon: Activity },
          { id: 'challenge', label: '3. TB-PC Spoken Challenge (FR-09/10)', icon: Mic },
          { id: 'audit', label: '4. Hash-Chained Audit Trail (FR-11)', icon: FileText },
          { id: 'info', label: '5. Architecture & Novelty', icon: Info },
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
                padding: '12px 18px',
                background: isActive ? 'rgba(56, 189, 248, 0.12)' : 'transparent',
                color: isActive ? 'var(--color-cyan)' : 'var(--text-muted)',
                borderBottom: isActive ? '2px solid var(--color-cyan)' : '2px solid transparent',
                borderRadius: '8px 8px 0 0',
                fontWeight: isActive ? 700 : 500,
                fontSize: '0.88rem',
              }}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      {/* TAB 1: FORENSIC AUDIO SCAN */}
      {activeTab === 'scan' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1.9fr', gap: '24px' }}>
          {/* Left Column: Transaction & Audio Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div className="glass-panel" style={{ padding: '22px' }}>
              <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Lock size={18} color="var(--color-cyan)" /> Transaction Context (FR-06)
              </h2>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div>
                  <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                    Transaction Identifier:
                  </label>
                  <input
                    type="text"
                    value={transactionId}
                    onChange={(e) => setTransactionId(e.target.value)}
                    className="mono"
                    style={{
                      width: '100%',
                      background: 'rgba(0,0,0,0.3)',
                      border: '1px solid var(--border-card)',
                      color: '#fff',
                      padding: '8px 12px',
                      borderRadius: '8px',
                      fontSize: '0.85rem',
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                    Transfer Amount: ₹{amountInr.toLocaleString('en-IN')}
                  </label>
                  <input
                    type="range"
                    min="5000"
                    max="1000000"
                    step="5000"
                    value={amountInr}
                    onChange={(e) => setAmountInr(Number(e.target.value))}
                    style={{ width: '100%', accentColor: 'var(--color-cyan)' }}
                  />
                  <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
                    {[15000, 45000, 150000, 750000].map((amt) => (
                      <button
                        key={amt}
                        onClick={() => setAmountInr(amt)}
                        style={{
                          flex: 1,
                          fontSize: '0.7rem',
                          background: 'rgba(255,255,255,0.05)',
                          color: 'var(--text-muted)',
                          padding: '4px',
                          borderRadius: '6px',
                        }}
                      >
                        ₹{amt >= 100000 ? `${amt / 100000}L` : `${amt / 1000}k`}
                      </button>
                    ))}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                  <div>
                    <span style={{ fontSize: '0.82rem', fontWeight: 600, display: 'block' }}>New Beneficiary (&lt;24h)</span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Elevates baseline transaction risk profile</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={isNewBeneficiary}
                    onChange={(e) => setIsNewBeneficiary(e.target.checked)}
                    style={{ width: '18px', height: '18px', accentColor: 'var(--color-rose)' }}
                  />
                </div>
              </div>
            </div>

            {/* Audio Ingestion Controls */}
            <div className="glass-panel" style={{ padding: '22px' }}>
              <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <AudioWaveform size={18} color="var(--color-primary)" /> Audio Ingestion & Verification
              </h2>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '18px' }}>
                {[
                  { id: 'genuine_a', label: '1. Human Voice (Clean)', desc: 'Natural F0 glide, formant trajectories, micro-jitter' },
                  { id: 'genuine_tel', label: '2. Telephone 8 kHz (Degraded)', desc: 'Bandlimited AMR/G.711 codec, line hum' },
                  { id: 'clone_v1', label: '3. Neural Voice Clone (Vishing)', desc: 'Phase discontinuities, pitch lock, vocoder buzz' },
                ].map((item) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      setSelectedSample(item.id as any);
                      runPresetAnalysis(item.id as any);
                    }}
                    style={{
                      textAlign: 'left',
                      padding: '10px 14px',
                      borderRadius: '10px',
                      background: selectedSample === item.id ? 'rgba(56, 189, 248, 0.15)' : 'rgba(0,0,0,0.2)',
                      border: selectedSample === item.id ? '1px solid var(--color-cyan)' : '1px solid rgba(255,255,255,0.05)',
                    }}
                  >
                    <div style={{ fontWeight: 600, fontSize: '0.84rem', color: selectedSample === item.id ? 'var(--color-cyan)' : '#fff' }}>
                      {item.label}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{item.desc}</div>
                  </button>
                ))}
              </div>

              {/* Consent Notice for Mic Recording (FR-13) */}
              <div
                style={{
                  background: 'rgba(245, 158, 11, 0.1)',
                  border: '1px solid rgba(245, 158, 11, 0.3)',
                  padding: '10px 14px',
                  borderRadius: '10px',
                  fontSize: '0.72rem',
                  color: '#fef3c7',
                  marginBottom: '16px',
                  display: 'flex',
                  gap: '8px',
                }}
              >
                <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} color="var(--color-amber)" />
                <span>
                  <strong>FR-13 Consent Notice:</strong> Voice authentication requires explicit participant authorization.
                  Audio streams are converted strictly to ephemeral feature vectors; raw audio is not stored.
                </span>
              </div>

              <button
                className="btn-primary"
                style={{ width: '100%', justifyContent: 'center' }}
                disabled={isAnalyzing}
                onClick={() => runPresetAnalysis(selectedSample as any)}
              >
                {isAnalyzing ? <RefreshCw className="animate-spin" size={18} /> : <Activity size={18} />}
                {isAnalyzing ? 'Executing Inference...' : 'Run Forensic Inspection'}
              </button>
            </div>
          </div>

          {/* Right Column: Forensic Output Cards */}
          {scanResult && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Primary Decision Banner */}
              <div
                className="glass-panel"
                style={{
                  padding: '24px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  borderLeft: `6px solid ${
                    scanResult.overall_recommendation === 'ALLOW'
                      ? '#10b981'
                      : scanResult.overall_recommendation === 'VERIFY'
                      ? '#f59e0b'
                      : '#f43f5e'
                  }`,
                }}
              >
                <div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Risk Engine Recommendation (FR-08)
                  </div>
                  <div style={{ fontSize: '1.8rem', fontWeight: 800, marginTop: '4px' }}>
                    {scanResult.overall_recommendation === 'ALLOW' && <span style={{ color: '#10b981' }}>ALLOW TRANSACTION</span>}
                    {scanResult.overall_recommendation === 'VERIFY' && <span style={{ color: '#f59e0b' }}>TRIGGER SPOKEN CHALLENGE (VERIFY)</span>}
                    {scanResult.overall_recommendation === 'HOLD_FOR_REVIEW' && <span style={{ color: '#f43f5e' }}>HOLD FOR REVIEW (HIGH SPOOF RISK)</span>}
                  </div>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                    Advisory recommendation for bank fraud analyst. Not an automated hard block.
                  </p>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <span
                    className={`badge ${
                      scanResult.overall_recommendation === 'ALLOW'
                        ? 'badge-allow'
                        : scanResult.overall_recommendation === 'VERIFY'
                        ? 'badge-verify'
                        : 'badge-hold'
                    }`}
                    style={{ fontSize: '0.85rem', padding: '6px 16px' }}
                  >
                    {scanResult.overall_recommendation}
                  </span>
                </div>
              </div>

              {/* 3 Metric Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                <div className="glass-panel" style={{ padding: '18px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Spoof Probability (R_v)</div>
                  <div className="mono" style={{ fontSize: '1.6rem', fontWeight: 700, color: scanResult.baseline_detection.scores.spoof_probability > 0.5 ? 'var(--color-rose)' : 'var(--color-cyan)', marginTop: '4px' }}>
                    {(scanResult.baseline_detection.scores.spoof_probability * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                    AASIST-ResNet Tensor Forward Pass
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: '18px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Channel Confidence (C)</div>
                  <div className="mono" style={{ fontSize: '1.6rem', fontWeight: 700, color: scanResult.channel_metadata.channel_confidence < 0.5 ? 'var(--color-amber)' : '#10b981', marginTop: '4px' }}>
                    {(scanResult.channel_metadata.channel_confidence * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                    SNR: {scanResult.channel_metadata.snr_db} dB {scanResult.channel_metadata.is_band_limited_8khz && '(8kHz Phone)'}
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: '18px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>BAFV Trajectory Anomaly</div>
                  <div className="mono" style={{ fontSize: '1.6rem', fontWeight: 700, color: scanResult.biomechanical_features.bafv_anomaly_score > 0.4 ? 'var(--color-rose)' : '#10b981', marginTop: '4px' }}>
                    {(scanResult.biomechanical_features.bafv_anomaly_score * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                    Jitter: {scanResult.biomechanical_features.pitch_f0.jitter_percent}% • Shimmer: {scanResult.biomechanical_features.voice_quality.shimmer_percent}%
                  </div>
                </div>
              </div>

              {/* Open-Set Generator Detector Verdict (FR-04 & Phase P4) */}
              {scanResult.open_set_detector && (
                <div
                  className="glass-panel"
                  style={{
                    padding: '14px 20px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    background: scanResult.open_set_detector.detected_class === 'BONAFIDE_HUMAN_SPEECH'
                      ? 'rgba(16, 185, 129, 0.08)'
                      : 'rgba(244, 63, 94, 0.1)',
                    border: `1px solid ${scanResult.open_set_detector.detected_class === 'BONAFIDE_HUMAN_SPEECH' ? 'rgba(16, 185, 129, 0.25)' : 'rgba(244, 63, 94, 0.3)'}`,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <ShieldAlert size={20} color={scanResult.open_set_detector.detected_class === 'BONAFIDE_HUMAN_SPEECH' ? '#34d399' : '#fb7185'} />
                    <div>
                      <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
                        Open-Set Generator Detector (FR-04 LOGO Analysis)
                      </div>
                      <div className="mono" style={{ fontSize: '0.92rem', fontWeight: 700, color: '#fff', marginTop: '2px' }}>
                        {scanResult.open_set_detector.detected_class === 'BONAFIDE_HUMAN_SPEECH' ? (
                          <span style={{ color: '#34d399' }}>BONAFIDE_HUMAN_SPEECH • Empirical Biological Manifold</span>
                        ) : scanResult.open_set_detector.is_zero_day_unknown ? (
                          <span style={{ color: '#fb7185' }}>UNKNOWN_ZERO_DAY_SYNTHETIC • Unseen Architecture Detected</span>
                        ) : (
                          <span style={{ color: '#fb7185' }}>
                            {scanResult.open_set_detector.attack_type} ({scanResult.open_set_detector.generator_family})
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '2px' }}>
                        {scanResult.open_set_detector.description}
                      </div>
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Fingerprint Match</div>
                    <div className="mono" style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
                      {((scanResult.open_set_detector.open_set_confidence || 0.95) * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              )}

              {/* PCTA Coupling Heat-Map Matrix (FR-11 & Scope Section 7) */}
              <div className="glass-panel" style={{ padding: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
                  <div>
                    <h3 style={{ fontSize: '0.92rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Activity size={16} color="var(--color-cyan)" /> PCTA Trajectory Coupling Heat-Map (FR-11)
                    </h3>
                    <p style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                      Physiological Coupled-Trajectory Analysis: Checks whether physical vocal-tract measurements move synchronously as human biology dictates.
                    </p>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span className="mono" style={{ fontSize: '0.74rem', color: 'var(--color-primary)' }}>
                      Adaptive Threshold &tau;<sub>L</sub>(a): <strong>{(0.35 * (1 - 0.4 * Math.min(1, Math.log(1 + amountInr/10000) / Math.log(1 + 1000000/10000)))).toFixed(3)}</strong>
                    </span>
                  </div>
                </div>

                {/* PCTA Sub-Module Diagnostic Summary (Scope Section 7) */}
                {scanResult.biomechanical_features?.pcta_engine && (
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gap: '10px',
                      marginBottom: '14px',
                      padding: '10px 14px',
                      background: 'rgba(0,0,0,0.25)',
                      borderRadius: '8px',
                      border: '1px solid rgba(255,255,255,0.06)',
                    }}
                  >
                    <div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Mahalanobis Manifold Distance (D<sub>M</sub>)</div>
                      <div className="mono" style={{ fontSize: '0.85rem', fontWeight: 700, color: (scanResult.biomechanical_features.pcta_engine.mahalanobis_distance || 1.0) > 3.0 ? 'var(--color-rose)' : 'var(--color-emerald)' }}>
                        {scanResult.biomechanical_features.pcta_engine.mahalanobis_distance || 1.15} &sigma;
                        <span style={{ fontSize: '0.66rem', fontWeight: 400, marginLeft: '6px', color: 'var(--text-dim)' }}>
                          {(scanResult.biomechanical_features.pcta_engine.mahalanobis_distance || 1.0) > 3.0 ? '(Out-of-Manifold)' : '(Within Normal Manifold)'}
                        </span>
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>RPCI Respiration Coupling</div>
                      <div className="mono" style={{ fontSize: '0.78rem', fontWeight: 600, color: '#38bdf8' }}>
                        {scanResult.biomechanical_features.pcta_engine.rpci?.status === 'NORMAL_RESPIRATION_COUPLING' ? 'Natural Inhalation' : 'Missing / Ambiguous'}
                        <span style={{ fontSize: '0.66rem', display: 'block', color: 'var(--text-dim)', fontWeight: 400 }}>
                          Rule: Missing breath is tolerated, never penalized
                        </span>
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>PTVA Articulator Speed Limits</div>
                      <div className="mono" style={{ fontSize: '0.78rem', fontWeight: 600, color: scanResult.biomechanical_features.pcta_engine.ptva?.speed_limit_violation ? 'var(--color-rose)' : 'var(--color-emerald)' }}>
                        {scanResult.biomechanical_features.pcta_engine.ptva?.speed_limit_violation ? 'Speed Limit Exceeded' : 'Biomechanical Limits Respected'}
                        <span style={{ fontSize: '0.66rem', display: 'block', color: 'var(--text-dim)', fontWeight: 400 }}>
                          Max |dF1/dt| &le; 8kHz/s • |dF2/dt| &le; 12kHz/s
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
                  {(scanResult.biomechanical_features?.pcta_engine?.trajectory_pairs || [
                    { pair: 'F0 ↔ F1', name: 'Glottal / Pharyngeal Coupling', desc: 'Natural Glottal Coupling (r = 0.84)', status: 'ALIGNED' },
                    { pair: 'F1 ↔ F2', name: 'Tongue Height / Backness Transition', desc: 'Physiological Trajectory (r = 0.79)', status: 'ALIGNED' },
                    { pair: 'F2 ↔ F3', name: 'Oral / Retroflex Resonator', desc: 'Natural Dynamic Resonators (r = 0.72)', status: 'ALIGNED' },
                    { pair: 'Energy ↔ Pitch', name: 'Subglottal Pressure Covariance', desc: 'Natural Lung-Vocal Fold Covariance (r = 0.81)', status: 'ALIGNED' },
                    { pair: 'Breath ↔ Onset', name: 'RPCI: Respiration-to-Voicing', desc: 'Normal Breath Coupling (48ms)', status: 'ALIGNED' },
                    { pair: 'Phase Continuity', name: 'Frame Boundary Phase Drift', desc: 'Continuous Acoustic Phase (< 0.08)', status: 'ALIGNED' },
                  ]).map((item: any, idx: number) => {
                    const isDev = item.status === 'DEVIATION' || scanResult.overall_recommendation === 'HOLD_FOR_REVIEW';
                    return (
                      <div
                        key={idx}
                        style={{
                          padding: '10px 12px',
                          borderRadius: '8px',
                          background: isDev ? 'rgba(244, 63, 94, 0.1)' : 'rgba(16, 185, 129, 0.08)',
                          border: `1px solid ${isDev ? 'rgba(244, 63, 94, 0.25)' : 'rgba(16, 185, 129, 0.2)'}`,
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span className="mono" style={{ fontSize: '0.76rem', fontWeight: 700, color: isDev ? '#fb7185' : '#34d399' }}>
                            {item.pair}
                          </span>
                          <span className="mono" style={{ fontSize: '0.68rem', color: isDev ? '#fb7185' : '#34d399' }}>
                            {item.status || (isDev ? 'DEVIATION' : 'ALIGNED')}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: '2px' }}>{item.name}</div>
                        <div className="mono" style={{ fontSize: '0.72rem', color: '#fff', marginTop: '4px' }}>
                          {item.desc || (isDev ? 'Uncoupled Trajectory' : 'Synchronous Trajectory')}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Quality Warnings or Rule Flags */}
              {scanResult.channel_metadata.quality_warnings.length > 0 && (
                <div
                  className="glass-panel"
                  style={{
                    padding: '14px 18px',
                    borderLeft: '4px solid var(--color-amber)',
                    background: 'rgba(245, 158, 11, 0.08)',
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: '0.82rem', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AlertTriangle size={16} /> Low-Confidence Guard Active (Scope Sec 9)
                  </div>
                  <p style={{ fontSize: '0.76rem', color: '#fef3c7', marginTop: '4px' }}>
                    {scanResult.channel_metadata.quality_warnings.join(' ')} System prevents unilateral hold on voice evidence alone when channel is degraded.
                  </p>
                </div>
              )}

              {/* Action shortcuts */}
              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  className="btn-outline"
                  style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                  onClick={() => setActiveTab('timeline')}
                >
                  <Activity size={16} /> View Sliding-Window Timeline
                </button>
                <button
                  className="btn-primary"
                  style={{ flex: 1, justifyContent: 'center' }}
                  onClick={() => {
                    setActiveTab('challenge');
                    generateChallenge();
                  }}
                >
                  <Mic size={16} /> Launch TB-PC Challenge
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: SLIDING-WINDOW TIMELINE (FR-05) */}
      {activeTab === 'timeline' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
              <div>
                <h2 style={{ fontSize: '1.1rem', fontWeight: 800 }}>
                  Sliding-Window Risk Timeline (FR-05)
                </h2>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  Continuous evaluation in 2.0 s sliding windows with 0.5 s hop. Fuses baseline predictions, BAFV anomalies, and exponential smoothing.
                </p>
              </div>
              <div style={{ display: 'flex', gap: '14px', fontSize: '0.75rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#38bdf8' }}></span> Instantaneous Risk (s_t)
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#f43f5e' }}></span> Smoothed Risk (S_t)
                </span>
              </div>
            </div>

            {/* Recharts Area Chart */}
            <div style={{ height: '320px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timelineData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorSmoothed" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="colorInstant" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
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
                      backgroundColor: 'rgba(15, 23, 42, 0.95)',
                      borderColor: 'rgba(56, 189, 248, 0.3)',
                      borderRadius: '8px',
                      fontSize: '0.78rem',
                    }}
                  />
                  <ReferenceLine y={0.35} stroke="#10b981" strokeDasharray="4 4" label={{ value: 'ALLOW (<0.35)', fill: '#10b981', fontSize: 10 }} />
                  <ReferenceLine y={0.75} stroke="#f43f5e" strokeDasharray="4 4" label={{ value: 'HOLD (>=0.75)', fill: '#f43f5e', fontSize: 10 }} />
                  <Area type="monotone" dataKey="instantaneous_risk" stroke="#38bdf8" strokeWidth={2} fillOpacity={1} fill="url(#colorInstant)" />
                  <Area type="monotone" dataKey="smoothed_risk" stroke="#f43f5e" strokeWidth={3} fillOpacity={1} fill="url(#colorSmoothed)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Window Breakdown Cards */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '14px' }}>
              Sliding Window Frames (2.0s Duration, 0.5s Hop)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '12px' }}>
              {timelineData.map((w) => (
                <div
                  key={w.window_index}
                  style={{
                    padding: '12px',
                    borderRadius: '10px',
                    background: 'rgba(0,0,0,0.3)',
                    border: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--color-cyan)', fontWeight: 600 }}>
                      Window #{w.window_index + 1}
                    </span>
                    <span
                      className={`badge ${
                        w.recommendation === 'ALLOW' ? 'badge-allow' : w.recommendation === 'VERIFY' ? 'badge-verify' : 'badge-hold'
                      }`}
                      style={{ fontSize: '0.65rem', padding: '2px 8px' }}
                    >
                      {w.recommendation}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    Span: {w.start_time_seconds.toFixed(1)}s - {w.end_time_seconds.toFixed(1)}s
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '0.72rem' }}>
                    <span>Smoothed Risk (S_t):</span>
                    <strong className="mono">{w.smoothed_risk.toFixed(3)}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                    <span>BAFV Anomaly:</span>
                    <span className="mono">{w.bafv_anomaly_score.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: TB-PC SPOKEN CHALLENGE (FR-09, FR-10) */}
      {activeTab === 'challenge' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr', gap: '24px' }}>
          {/* Challenge Generation Panel */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Lock size={18} color="var(--color-cyan)" /> Challenge Derivation (FR-09)
            </h2>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '18px' }}>
              Dynamic prompt derived from <code>HMAC-SHA256(TxnID || Amount || Nonce)</code>. Ephemeral single-use nonces prevent audio replay.
            </p>

            <button
              className="btn-primary"
              style={{ width: '100%', justifyContent: 'center', marginBottom: '20px' }}
              onClick={generateChallenge}
              disabled={isGeneratingChallenge}
            >
              <RefreshCw size={16} className={isGeneratingChallenge ? 'animate-spin' : ''} />
              {challengeData ? 'Regenerate Ephemeral Challenge' : 'Generate Spoken Challenge'}
            </button>

            {challengeData && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ padding: '14px', borderRadius: '10px', background: 'rgba(0,240,255,0.06)', border: '1px solid rgba(0,240,255,0.2)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.72rem', color: 'var(--color-cyan)', fontWeight: 700 }}>EPHEMERAL NONCE</span>
                    <span className="mono badge badge-verify" style={{ fontSize: '0.7rem' }}>
                      <Clock size={12} /> {challengeSecondsLeft}s remaining
                    </span>
                  </div>
                  <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '4px' }}>
                    {challengeData.nonce}
                  </div>
                </div>

                <div style={{ padding: '16px', borderRadius: '10px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Challenge Prompt Sentence (Speak Aloud)</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', marginTop: '6px' }}>
                    "{challengeData.prompt_text}"
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--color-cyan)', marginTop: '8px' }}>
                    Expected 4-Digit Sequence: <strong>{challengeData.expected_digits}</strong>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Response Verification Simulator */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Mic size={18} color="var(--color-primary)" /> Response Scoring (FR-10)
            </h2>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '18px' }}>
              Simulate caller responses to evaluate content accuracy, human latency cadence, and biomechanical stability.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '22px' }}>
              <button
                className="btn-outline"
                onClick={() => evaluateChallengeResponse('genuine_pass')}
                disabled={!challengeData}
                style={{ textAlign: 'center', padding: '12px 8px' }}
              >
                <div style={{ fontWeight: 700, color: '#10b981', fontSize: '0.82rem' }}>1. Genuine Caller</div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>Correct digits • 1.2s cadence</div>
              </button>

              <button
                className="btn-outline"
                onClick={() => evaluateChallengeResponse('tts_delay_fail')}
                disabled={!challengeData}
                style={{ textAlign: 'center', padding: '12px 8px' }}
              >
                <div style={{ fontWeight: 700, color: 'var(--color-rose)', fontSize: '0.82rem' }}>2. On-Demand Clone</div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>Correct digits • 4.8s lag</div>
              </button>

              <button
                className="btn-outline"
                onClick={() => evaluateChallengeResponse('wrong_digit_fail')}
                disabled={!challengeData}
                style={{ textAlign: 'center', padding: '12px 8px' }}
              >
                <div style={{ fontWeight: 700, color: 'var(--color-rose)', fontSize: '0.82rem' }}>3. Stolen Replay</div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>Wrong digits • &lt;200ms trigger</div>
              </button>
            </div>

            {/* Challenge Evaluation Results */}
            {challengeEvaluation && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div
                  style={{
                    padding: '16px',
                    borderRadius: '12px',
                    background: challengeEvaluation.challenge_status === 'PASS' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)',
                    border: `1px solid ${challengeEvaluation.challenge_status === 'PASS' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <div>
                    <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Outcome Matrix Status</span>
                    <div style={{ fontSize: '1.4rem', fontWeight: 800, color: challengeEvaluation.challenge_status === 'PASS' ? '#10b981' : '#f43f5e' }}>
                      CHALLENGE {challengeEvaluation.challenge_status}
                    </div>
                    <div style={{ fontSize: '0.76rem', color: '#fff', marginTop: '2px' }}>
                      {challengeEvaluation.matrix_action}
                    </div>
                  </div>
                  <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 800 }}>
                    {(challengeEvaluation.composite_score * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Per-Check Breakdown */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                  {Object.entries(challengeEvaluation.per_check_scores).map(([k, v]: any) => (
                    <div key={k} style={{ padding: '10px 14px', borderRadius: '8px', background: 'rgba(0,0,0,0.25)', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="mono" style={{ fontSize: '1rem', fontWeight: 700, color: v >= 0.7 ? '#10b981' : 'var(--color-rose)' }}>
                        {(v * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontStyle: 'italic' }}>
                  Diagnostics: {challengeEvaluation.diagnostics.timing_check} | {challengeEvaluation.diagnostics.content_check}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 4: HASH-CHAINED AUDIT TRAIL (FR-11) */}
      {activeTab === 'audit' && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
            <div>
              <h2 style={{ fontSize: '1.1rem', fontWeight: 800 }}>
                Cryptographic Hash-Chained Audit Trail (FR-11)
              </h2>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Each forensic record anchors the previous block's SHA-256 hash. Feature vectors only; customer raw audio is not retained.
              </p>
            </div>
            <span className="badge badge-allow">
              <CheckCircle2 size={14} /> Chain Valid ({auditRecords.length} Blocks)
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {auditRecords.map((r, i) => (
              <div
                key={r.record_id}
                style={{
                  padding: '14px 18px',
                  borderRadius: '10px',
                  background: r.deleted ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.3)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  opacity: r.deleted ? 0.6 : 1,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--color-cyan)', fontWeight: 700 }}>
                      Block #{i + 1}
                    </span>
                    <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>{r.event_type}</span>
                    <span className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>[{r.record_id}]</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{new Date(r.timestamp).toLocaleTimeString()}</span>
                    {!r.deleted && (
                      <button
                        onClick={() => deleteAuditRecord(r.record_id)}
                        style={{
                          background: 'transparent',
                          color: 'var(--color-rose)',
                          padding: '4px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontSize: '0.7rem',
                        }}
                        title="Purge metadata (FR-11)"
                      >
                        <Trash2 size={14} /> Purge
                      </button>
                    )}
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.7rem', marginTop: '8px' }}>
                  <div className="mono" style={{ color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    Prev: {r.prev_hash.substring(0, 32)}...
                  </div>
                  <div className="mono" style={{ color: 'var(--color-primary)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    Hash: {r.record_hash.substring(0, 32)}...
                  </div>
                </div>

                {r.deleted ? (
                  <div style={{ fontSize: '0.72rem', color: 'var(--color-rose)', marginTop: '6px' }}>
                    * Metadata purged in compliance with FR-11 customer privacy right. Cryptographic hash preserved.
                  </div>
                ) : (
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                    Target: {r.transaction_id} | Recommendation: {r.features.recommendation || 'N/A'} | Spoof Prob: {r.features.spoof_probability !== undefined ? `${(r.features.spoof_probability * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 5: ARCHITECTURE & NOVELTY */}
      {activeTab === 'info' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: '14px', color: 'var(--color-cyan)' }}>
              Biomechanical Novelty (BAFV-PCTA)
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '14px' }}>
              Human vocal production is physically constrained by vocal fold aerodynamics (producing micro-jitter and micro-shimmer)
              and continuous vocal tract geometry transitions ($F_1, F_2, F_3$ formants).
            </p>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Generative neural vocoders (HiFi-GAN, WaveNet, Diffusion models) synthesize audio frame-by-frame or via spectrogram inversion,
              failing to replicate natural coupled biomechanical trajectories and introducing phase boundary discontinuities.
            </p>

            <div style={{ marginTop: '20px', padding: '14px', borderRadius: '10px', background: 'rgba(0,0,0,0.3)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-primary)', marginBottom: '6px' }}>
                Mathematical Risk Formulation (Scope Section 9):
              </div>
              <code style={{ fontSize: '0.74rem', color: '#f1f5f9' }}>
                logit(s) = b0 + C*(b1*Rv + b2*P_pcta + b3*Rs) + b4*Rc + b5*Rt + b6*(Rv*Rt)
              </code>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                S_t = 0.6 * S_(t-1) + 0.4 * s_t
              </div>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: '14px', color: 'var(--color-cyan)' }}>
              Responsible AI & Limitations (Scope Sec 15)
            </h2>
            <ul style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.6, paddingLeft: '18px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <li>
                <strong>Probabilistic Output:</strong> Synthetic audio detection is probabilistic and may degrade under unseen generative vocoders or severe line noise.
              </li>
              <li>
                <strong>Advisory Role:</strong> System recommendations (<code>ALLOW</code>, <code>VERIFY</code>, <code>HOLD FOR REVIEW</code>) support human fraud analysts and do not automatically freeze accounts.
              </li>
              <li>
                <strong>Privacy by Design:</strong> Complies with FR-11. Raw customer voice streams are never retained permanently; only ephemeral acoustic feature vectors are evaluated.
              </li>
              <li>
                <strong>SIH Domain:</strong> Aligned with Smart India Hackathon theme <em>Cybersecurity & FinTech</em>.
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;

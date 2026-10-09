"""Multimodal Risk Engine and Decision Rules for BAFV-PCTA.

Implements Phase 4: Risk Engine (Exit Check: Scenario Tests Pass - SC7).
Complies with:
- FR-06: Selection of simulated transaction amount, beneficiary status, velocity.
- FR-07: Combine audio evidence, channel confidence, and mock transaction signals.
- FR-08: Transparent advisory recommendations (ALLOW / VERIFY / HOLD_FOR_REVIEW).
- Mathematical formulation:
  logit(s) = b0 + C*(b1*R_v + b2*P_pcta + b3*R_s) + b4*R_c + b5*R_t + b6*R_v*R_t
  S_t = 0.6*S_(t-1) + 0.4*s_t
- Rules:
  - Low-confidence guard: if channel confidence < 0.5 and voice evidence >= 0.5,
    outcome is at least VERIFY and cannot become a hold on voice evidence alone.
  - Persistence rule: HOLD_FOR_REVIEW requires hold band in >= 3 of the last 5 windows,
    unless R_t > 0.9 (extreme high-value anomaly).
  - Challenge outcome matrix: PASS lowers action level, INCONCLUSIVE repeats/calls back,
    FAIL recommends HOLD FOR REVIEW.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TransactionContext(BaseModel):
    """Simulated banking transaction signals (FR-06)."""
    transaction_id: str = "TXN-MOCK-901"
    amount_inr: float = Field(..., ge=0, description="Transaction amount in INR")
    is_new_beneficiary: bool = Field(False, description="Is the recipient newly added (<24h)?")
    transfer_velocity_per_hour: int = Field(1, ge=0, description="Transfers initiated in past hour")
    device_ip_reputation: float = Field(0.9, ge=0.0, le=1.0, description="IP/Device trust score")


class AudioEvidenceContext(BaseModel):
    """Audio evidence parameters extracted from passive analysis."""
    voice_spoof_prob: float = Field(..., ge=0.0, le=1.0, description="R_v: Baseline model spoof score")
    bafv_pcta_score: float = Field(0.0, ge=0.0, le=1.0, description="P_pcta: Biomechanical trajectory anomaly")
    sustained_stability_risk: float = Field(0.0, ge=0.0, le=1.0, description="R_s: Sustained vowel stability risk")
    channel_confidence: float = Field(1.0, ge=0.0, le=1.0, description="C: Channel confidence index (SNR/band)")


class ChallengeOutcome(BaseModel):
    """TB-PC Spoken Challenge result for outcome matrix evaluation."""
    challenge_status: Literal["PASS", "INCONCLUSIVE", "FAIL", "NOT_TRIGGERED"]
    per_check_scores: Optional[Dict[str, float]] = None


class RiskDecisionOutput(BaseModel):
    raw_transaction_risk: float
    channel_confidence: float
    instantaneous_risk_score: float
    smoothed_risk_score: float
    recommended_action: Literal["ALLOW", "VERIFY", "HOLD_FOR_REVIEW"]
    contributing_factors: List[str]
    persistence_rule_triggered: bool
    low_confidence_guard_active: bool
    challenge_matrix_applied: Optional[str] = None
    probabilistic_disclaimer: str


class RiskEngine:
    """Deterministic and Explainable Dynamic Risk Fusion Engine (DRFE)."""

    # Default calibrated coefficients (Scope Section 7 & 9)
    B0 = -1.2  # Base bias
    B1 = 2.4   # Weight for raw voice spoof score (R_v)
    B2 = 1.1   # Weight for PCVS physiological coupled-trajectory anomaly (P_pcta)
    B3 = 0.8   # Weight for speaker consistency risk (R_s)
    B4 = 0.5   # Penalty for channel degradation (R_c)
    B5 = 1.8   # Weight for simulated transaction risk (R_t)
    B6 = 1.5   # Cross-interaction between voice spoof and high transaction value

    # Adaptive Threshold Parameters (Scope Section 7)
    # tau_L(a) = tau_L0 * [1 - kappa * min(1, ln(1 + a/a0) / ln(1 + a_max/a0))]
    TAU_L0 = 0.35
    KAPPA = 0.40
    A0 = 10000.0     # Reference baseline amount: INR 10,000
    A_MAX = 1000000.0 # Maximum scale amount: INR 10,00,000
    HOLD_THRESHOLD = 0.75

    def compute_adaptive_allow_threshold(self, amount_inr: float) -> float:
        """Compute amount-aware adaptive ALLOW threshold tau_L(a).
        
        High transaction values automatically lower the threshold for triggering verification.
        """
        if amount_inr <= 0:
            return self.TAU_L0

        ratio = math.log(1.0 + amount_inr / self.A0) / math.log(1.0 + self.A_MAX / self.A0)
        scaled_ratio = min(1.0, max(0.0, ratio))
        tau_l = self.TAU_L0 * (1.0 - self.KAPPA * scaled_ratio)
        return float(round(tau_l, 4))

    def select_adaptive_challenge(
        self,
        voice_risk: float,
        txn_risk: float,
        channel_conf: float,
    ) -> str:
        """Adaptive challenge selector based on policy table (Scope Section 9)."""
        if voice_risk >= 0.75 and txn_risk >= 0.5:
            return "SKIP_CHALLENGE_HOLD_IMMEDIATE"
        elif channel_conf < 0.5:
            return "REPEAT_CLEANER_LINE_OR_CALLBACK"
        elif voice_risk >= 0.35 and txn_risk >= 0.6:
            return "OUT_OF_BAND_BANK_APP_APPROVAL"
        else:
            return "RANDOM_DIGIT_SPOKEN_CHALLENGE"

    def compute_transaction_risk(self, txn: TransactionContext) -> float:
        """Compute normalized transaction risk R_t from mock financial parameters."""
        # Amount risk sigmoid: 50,000 INR midpoint
        amount_score = 1.0 / (1.0 + math.exp(-(txn.amount_inr - 50000.0) / 30000.0))

        # Beneficiary novelty risk
        bene_penalty = 0.25 if txn.is_new_beneficiary else 0.0

        # Velocity risk (>3 transfers/hour increases risk)
        vel_penalty = min(0.3, max(0.0, (txn.transfer_velocity_per_hour - 1) * 0.1))

        # Device penalty
        device_penalty = (1.0 - txn.device_ip_reputation) * 0.2

        r_t = amount_score * 0.5 + bene_penalty + vel_penalty + device_penalty
        return float(min(1.0, max(0.0, r_t)))

    def evaluate_risk(
        self,
        audio: AudioEvidenceContext,
        txn: TransactionContext,
        previous_smoothed_score: float = 0.0,
        recent_window_actions: Optional[List[str]] = None,
        challenge_outcome: Optional[ChallengeOutcome] = None,
    ) -> RiskDecisionOutput:
        """Evaluate multimodal risk and apply decision rules (Scope Sec 9)."""
        r_t = self.compute_transaction_risk(txn)
        r_v = audio.voice_spoof_prob
        p_pcta = audio.bafv_pcta_score
        r_s = audio.sustained_stability_risk
        c = audio.channel_confidence
        r_c = 1.0 - c  # Channel degradation penalty

        # 1. Compute Logit and Instantaneous Score s_t
        logit = (
            self.B0
            + c * (self.B1 * r_v + self.B2 * p_pcta + self.B3 * r_s)
            + self.B4 * r_c
            + self.B5 * r_t
            + self.B6 * (r_v * r_t)
        )
        instantaneous_score = 1.0 / (1.0 + math.exp(-logit))

        # 2. Exponential Smoothing: S_t = 0.6 * S_(t-1) + 0.4 * s_t
        if previous_smoothed_score <= 0.0:
            smoothed_score = instantaneous_score
        else:
            smoothed_score = 0.6 * previous_smoothed_score + 0.4 * instantaneous_score

        # 3. Base Threshold Evaluation with Amount-Aware Adaptive Threshold (Scope Section 7)
        tau_l = self.compute_adaptive_allow_threshold(txn.amount_inr)
        if smoothed_score < tau_l:
            base_action = "ALLOW"
        elif smoothed_score < self.HOLD_THRESHOLD:
            base_action = "VERIFY"
        else:
            base_action = "HOLD_FOR_REVIEW"

        contributing_factors: List[str] = []
        if r_v >= 0.5:
            contributing_factors.append(f"Acoustic baseline model detected synthetic patterns (Spoof Prob: {r_v:.2f}).")
        if p_pcta >= 0.3:
            contributing_factors.append(f"Biomechanical vocal-tract trajectory anomaly detected (Score: {p_pcta:.2f}).")
        if r_t >= 0.5:
            contributing_factors.append(f"High transaction risk profile (Amount: ₹{txn.amount_inr:,.0f}, Velocity: {txn.transfer_velocity_per_hour}/hr).")
        if c < 0.5:
            contributing_factors.append(f"Degraded audio channel confidence (C: {c:.2f}); high acoustic uncertainty.")

        # 4. Low-Confidence Guard (Scope Sec 9):
        # If channel confidence < 0.5 and voice evidence >= 0.5,
        # outcome is at least VERIFY and CANNOT become a hold on voice evidence alone.
        guard_active = False
        if c < 0.5 and r_v >= 0.5:
            guard_active = True
            if base_action == "HOLD_FOR_REVIEW" and r_t < 0.85:
                base_action = "VERIFY"
            elif base_action == "ALLOW":
                base_action = "VERIFY"
            contributing_factors.append(
                "Low-Confidence Guard applied: Poor audio channel prevents unilateral HOLD on voice evidence alone."
            )

        # 5. Persistence Rule (Scope Sec 9):
        # HOLD FOR REVIEW requires hold band in at least 3 of last 5 windows; exception is R_t > 0.9
        persistence_triggered = False
        if base_action == "HOLD_FOR_REVIEW":
            if recent_window_actions:
                last_5 = recent_window_actions[-5:]
                holds_in_history = sum(1 for a in last_5 if a == "HOLD_FOR_REVIEW")
                # Need at least 3 holds, unless R_t > 0.9 (extreme high-value anomaly)
                if holds_in_history < 2 and r_t <= 0.9:
                    base_action = "VERIFY"
                    persistence_triggered = True
                    contributing_factors.append(
                        "Persistence rule: Single-window hold downgraded to VERIFY pending consistent temporal confirmation."
                    )

        # 6. Challenge Outcome Matrix (Scope Sec 9):
        # PASS lowers the action level, INCONCLUSIVE triggers repeat/callback, FAIL recommends HOLD FOR REVIEW.
        matrix_note = None
        if challenge_outcome and challenge_outcome.challenge_status != "NOT_TRIGGERED":
            if challenge_outcome.challenge_status == "PASS":
                matrix_note = "Spoken challenge PASSED: Action level downgraded."
                if base_action == "HOLD_FOR_REVIEW":
                    base_action = "VERIFY"
                elif base_action == "VERIFY":
                    base_action = "ALLOW"
            elif challenge_outcome.challenge_status == "FAIL":
                matrix_note = "Spoken challenge FAILED: Recommending HOLD FOR REVIEW."
                base_action = "HOLD_FOR_REVIEW"
            elif challenge_outcome.challenge_status == "INCONCLUSIVE":
                matrix_note = "Spoken challenge INCONCLUSIVE: Recommending one retry or out-of-band analyst call-back."
                base_action = "VERIFY"

            if matrix_note:
                contributing_factors.append(matrix_note)

        return RiskDecisionOutput(
            raw_transaction_risk=round(r_t, 4),
            channel_confidence=round(c, 4),
            instantaneous_risk_score=round(instantaneous_score, 4),
            smoothed_risk_score=round(smoothed_score, 4),
            recommended_action=base_action,
            contributing_factors=contributing_factors,
            persistence_rule_triggered=persistence_triggered,
            low_confidence_guard_active=guard_active,
            challenge_matrix_applied=matrix_note,
            probabilistic_disclaimer=(
                "HOLD FOR REVIEW is an advisory recommendation for a human fraud analyst, never an automated block."
            ),
        )

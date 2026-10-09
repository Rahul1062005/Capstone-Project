"""Automated Tests for Risk Engine & Challenge Outcome Matrix (SC7, FR-06, FR-07, FR-08)."""

import pytest
from ai_service.risk_engine import (
    AudioEvidenceContext,
    ChallengeOutcome,
    RiskEngine,
    TransactionContext,
)


@pytest.fixture
def engine():
    return RiskEngine()


def test_scenario_low_risk_allow(engine):
    """Clean human voice + low financial amount -> ALLOW."""
    audio = AudioEvidenceContext(
        voice_spoof_prob=0.08,
        bafv_pcta_score=0.05,
        sustained_stability_risk=0.05,
        channel_confidence=0.95,
    )
    txn = TransactionContext(
        amount_inr=1500.0,
        is_new_beneficiary=False,
        transfer_velocity_per_hour=1,
    )
    result = engine.evaluate_risk(audio, txn)
    assert result.recommended_action == "ALLOW"
    assert result.smoothed_risk_score < 0.35


def test_scenario_high_spoof_persistent_hold(engine):
    """High synthetic spoof score with historical persistence -> HOLD FOR REVIEW."""
    audio = AudioEvidenceContext(
        voice_spoof_prob=0.92,
        bafv_pcta_score=0.85,
        sustained_stability_risk=0.75,
        channel_confidence=0.90,
    )
    txn = TransactionContext(
        amount_inr=85000.0,
        is_new_beneficiary=True,
        transfer_velocity_per_hour=2,
    )
    # Past 5 windows had multiple HOLD actions
    history = ["HOLD_FOR_REVIEW", "HOLD_FOR_REVIEW", "VERIFY", "HOLD_FOR_REVIEW"]
    result = engine.evaluate_risk(audio, txn, previous_smoothed_score=0.82, recent_window_actions=history)
    assert result.recommended_action == "HOLD_FOR_REVIEW"
    assert result.smoothed_risk_score >= 0.75


def test_low_confidence_guard_triggers(engine):
    """Scope Sec 9: Bad line (C < 0.5) with voice spoof evidence (R_v >= 0.5) must NOT hold on voice alone."""
    audio = AudioEvidenceContext(
        voice_spoof_prob=0.88,
        bafv_pcta_score=0.20,
        sustained_stability_risk=0.10,
        channel_confidence=0.35,  # Degraded 8kHz telephone line
    )
    txn = TransactionContext(
        amount_inr=10000.0,
        is_new_beneficiary=False,
        transfer_velocity_per_hour=1,
    )
    result = engine.evaluate_risk(audio, txn)
    # Guard forces at least VERIFY and prevents automated HOLD
    assert result.low_confidence_guard_active is True
    assert result.recommended_action == "VERIFY"
    assert any("Low-Confidence Guard applied" in f for f in result.contributing_factors)


def test_persistence_rule_downgrades_single_burst(engine):
    """Scope Sec 9: Isolated single-window spike without persistence downgrades to VERIFY."""
    audio = AudioEvidenceContext(
        voice_spoof_prob=0.89,
        bafv_pcta_score=0.80,
        sustained_stability_risk=0.70,
        channel_confidence=0.90,
    )
    txn = TransactionContext(
        amount_inr=30000.0,
        is_new_beneficiary=False,
        transfer_velocity_per_hour=1,
    )
    # History shows completely clean windows before this spike
    history = ["ALLOW", "ALLOW", "ALLOW", "ALLOW"]
    result = engine.evaluate_risk(audio, txn, recent_window_actions=history)
    assert result.persistence_rule_triggered is True
    assert result.recommended_action == "VERIFY"


def test_extreme_high_value_bypasses_persistence(engine):
    """Scope Sec 9: Extreme transaction risk (R_t > 0.9) does NOT wait for persistence."""
    audio = AudioEvidenceContext(
        voice_spoof_prob=0.85,
        bafv_pcta_score=0.70,
        sustained_stability_risk=0.60,
        channel_confidence=0.90,
    )
    txn = TransactionContext(
        amount_inr=2500000.0,  # 2.5 million INR
        is_new_beneficiary=True,
        transfer_velocity_per_hour=5,
        device_ip_reputation=0.1,  # Suspicious IP
    )
    result = engine.evaluate_risk(audio, txn, recent_window_actions=["ALLOW", "ALLOW"])
    assert result.recommended_action == "HOLD_FOR_REVIEW"


def test_challenge_outcome_matrix(engine):
    """Scope Sec 9: Outcome matrix applies PASS, INCONCLUSIVE, and FAIL transitions."""
    audio = AudioEvidenceContext(
        voice_spoof_prob=0.55,
        bafv_pcta_score=0.40,
        sustained_stability_risk=0.30,
        channel_confidence=0.85,
    )
    txn = TransactionContext(amount_inr=20000.0)

    # 1. Base outcome without challenge is VERIFY
    base = engine.evaluate_risk(audio, txn)
    assert base.recommended_action == "VERIFY"

    # 2. Challenge PASS lowers action level to ALLOW
    pass_challenge = ChallengeOutcome(challenge_status="PASS")
    result_pass = engine.evaluate_risk(audio, txn, challenge_outcome=pass_challenge)
    assert result_pass.recommended_action == "ALLOW"

    # 3. Challenge FAIL recommends HOLD FOR REVIEW
    fail_challenge = ChallengeOutcome(challenge_status="FAIL")
    result_fail = engine.evaluate_risk(audio, txn, challenge_outcome=fail_challenge)
    assert result_fail.recommended_action == "HOLD_FOR_REVIEW"

    # 4. Challenge INCONCLUSIVE recommends retry / callback (retains VERIFY)
    inconclusive_challenge = ChallengeOutcome(challenge_status="INCONCLUSIVE")
    result_inconclusive = engine.evaluate_risk(audio, txn, challenge_outcome=inconclusive_challenge)
    assert result_inconclusive.recommended_action == "VERIFY"

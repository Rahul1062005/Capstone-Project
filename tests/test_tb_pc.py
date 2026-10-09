"""Tests for TB-PC Cryptographic Challenge and Response Verifier (SC5, FR-09, FR-10)."""

import time
import pytest
import numpy as np

from ai_service.tb_pc_verifier import TBPCCryptoChallenge, TBPCResponseVerifier


@pytest.fixture
def verifier():
    return TBPCResponseVerifier()


def test_crypto_challenge_generation():
    """FR-09: Challenge derived from HMAC-SHA256 of transaction ID, amount, and nonce."""
    secret = "BANK-SECRET-KEY-2026"
    txn_id = "TXN-88291"
    amount = 45000.0
    nonce = "NONCE-XYZ-12345"

    challenge = TBPCCryptoChallenge.generate_challenge(
        secret_key=secret,
        transaction_id=txn_id,
        amount_inr=amount,
        nonce=nonce,
        validity_seconds=60,
    )

    assert challenge["transaction_id"] == txn_id
    assert challenge["nonce"] == nonce
    assert len(challenge["expected_digits"]) == 4
    assert len(challenge["hmac_digest"]) == 16
    assert "Authorize transaction" in challenge["prompt_text"]
    assert challenge["expires_at_utc"] > challenge["created_at_utc"]


def test_challenge_pass_roundtrip(verifier):
    """SC5, FR-10: Genuine biological human response with correct digits passes challenge."""
    sr = 16000
    t = np.linspace(0, 2.5, int(sr * 2.5), endpoint=False)
    # Biological speech simulation with F0 curve
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * t)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    waveform = (0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase)).astype(np.float32)

    prompt_time = time.time()
    response_time = prompt_time + 1.2  # 1.2s reading latency (healthy human range)

    result = verifier.verify_challenge(
        audio_waveform=waveform,
        expected_digits="4819",
        transcribed_text="4819 alpha victor",
        prompt_displayed_at=prompt_time,
        response_started_at=response_time,
    )

    assert result["challenge_status"] in ["PASS", "INCONCLUSIVE"]
    assert result["composite_score"] > 0.50
    assert "per_check_scores" in result
    assert result["per_check_scores"]["content_match_score"] == 1.0


def test_challenge_fail_on_incorrect_digits(verifier):
    """FR-10: Attacker failing digit check or failing naturalness gets FAIL."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    prompt_time = time.time()
    response_time = prompt_time + 5.5  # 5.5s excessive delay (TTS generation lag)

    result = verifier.verify_challenge(
        audio_waveform=waveform,
        expected_digits="7721",
        transcribed_text="9999 wrong code",
        prompt_displayed_at=prompt_time,
        response_started_at=response_time,
    )

    assert result["challenge_status"] == "FAIL"
    assert result["per_check_scores"]["content_match_score"] < 0.5
    assert result["detailed_diagnostics"]["content_check"] == "DIGIT_MISMATCH_OR_MISSING"

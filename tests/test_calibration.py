import numpy as np

from model.calibration import expected_calibration_error, fit_temperature, sigmoid


def make_data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(0, 2, n)
    labels = (rng.random(n) < sigmoid(true_logits)).astype(int)
    return true_logits, labels


def test_recovers_overconfidence():
    logits, labels = make_data()
    # A model 3x too confident should get T close to 3.
    assert abs(fit_temperature(logits * 3, labels) - 3) < 0.2


def test_scaling_reduces_ece():
    logits, labels = make_data()
    overconfident = logits * 3
    t = fit_temperature(overconfident, labels)
    assert expected_calibration_error(sigmoid(overconfident / t), labels) < expected_calibration_error(
        sigmoid(overconfident), labels
    )


def test_ece_is_small_when_calibrated():
    logits, labels = make_data()
    assert expected_calibration_error(sigmoid(logits), labels) < 0.02


def test_temperature_is_clamped():
    # Anti-correlated scores would push T toward 0 without the clamp.
    labels = np.array([0, 1] * 50)
    assert fit_temperature(np.where(labels == 1, -5.0, 5.0), labels) >= 0.05

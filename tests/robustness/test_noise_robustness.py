"""
Noise robustness tests for Beat2Bit ECG arrhythmia detection.
Tests model performance under various noise conditions that may affect
ECG signal quality in real-world scenarios.
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.benchmarking.model_evaluator import evaluate_model_predictions
from src.benchmarking.comparison_engine import compare_model_performance


def add_baseline_wander(signal, amplitude=0.5, frequency=0.5):
    """
    Add baseline wander (low frequency noise) to ECG signal.

    Args:
        signal: Input ECG signal (numpy array)
        amplitude: Amplitude of the wander
        frequency: Frequency of the wander in Hz

    Returns:
        Signal with baseline wander added
    """
    t = np.arange(len(signal)) / 360.0  # Assuming 360 Hz sampling
    wander = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal + wander


def add_muscle_artifact(signal, amplitude=0.3, duration=0.2):
    """
    Add muscle artifact (EMG noise) to ECG signal.

    Args:
        signal: Input ECG signal (numpy array)
        amplitude: Amplitude of the artifact
        duration: Duration of artifact bursts in seconds

    Returns:
        Signal with muscle artifact added
    """
    noisy_signal = signal.copy()
    # Add random bursts of high-frequency noise
    sample_rate = 360  # Hz
    burst_samples = int(duration * sample_rate)

    for _ in range(int(len(signal) / (sample_rate * 2))):  # Every 2 seconds on average
        start_idx = np.random.randint(0, len(signal) - burst_samples)
        noise = np.random.normal(0, amplitude, burst_samples)
        noisy_signal[start_idx:start_idx + burst_samples] += noise

    return noisy_signal


def add_electrode_motion(signal, amplitude=0.2, frequency=2.0):
    """
    Add electrode motion artifact to ECG signal.

    Args:
        signal: Input ECG signal (numpy array)
        amplitude: Amplitude of the motion artifact
        frequency: Frequency of the motion in Hz

    Returns:
        Signal with electrode motion artifact added
    """
    t = np.arange(len(signal)) / 360.0  # Assuming 360 Hz sampling
    motion = amplitude * np.sin(2 * np.pi * frequency * t) * np.exp(-t/10)  # Decaying oscillation
    return signal + motion


def add_white_noise(signal, snr_db=20):
    """
    Add white Gaussian noise to achieve target SNR.

    Args:
        signal: Input ECG signal (numpy array)
        snr_db: Target signal-to-noise ratio in dB

    Returns:
        Signal with white noise added
    """
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def create_synthetic_ecg_batch(n_samples=100, signal_length=180):
    """
    Create synthetic ECG-like signals for testing.

    Args:
        n_samples: Number of samples to generate
        signal_length: Length of each signal

    Returns:
        Tuple of (signals, labels) where labels are 0 (normal) or 1 (abnormal)
    """
    np.random.seed(42)  # For reproducibility

    signals = []
    labels = []

    for i in range(n_samples):
        # Create base signal with some ECG-like characteristics
        t = np.linspace(0, 4*np.pi, signal_length)  # Multiple periods

        # Generate normal vs abnormal based on random choice
        is_abnormal = np.random.rand() > 0.7  # 30% abnormal

        if is_abnormal:
            # Abnormal signal: add some irregularities
            signal = np.sin(t) + 0.5*np.sin(2*t) + 0.3*np.random.normal(0, 0.5, signal_length)
            # Add occasional spikes to mimic ectopic beats
            spike_indices = np.random.choice(signal_length, size=3, replace=False)
            signal[spike_indices] += np.random.uniform(2.0, 3.0, 3)
            labels.append(1)
        else:
            # Normal signal: regular sinus rhythm
            signal = np.sin(t) + 0.2*np.sin(3*t) + 0.1*np.random.normal(0, 0.3, signal_length)
            labels.append(0)

        signals.append(signal)

    return np.array(signals), np.array(labels)


class TestNoiseRobustness(unittest.TestCase):
    """Test model robustness to various types of noise and artifacts."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model for testing
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(180, 1)),
            tf.keras.layers.Conv1D(16, kernel_size=7, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Create test data
        self.X_test, self.y_test = create_synthetic_ecg_batch(n_samples=50, signal_length=180)
        self.X_test = self.X_test[..., np.newaxis]  # Add channel dimension

        # Get baseline predictions
        self.y_pred_baseline = self.model.predict(self.X_test, verbose=0)
        self.y_pred_binary_baseline = (self.y_pred_baseline >= 0.5).astype(int)
        self.baseline_metrics = evaluate_model_predictions(self.y_test, self.y_pred_baseline)

    def test_baseline_wander_robustness(self):
        """Test robustness to baseline wander noise."""
        wander_amplitudes = [0.1, 0.3, 0.5, 0.8]

        for amplitude in wander_amplitudes:
            with self.subTest(wander_amplitude=amplitude):
                # Apply baseline wander
                X_test_wander = np.array([
                    add_baseline_wander(signal.flatten(), amplitude=amplitude)
                    for signal in self.X_test
                ])
                X_test_wander = X_test_wander[..., np.newaxis]

                # Get predictions
                y_pred_wander = self.model.predict(X_test_wander, verbose=0)

                # Evaluate performance
                metrics_wander = evaluate_model_predictions(self.y_test, y_pred_wander)

                # Performance should not degrade catastrophically
                # Allow up to 15% drop in F1-score
                f1_drop = self.baseline_metrics['f1_score'] - metrics_wander['f1_score']
                self.assertLess(f1_drop, 0.15,
                              f"F1-score dropped by {f1_drop:.3f} with wander amplitude {amplitude}")

    def test_muscle_artifact_robustness(self):
        """Test robustness to muscle artifact (EMG noise)."""
        artifact_amplitudes = [0.1, 0.2, 0.4, 0.6]

        for amplitude in artifact_amplitudes:
            with self.subTest(muscle_amplitude=amplitude):
                # Apply muscle artifact
                X_test_muscle = np.array([
                    add_muscle_artifact(signal.flatten(), amplitude=amplitude)
                    for signal in self.X_test
                ])
                X_test_muscle = X_test_muscle[..., np.newaxis]

                # Get predictions
                y_pred_muscle = self.model.predict(X_test_muscle, verbose=0)

                # Evaluate performance
                metrics_muscle = evaluate_model_predictions(self.y_test, y_pred_muscle)

                # Performance should not degrade catastrophically
                f1_drop = self.baseline_metrics['f1_score'] - metrics_muscle['f1_score']
                self.assertLess(f1_drop, 0.20,
                              f"F1-score dropped by {f1_drop:.3f} with muscle amplitude {amplitude}")

    def test_electrode_motion_robustness(self):
        """Test robustness to electrode motion artifacts."""
        motion_amplitudes = [0.05, 0.1, 0.2, 0.3]

        for amplitude in motion_amplitudes:
            with self.subTest(motion_amplitude=amplitude):
                # Apply electrode motion
                X_test_motion = np.array([
                    add_electrode_motion(signal.flatten(), amplitude=amplitude)
                    for signal in self.X_test
                ])
                X_test_motion = X_test_motion[..., np.newaxis]

                # Get predictions
                y_pred_motion = self.model.predict(X_test_motion, verbose=0)

                # Evaluate performance
                metrics_motion = evaluate_model_predictions(self.y_test, y_pred_motion)

                # Performance should not degrade catastrophically
                f1_drop = self.baseline_metrics['f1_score'] - metrics_motion['f1_score']
                self.assertLess(f1_drop, 0.15,
                              f"F1-score dropped by {f1_drop:.3f} with motion amplitude {amplitude}")

    def test_white_noise_robustness(self):
        """Test robustness to white Gaussian noise."""
        snr_levels = [30, 20, 15, 10, 5]  # dB

        for snr_db in snr_levels:
            with self.subTest(snr_db=snr_db):
                # Apply white noise
                X_test_noisy = np.array([
                    add_white_noise(signal.flatten(), snr_db=snr_db)
                    for signal in self.X_test
                ])
                X_test_noisy = X_test_noisy[..., np.newaxis]

                # Get predictions
                y_pred_noisy = self.model.predict(X_test_noisy, verbose=0)

                # Evaluate performance
                metrics_noisy = evaluate_model_predictions(self.y_test, y_pred_noisy)

                # Performance should not degrade catastrophically
                f1_drop = self.baseline_metrics['f1_score'] - metrics_noisy['f1_score']
                self.assertLess(f1_drop, 0.25,
                              f"F1-score dropped by {f1_drop:.3f} at SNR {snr_db} dB")

    def test_combined_noise_robustness(self):
        """Test robustness to combined noise types."""
        # Apply multiple noise types simultaneously
        X_test_combined = self.X_test.copy()

        # Add baseline wander
        X_test_combined = np.array([
            add_baseline_wander(signal.flatten(), amplitude=0.3)
            for signal in X_test_combined
        ])

        # Add muscle artifact
        X_test_combined = np.array([
            add_muscle_artifact(signal.flatten(), amplitude=0.2)
            for signal in X_test_combined
        ])

        # Add white noise
        X_test_combined = np.array([
            add_white_noise(signal.flatten(), snr_db=15)
            for signal in X_test_combined
        ])

        X_test_combined = X_test_combined[..., np.newaxis]

        # Get predictions
        y_pred_combined = self.model.predict(X_test_combined, verbose=0)

        # Evaluate performance
        metrics_combined = evaluate_model_predictions(self.y_test, y_pred_combined)

        # With combined noise, we expect some degradation but not failure
        f1_drop = self.baseline_metrics['f1_score'] - metrics_combined['f1_score']
        self.assertLess(f1_drop, 0.30,
                      f"F1-score dropped by {f1_drop:.3f} with combined noise")

    def test_performance_degradation_trend(self):
        """Test that performance degradation follows expected trend with increasing noise."""
        snr_levels = [30, 25, 20, 15, 10, 5]
        f1_scores = []

        for snr_db in snr_levels:
            # Apply white noise
            X_test_noisy = np.array([
                add_white_noise(signal.flatten(), snr_db=snr_db)
                for signal in self.X_test
            ])
            X_test_noisy = X_test_noisy[..., np.newaxis]

            # Get predictions
            y_pred_noisy = self.model.predict(X_test_noisy, verbose=0)

            # Evaluate performance
            metrics_noisy = evaluate_model_predictions(self.y_test, y_pred_noisy)
            f1_scores.append(metrics_noisy['f1_score'])

        # F1 scores should generally decrease as noise increases (SNR decreases)
        # Allow for small fluctuations due to randomness
        for i in range(1, len(f1_scores)):
            # Score should not increase significantly as noise increases
            self.assertLessEqual(f1_scores[i] - f1_scores[i-1], 0.05,
                               f"F1 score increased unexpectedly from SNR {snr_levels[i-1]} to {snr_levels[i]} dB")


class TestSignalQualityMetrics(unittest.TestCase):
    """Test signal quality metrics and their correlation with model performance."""

    def test_signal_energy_calculation(self):
        """Test calculation of signal energy as a quality metric."""
        signals, _ = create_synthetic_ecg_batch(n_samples=10, signal_length=180)

        # Calculate signal energy (sum of squares)
        energies = np.sum(signals**2, axis=1)

        # Energies should be positive
        self.assertTrue(np.all(energies >= 0))

        # Normalize for comparison
        normalized_energies = energies / np.max(energies)
        self.assertTrue(np.all(normalized_energies <= 1.0))
        self.assertTrue(np.all(normalized_energies >= 0))

    def test_snr_estimation(self):
        """Test SNR estimation from clean and noisy signals."""
        clean_signal, _ = create_synthetic_ecg_batch(n_samples=5, signal_length=180)
        clean_signal = clean_signal[0]  # Take first sample

        # Add known amount of noise
        noise_level = 0.5
        noise = np.random.normal(0, noise_level, len(clean_signal))
        noisy_signal = clean_signal + noise

        # Calculate estimated SNR
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean(noise**2)
        if noise_power > 0:
            snr_estimated = 10 * np.log10(signal_power / noise_power)
            snr_expected = 10 * np.log10(signal_power / noise_level**2)

            # Should be close (within 1 dB due to randomness)
            self.assertAlmostEqual(snr_estimated, snr_expected, delta=1.0)


if __name__ == '__main__':
    unittest.main()
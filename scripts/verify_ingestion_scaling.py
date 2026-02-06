
import numpy as np
import scipy.signal
# Mocking the scaling logic to verify correctness
def test_scaling_logic():
    print("Testing Scaling Logic...")
    
    # Simulate a signal (e.g., 360Hz) - 1 second duration = 360 samples
    original_fs = 360
    target_fs = 250
    duration_sec = 1
    
    t = np.linspace(0, duration_sec, original_fs * duration_sec, endpoint=False)
    
    # Create a signal with a "Peak" at 0.5 seconds (Middle)
    # Peak index should be 180
    peak_time = 0.5
    original_peak_idx = int(peak_time * original_fs) # 180
    
    signal = np.zeros_like(t)
    # Gaussian bump at peak
    signal = np.exp(-((t - peak_time)**2) / 0.001)
    
    print(f"Original Signal: FS={original_fs}, Peak Index={original_peak_idx}, Peak Value={signal[original_peak_idx]:.4f}")
    assert signal[original_peak_idx] > 0.9, "Original peak construction failed"

    # --- SIMULATE THE FIX ---
    scale_factor = target_fs / original_fs # 0.694...
    
    # Resample
    new_len = int(len(signal) * scale_factor)
    resampled_signal = scipy.signal.resample(signal, new_len)
    
    # Scale Index
    new_peak_idx = int(original_peak_idx * scale_factor) # Should be ~125
    
    print(f"Resampled Signal: FS={target_fs}")
    print(f"Scaled Index: {new_peak_idx}")
    print(f"Value at Scaled Index: {resampled_signal[new_peak_idx]:.4f}")
    
    # PEAK ALIGNMENT CHECK
    # Check neighbors to ensure it's a local max
    neighbors = resampled_signal[new_peak_idx-5 : new_peak_idx+5]
    local_max = np.max(neighbors)
    
    is_aligned = resampled_signal[new_peak_idx] >= (local_max * 0.95)
    
    if is_aligned:
        print("PASS: Annotation aligns with signal peak.")
    else:
        print(f"FAIL: Value {resampled_signal[new_peak_idx]:.4f} is not the peak (Max: {local_max:.4f})")
    
    assert is_aligned, "Scaling logic failed to align peak"

if __name__ == "__main__":
    test_scaling_logic()

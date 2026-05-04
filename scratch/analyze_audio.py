import torchaudio
import torch

def analyze(file_path):
    print(f"Analyzing {file_path}...")
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        print(f"  Shape: {waveform.shape}")
        print(f"  Sample Rate: {sample_rate}")
        print(f"  Max: {waveform.max().item():.4f}")
        print(f"  Min: {waveform.min().item():.4f}")
        print(f"  Mean: {waveform.mean().item():.4f}")
        print(f"  Abs Max: {waveform.abs().max().item():.4f}")
        
        # Check for clipping
        clipping_count = (waveform.abs() >= 1.0).sum().item()
        if clipping_count > 0:
            print(f"  CLIPPING DETECTED: {clipping_count} samples hit 1.0 or higher")
        
        # Check for silence or low signal
        if waveform.abs().max() < 1e-4:
            print("  SILENCE DETECTED")
            
    except Exception as e:
        print(f"  Error: {e}")

analyze("ses.flac")
analyze("download (1).flac")

"""ACE-Step 1.5 Diagnostic Tools
Provides tools to analyze generated audio and system logs.
"""

import os
import torch
import numpy as np
import librosa
from datetime import datetime
from loguru import logger

def analyze_audio_health(audio_path, params=None):
    """Deep analysis of audio file for quality issues."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = np.mean(librosa.feature.rms(y=y))
        peak = np.max(np.abs(y))
        
        results = {
            "path": audio_path,
            "rms": rms,
            "peak": peak,
            "status": "HEALTHY"
        }
        
        if rms < 0.0001:
            results["status"] = "SILENT"
        elif rms < 0.05:
            results["status"] = "LOW_ENERGY_NOISE"
            
        # Log to file
        log_file = "/content/acestep_diagnostic_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {os.path.basename(audio_path)}\n")
            f.write(f"  Status: {results['status']}\n")
            f.write(f"  RMS: {rms:.6f}, Peak: {peak:.6f}\n")
            if params:
                f.write(f"  Caption: {params.get('caption', 'N/A')}\n")
                f.write(f"  Lyrics: {params.get('lyrics', 'N/A')[:100]}...\n")
            f.write("-" * 40 + "\n")
            
        return results
    except Exception as e:
        logger.error(f"Diagnostic failed for {audio_path}: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Example usage
    output_dir = "/content/acestep_outputs"
    if os.path.exists(output_dir):
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.flac', '.wav', '.mp3'))]
        if files:
            print(f"Analyzing {len(files)} files in {output_dir}...")
            for f in files:
                res = analyze_audio_health(f)
                print(f"{os.path.basename(f)}: {res['status']} (RMS: {res['rms']:.4f})")
        else:
            print("No audio files found in output directory.")
    else:
        print(f"Output directory {output_dir} not found.")

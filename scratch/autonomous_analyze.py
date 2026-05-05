import os
import subprocess
import re

def get_latest_audio():
    files = [f for f in os.listdir('.') if f.endswith('.flac')]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def analyze_with_ffmpeg(filename):
    print(f"\n--- Analiz Ediliyor (FFmpeg): {filename} ---")
    # volumedetect filter gives us mean_volume and max_volume
    cmd = f"ffmpeg -i \"{filename}\" -af volumedetect -f null /dev/null"
    result = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, text=True)
    
    output = result.stderr
    mean_vol = re.search(r"mean_volume: ([\-\d\.]+) dB", output)
    max_vol = re.search(r"max_volume: ([\-\d\.]+) dB", output)
    
    if mean_vol:
        print(f"Ortalama Ses Seviyesi: {mean_vol.group(1)} dB")
    if max_vol:
        print(f"Maksimum Ses Seviyesi: {max_vol.group(1)} dB")
    
    # Check if it's silence (usually < -60 dB)
    if mean_vol and float(mean_vol.group(1)) < -60:
        print("SONUÇ: DOSYA SESSİZ.")
    elif max_vol and float(max_vol.group(1)) > -0.1:
        print("SONUÇ: SES PATLAMASI (CLIPPING) VAR.")
    else:
        print("SONUÇ: SES DALGASI VAR, İÇERİĞİ ANALİZ EDİLMELİ.")

latest = get_latest_audio()
if latest:
    analyze_with_ffmpeg(latest)
else:
    print("Müzik dosyası bulunamadı.")

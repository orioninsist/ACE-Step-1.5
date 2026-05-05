import librosa
import numpy as np
import os

def analyze(filename):
    print(f"\n--- Analiz Ediliyor: {filename} ---")
    if not os.path.exists(filename):
        print("Dosya bulunamadı.")
        return
    
    y, sr = librosa.load(filename, sr=None)
    
    # 1. Enerji
    rms = np.mean(librosa.feature.rms(y=y))
    print(f"Ortalama RMS (Enerji): {rms:.6f}")
    
    # 2. Spektral İstatistikler
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    print(f"Spektral Merkez (Parlaklık): {centroid:.2f} Hz")
    
    # 3. Sessizlik Kontrolü
    non_silent = librosa.effects.split(y, top_db=60)
    print(f"Sessiz olmayan bölümlerin sayısı: {len(non_silent)}")
    
    # 4. Tahmin
    if rms < 0.0001:
        print("SONUÇ: DOSYA NEREDEYSE TAMAMEN SESSİZ.")
    elif centroid > 10000:
        print("SONUÇ: ÇOK YÜKSEK FREKANSLI GÜRÜLTÜ (Tıslama/Hiss) VAR.")
    elif centroid < 500:
        print("SONUÇ: ÇOK DÜŞÜK FREKANSLI UĞULTU VAR.")
    else:
        print("SONUÇ: SES DALGASI VAR AMA İÇERİĞİ ANLAŞILAMIYOR.")

for f in ["6.flac", "5.flac", "ses.flac"]:
    analyze(f)

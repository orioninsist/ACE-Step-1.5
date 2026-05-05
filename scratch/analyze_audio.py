import torch
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def analyze_audio_output(audio_path, params_used=None):
    print(f"🔍 Ses Analizi Başlatıldı: {os.path.basename(audio_path)}")
    
    if not os.path.exists(audio_path):
        print(f"❌ Dosya bulunamadı: {audio_path}")
        return

    # 1. Ses Dosyasını Yükle
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 2. Enerji Analizi (RMS)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    max_rms = np.max(rms)
    
    print(f"📊 Süre: {duration:.2f} saniye")
    print(f"📊 Ortalama Enerji (RMS): {avg_rms:.4f}")
    print(f"📊 Maksimum Enerji: {max_rms:.4f}")

    if avg_rms < 0.001:
        print("⚠️ SES ÇOK SESSIZ: Model sessizlik üretmiş olabilir.")
    elif avg_rms > 0.5:
        print("⚠️ SES ÇOK GÜRÜLTÜLÜ/PATLAMIŞ: Clipping (kırpma) olabilir.")
    
    # 3. Spektrogram Analizi (Görselleştirme için)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spektrogram: {os.path.basename(audio_path)}")
    plt.show()

    # 4. Parametre Kontrolü
    if params_used:
        print("\n📝 Kullanılan Parametrelerin Kontrolü:")
        print(f"   - Caption: '{params_used.get('caption', 'BOŞ')}'")
        print(f"   - Lyrics: '{params_used.get('lyrics', 'BOŞ')[:100]}...'")
        print(f"   - BPM: {params_used.get('bpm')}")
        print(f"   - Guidance Scale: {params_used.get('guidance_scale')}")
        print(f"   - Task Type: {params_used.get('task_type')}")
        
        if not params_used.get('caption') or len(params_used.get('caption')) < 5:
            print("❌ HATA: Prompt (Caption) çok kısa veya boş! Model ne üreteceğini bilmiyor.")
        
        if "[instrumental]" in params_used.get('lyrics', '').lower() and not params_used.get('instrumental'):
             print("ℹ️ NOT: Lyrics kısmında [Instrumental] etiketi var, vokal üretilmemesi normal.")

    # 5. Dinleme
    display(Audio(y, rate=sr))

# Önceki üretimden çıkan sonuçları al (Varsayalım 'results' değişkeninde)
if 'results' in globals() and results.success:
    for audio_info in results.audios:
        analyze_audio_output(audio_info['path'], audio_info['params'])
else:
    print("❌ Analiz edilecek başarılı bir üretim sonucu bulunamadı.")

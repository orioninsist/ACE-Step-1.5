import json
import os

notebook_path = "ACE_Step_1_5_A100_Colab.ipynb"

if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the analysis cell (the one with deep_analysis)
target_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def deep_analysis" in source:
            target_cell = cell
            break

if target_cell:
    new_source = [
        "import torch\n",
        "import numpy as np\n",
        "import librosa\n",
        "import librosa.display\n",
        "import matplotlib.pyplot as plt\n",
        "import os\n",
        "from datetime import datetime\n",
        "from IPython.display import Audio, display\n",
        "\n",
        "def deep_analysis(audio_path, params):\n",
        "    print(f'\\n--- DERİN ANALİZ: {os.path.basename(audio_path)} ---')\n",
        "    y, sr = librosa.load(audio_path, sr=None)\n",
        "    \n",
        "    # Enerji Analizi\n",
        "    rms = np.mean(librosa.feature.rms(y=y))\n",
        "    peak = np.max(np.abs(y))\n",
        "    print(f'Enerji Seviyesi (RMS): {rms:.6f}')\n",
        "    print(f'Tepe Değer (Peak): {peak:.6f}')\n",
        "    \n",
        "    # Spektrogram\n",
        "    plt.figure(figsize=(10, 4))\n",
        "    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)\n",
        "    S_dB = librosa.power_to_db(S, ref=np.max)\n",
        "    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)\n",
        "    plt.colorbar(format='%+2.0f dB')\n",
        "    plt.title('Mel-Spektrogram (Ses Karakteristiği)')\n",
        "    plt.show()\n",
        "    \n",
        "    status = \"BAŞARILI\"\n",
        "    if rms < 0.0001:\n",
        "        status = \"HATA: SESSİZ\"\n",
        "        print('❌ HATA: Ses tamamen sessiz veya çok zayıf.')\n",
        "    elif rms < 0.05:\n",
        "        status = \"UYARI: DÜŞÜK ENERJİ / GÜRÜLTÜ\"\n",
        "        print('⚠️ DURUM: Sadece düşük seviyeli gürültü var. Prompt etkisiz kalmış olabilir.')\n",
        "    \n",
        "    # Log Dosyasına Yaz\n",
        "    log_file = \"/content/acestep_diagnostic_log.txt\"\n",
        "    try:\n",
        "        with open(log_file, \"a\", encoding=\"utf-8\") as f:\n",
        "            f.write(f\"[{datetime.now()}] {os.path.basename(audio_path)} - Status: {status} - RMS: {rms:.6f}\\n\")\n",
        "            f.write(f\"Prompt: {params.get('caption')}\\n\")\n",
        "            f.write(f\"Lyrics: {params.get('lyrics')}\\n\")\n",
        "            f.write(\"-\" * 50 + \"\\n\")\n",
        "    except Exception as e:\n",
        "        print(f\"Log yazilamadi: {e}\")\n",
        "    \n",
        "    print(f'\\nKullanılan Prompt: {params.get(\"caption\")}')\n",
        "    print(f'Kullanılan Lyrics: {params.get(\"lyrics\")[:200]}...')\n",
        "    print(f'Diagnostic log güncellendi: {log_file}')\n",
        "    display(Audio(y, rate=sr))\n",
        "\n",
        "# Hem \\'result\\' hem \\'results\\' kontrol et (Notebook\\'ta result kullanılıyor)\n",
        "current_result = globals().get(\\'result\\') or globals().get(\\'results\\')\n",
        "\n",
        "if current_result and current_result.success:\n",
        "    for audio in current_result.audios:\n",
        "        deep_analysis(audio[\\'path\\'], audio[\\'params\\'])\n",
        "else:\n",
        "    print(\\'Analiz için geçerli bir üretim sonucu bulunamadı.\\')\n",
        "    if current_result and not current_result.success:\n",
        "        print(f\"Hata Mesajı: {current_result.error}\")\n"
    ]
    target_cell['source'] = new_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("✅ Notebook analysis cell updated successfully.")
else:
    print("❌ Could not find analysis cell in notebook.")

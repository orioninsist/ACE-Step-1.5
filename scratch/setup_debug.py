import os
import json

# 1. ACE-Step Core Debug Yaması
filepath = "/mnt/samsung/orion-backup-local/projects/ACE-Step-1.5/acestep/core/generation/handler/generate_music_execute.py"
with open(filepath, 'r') as f:
    content = f.read()

debug_code = """
        print(f"\\n--- [DEBUG] GENERATION INPUTS ---")
        print(f"   Task Type: {task_type}")
        print(f"   Captions[0]: {service_inputs['captions_batch'][0]}")
        print(f"   Lyrics[0]: {service_inputs['lyrics_batch'][0]}")
        print(f"   Metas[0]: {service_inputs['metas_batch'][0]}")
        print(f"   Guidance Scale: {guidance_scale}")
        print(f"   Inference Steps: {inference_steps}")
        print(f"   Has Audio Codes: {service_inputs['audio_code_hints_batch'] is not None}")
        if service_inputs['audio_code_hints_batch']:
            print(f"   Audio Codes Length: {len(str(service_inputs['audio_code_hints_batch'][0]))}")
        print(f"--- [DEBUG] END ---\\n")
"""

if 'print(f"\\n--- [DEBUG] GENERATION INPUTS ---")' not in content:
    # _service_target içine ekle
    marker = "def _service_target():"
    if marker in content:
        patched_content = content.replace(marker, marker + debug_code)
        with open(filepath, 'w') as f:
            f.write(patched_content)
        print("✅ Core library debug yaması uygulandı.")
    else:
        print("❌ Core library dosyası beklenen formatta değil.")
else:
    print("ℹ️ Core library zaten yamalı.")

# 2. Notebook'a Analiz Hücresi Ekleme
# Not: Bu kod notebook'u okuyup sonuna bir hücre ekleyebilir.
notebook_path = "/mnt/samsung/orion-backup-local/projects/ACE-Step-1.5/ACE_Step_1_5_A100_Colab.ipynb"
with open(notebook_path, 'r') as f:
    nb = json.load(f)

analysis_cell_source = [
    "import torch\n",
    "import numpy as np\n",
    "import librosa\n",
    "import librosa.display\n",
    "import matplotlib.pyplot as plt\n",
    "from IPython.display import Audio, display\n",
    "\n",
    "def deep_analysis(audio_path, params):\n",
    "    print(f'\\n--- DERİN ANALİZ: {os.path.basename(audio_path)} ---')\n",
    "    y, sr = librosa.load(audio_path, sr=None)\n",
    "    rms = np.mean(librosa.feature.rms(y=y))\n",
    "    print(f'Enerji Seviyesi: {rms:.6f}')\n",
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
    "    if rms < 0.0001:\n",
    "        print('HATA: Ses tamamen sessiz veya çok zayıf. Model çıktı üretmemiş.')\n",
    "    elif rms > 0.001 and rms < 0.05:\n",
    "        print('DURUM: Sadece düşük seviyeli gürültü var. Prompt etkisiz kalmış olabilir.')\n",
    "    \n",
    "    print(f'\\nKullanılan Prompt: {params.get(\"caption\")}')\n",
    "    print(f'Kullanılan Lyrics: {params.get(\"lyrics\")[:200]}...')\n",
    "    display(Audio(y, rate=sr))\n",
    "\n",
    "if 'results' in globals() and results.success:\n",
    "    for audio in results.audios:\n",
    "        deep_analysis(audio['path'], audio['params'])\n",
    "else:\n",
    "    print('Analiz için geçerli bir üretim sonucu bulunamadı.')"
]

# Hücrenin zaten olup olmadığını kontrol et
already_exists = any("deep_analysis" in "".join(c.get('source', [])) for c in nb['cells'])

if not already_exists:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"collapsed": False},
        "outputs": [],
        "source": analysis_cell_source
    }
    # "6) TURK TEST" hücresinden sonrasını bul
    insert_idx = len(nb['cells'])
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and "6) TURK TEST" in "".join(cell['source']):
            # Üretim hücresini de geçip sonrasına ekle
            insert_idx = i + 2 
            break
    
    nb['cells'].insert(insert_idx, new_cell)
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("✅ Notebook'a Analiz Hücresi eklendi.")
else:
    print("ℹ️ Analiz hücresi zaten mevcut.")

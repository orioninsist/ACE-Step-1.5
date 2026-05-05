import json
from pathlib import Path

notebook_path = Path("ACE_Step_1_5_A100_Colab.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_patch_source = [
    "#@title Ses Çakışması (Overlap) Düzeltme Yaması\n",
    "# Bu hücre GitHub'dan indirilen koddaki hatalı ses çakışma mantığını düzeltir.\n",
    "import sys\n",
    "import re\n",
    "from pathlib import Path\n",
    "\n",
    "file_path = Path(\"/content/ACE-Step-1.5/acestep/core/generation/handler/vae_decode_chunks.py\")\n",
    "if file_path.exists():\n",
    "    content = file_path.read_text()\n",
    "    \n",
    "    # Yeni çok kısa (1 latent frame) crossfade mantığı. \n",
    "    # Çiftleşme (stacking/chorus) ve cızırtı (crackling) sorunlarını tamamen çözer.\n",
    "    new_gpu = '''            window = torch.ones(1, 1, actual_len, dtype=audio_chunk.dtype, device=audio_chunk.device)\n",
    "            fade_in_latent = core_start - win_start\n",
    "            fade_out_latent = win_end - core_end\n",
    "            \n",
    "            if fade_in_latent > 0:\n",
    "                center_audio = int(round(fade_in_latent * upsample_factor))\n",
    "                cf_len = max(int(round(upsample_factor)), 2)\n",
    "                cf_half = cf_len // 2\n",
    "                start_audio = center_audio - cf_half\n",
    "                end_audio = start_audio + cf_len\n",
    "                if start_audio > 0: window[..., :start_audio] = 0.0\n",
    "                if cf_len > 0: window[..., start_audio:end_audio] = torch.linspace(0, 1, cf_len, dtype=audio_chunk.dtype, device=audio_chunk.device)\n",
    "                    \n",
    "            if fade_out_latent > 0:\n",
    "                center_audio = int(round(fade_out_latent * upsample_factor))\n",
    "                cf_len = max(int(round(upsample_factor)), 2)\n",
    "                cf_half = cf_len // 2\n",
    "                start_audio_from_end = center_audio + cf_half\n",
    "                end_audio_from_end = center_audio - cf_half\n",
    "                idx_start = actual_len - start_audio_from_end\n",
    "                idx_end = actual_len - end_audio_from_end\n",
    "                if cf_len > 0: window[..., idx_start:idx_end] = torch.linspace(1, 0, cf_len, dtype=audio_chunk.dtype, device=audio_chunk.device)\n",
    "                if end_audio_from_end > 0: window[..., idx_end:] = 0.0'''\n",
    "\n",
    "    new_cpu = new_gpu.replace(\"device=audio_chunk.device\", \"device=\\\"cpu\\\"\")\n",
    "\n",
    "    # Remove old block by regex to be safe against different versions\n",
    "    pattern = re.compile(r'\\s*window = torch\\.ones\\(1, 1, actual_len.*?if end_audio > 0:\\s*window\\[\\.\\.\\., idx_end:\\] = 0\\.0', re.DOTALL)\n",
    "    content = pattern.sub('', content)\n",
    "\n",
    "    # Also remove older fallback patterns\n",
    "    pattern2 = re.compile(r'\\s*window = torch\\.ones\\(1, 1, actual_len.*?if fade_out_audio > 0:\\s*window\\[\\.\\.\\., -fade_out_audio:\\] = torch\\.linspace\\(1, 0, fade_out_audio, dtype=audio_chunk\\.dtype, device=[^\\)]+\\)', re.DOTALL)\n",
    "    content = pattern2.sub('', content)\n",
    "\n",
    "    # Inject new block after actual_len\n",
    "    content = content.replace('audio_chunk = audio_chunk[:, :, :actual_len]\\n', 'audio_chunk = audio_chunk[:, :, :actual_len]\\n' + new_gpu + '\\n')\n",
    "    content = content.replace('audio_chunk = audio_chunk[:, :, :actual_len].cpu()\\n', 'audio_chunk = audio_chunk[:, :, :actual_len].cpu()\\n' + new_cpu + '\\n')\n",
    "\n",
    "    file_path.write_text(content)\n",
    "    print(\"✅ VAE Overlap-Add yaması başarıyla uygulandı! (Yeni Süper-Kısa Crossfade)\")\n"
]

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 0:
        if 'Ses Çakışması' in cell['source'][0] or 'Yaması' in cell['source'][0]:
            cell['source'] = new_patch_source
            found = True
            break

if not found:
    print("Patch cell not found in notebook!")
else:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook patched successfully!")

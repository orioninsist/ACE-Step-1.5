import json

with open("ACE_Step_1_5_A100_Colab.ipynb", "r") as f:
    notebook = json.load(f)

# Find the setup cell
setup_cell_idx = -1
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and any("git clone" in line for line in cell['source']):
        setup_cell_idx = i
        break

patch_code = """import sys
from pathlib import Path

file_path = Path("/content/ACE-Step-1.5/acestep/core/generation/handler/vae_decode_chunks.py")
if file_path.exists():
    content = file_path.read_text()
    
    old_gpu = '''            window = torch.ones(1, 1, actual_len, dtype=audio_chunk.dtype, device=audio_chunk.device)
            fade_in_latent = core_start - win_start
            fade_out_latent = win_end - core_end
            fade_in_audio = int(round(fade_in_latent * upsample_factor))
            fade_out_audio = int(round(fade_out_latent * upsample_factor))

            if fade_in_audio > 0:
                window[..., :fade_in_audio] = torch.linspace(0, 1, fade_in_audio, dtype=audio_chunk.dtype, device=audio_chunk.device)
            if fade_out_audio > 0:
                window[..., -fade_out_audio:] = torch.linspace(1, 0, fade_out_audio, dtype=audio_chunk.dtype, device=audio_chunk.device)'''

    new_gpu = '''            window = torch.ones(1, 1, actual_len, dtype=audio_chunk.dtype, device=audio_chunk.device)
            fade_in_latent = core_start - win_start
            fade_out_latent = win_end - core_end
            
            if fade_in_latent > 0:
                cf_half = fade_in_latent // 2
                fade_in_start = fade_in_latent - cf_half
                fade_in_end = fade_in_latent + cf_half
                start_audio = int(round(fade_in_start * upsample_factor))
                end_audio = int(round(fade_in_end * upsample_factor))
                cf_len = end_audio - start_audio
                if start_audio > 0: window[..., :start_audio] = 0.0
                if cf_len > 0: window[..., start_audio:end_audio] = torch.linspace(0, 1, cf_len, dtype=audio_chunk.dtype, device=audio_chunk.device)
                    
            if fade_out_latent > 0:
                cf_half = fade_out_latent // 2
                fade_out_start = fade_out_latent + cf_half
                fade_out_end = fade_out_latent - cf_half
                start_audio = int(round(fade_out_start * upsample_factor))
                end_audio = int(round(fade_out_end * upsample_factor))
                cf_len = start_audio - end_audio
                idx_start = actual_len - start_audio
                idx_end = actual_len - end_audio
                if cf_len > 0: window[..., idx_start:idx_end] = torch.linspace(1, 0, cf_len, dtype=audio_chunk.dtype, device=audio_chunk.device)
                if end_audio > 0: window[..., idx_end:] = 0.0'''

    old_cpu = '''            window = torch.ones(1, 1, actual_len, dtype=audio_chunk.dtype, device="cpu")
            fade_in_latent = core_start - win_start
            fade_out_latent = win_end - core_end
            fade_in_audio = int(round(fade_in_latent * upsample_factor))
            fade_out_audio = int(round(fade_out_latent * upsample_factor))

            if fade_in_audio > 0:
                window[..., :fade_in_audio] = torch.linspace(0, 1, fade_in_audio, dtype=audio_chunk.dtype, device="cpu")
            if fade_out_audio > 0:
                window[..., -fade_out_audio:] = torch.linspace(1, 0, fade_out_audio, dtype=audio_chunk.dtype, device="cpu")'''

    new_cpu = '''            window = torch.ones(1, 1, actual_len, dtype=audio_chunk.dtype, device="cpu")
            fade_in_latent = core_start - win_start
            fade_out_latent = win_end - core_end
            
            if fade_in_latent > 0:
                cf_half = fade_in_latent // 2
                fade_in_start = fade_in_latent - cf_half
                fade_in_end = fade_in_latent + cf_half
                start_audio = int(round(fade_in_start * upsample_factor))
                end_audio = int(round(fade_in_end * upsample_factor))
                cf_len = end_audio - start_audio
                if start_audio > 0: window[..., :start_audio] = 0.0
                if cf_len > 0: window[..., start_audio:end_audio] = torch.linspace(0, 1, cf_len, dtype=audio_chunk.dtype, device="cpu")
                    
            if fade_out_latent > 0:
                cf_half = fade_out_latent // 2
                fade_out_start = fade_out_latent + cf_half
                fade_out_end = fade_out_latent - cf_half
                start_audio = int(round(fade_out_start * upsample_factor))
                end_audio = int(round(fade_out_end * upsample_factor))
                cf_len = start_audio - end_audio
                idx_start = actual_len - start_audio
                idx_end = actual_len - end_audio
                if cf_len > 0: window[..., idx_start:idx_end] = torch.linspace(1, 0, cf_len, dtype=audio_chunk.dtype, device="cpu")
                if end_audio > 0: window[..., idx_end:] = 0.0'''

    content = content.replace(old_gpu, new_gpu)
    content = content.replace(old_cpu, new_cpu)
    file_path.write_text(content)
    print("✅ VAE Overlap-Add yamasi basariyla uygulandi!")
"""

patch_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "#@title Ses Çakışması (Overlap) Düzeltme Yaması\n",
        "# Bu hücre GitHub'dan indirilen koddaki hatalı ses çakışma mantığını düzeltir.\n"
    ] + [line + "\n" for line in patch_code.split("\n")]
}

if setup_cell_idx != -1:
    notebook['cells'].insert(setup_cell_idx + 1, patch_cell)
    with open("ACE_Step_1_5_A100_Colab.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
    print("Notebook patched successfully.")
else:
    print("Could not find setup cell.")

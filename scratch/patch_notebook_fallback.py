import json
import os

notebook_path = "/mnt/samsung/orion-backup-local/projects/ACE-Step-1.5/ACE_Step_1_5_A100_Colab.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "caption_value = formatted.caption or prompt" in source:
            # Found the cell!
            print("Found cell to patch in notebook.")
            new_source = source.replace(
                "        caption_value = formatted.caption or prompt",
                '        # Robustness: LLM çıktısı boşsa veya meta-veri anahtarları içeriyorsa orijinal prompt\'a dön\n        caption_cand = (formatted.caption or "").strip()\n        if not caption_cand or any(k in caption_cand.lower() for k in ["bpm:", "duration:", "caption:", "lyrics:"]):\n            print("⚠️ LLM\'den hatalı veya boş prompt geldi, orijinal prompt kullanılıyor.")\n            caption_value = prompt\n        else:\n            caption_value = caption_cand'
            )
            cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in new_source.split("\n")]
            # Remove the last empty string if split added one
            if cell['source'][-1] == "\n":
                cell['source'].pop()

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")

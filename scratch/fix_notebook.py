import json
import os

notebook_path = "ACE_Step_1_5_A100_Colab.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 4. Hücreyi (Importlar) güncelle
# 5. Hücreyi (Normalizasyon) güncelle

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 4. Hücre kontrolü
        if "Servisi baslat" in source and "AceStepHandler" in source:
            new_source = [
                "#@title 4) Servisi baslat: A100 en iyi kalite\n",
                "import torch\n",
                "import numpy as np\n",
                "import os\n",
                "import shutil\n",
                "import site\n",
                "import subprocess\n",
                "import sys\n",
                "from pathlib import Path\n",
                "from IPython.display import Audio, display\n",
                "\n",
                "PROJECT_ROOT = Path(\"/content/ACE-Step-1.5\")\n",
                "DRIVE_ROOT = Path(\"/content/drive/MyDrive/path\")\n",
                "CHECKPOINTS_DIR = DRIVE_ROOT / \"acestep_checkpoints\"\n",
                "OUTPUT_DIR = Path(\"/content/acestep_outputs\")\n",
                "VENV_DIR = PROJECT_ROOT / \".venv\"\n",
                "\n",
                "def _find_site_packages():\n",
                "    return next(VENV_DIR.glob(\"lib/python*/site-packages\"), None)\n",
                "\n",
                "site_packages = _find_site_packages()\n",
                "if site_packages is None:\n",
                "    print(\".venv bulunamadi; uv sync tekrar calistiriliyor...\")\n",
                "    uv_bin = shutil.which(\"uv\") or \"/root/.local/bin/uv\"\n",
                "    subprocess.run([uv_bin, \"sync\"], cwd=PROJECT_ROOT, check=True)\n",
                "    site_packages = _find_site_packages()\n",
                "\n",
                "site.addsitedir(str(site_packages))\n",
                "sys.path.insert(0, str(site_packages))\n",
                "sys.path.insert(0, str(PROJECT_ROOT))\n",
                "\n",
                "from acestep.handler import AceStepHandler\n",
                "from acestep.llm_inference import LLMHandler\n",
                "from acestep.model_downloader import ensure_dit_model, ensure_lm_model, get_checkpoints_dir\n",
                "from acestep.inference import GenerationParams, GenerationConfig, format_sample, generate_music\n",
                "\n",
                "DIT_MODEL = \"acestep-v15-xl-sft\"\n",
                "LM_MODEL = \"acestep-5Hz-lm-4B\"\n",
                "BACKEND = \"vllm\"\n",
                "DEVICE = \"cuda\"\n",
                "\n",
                "checkpoints = get_checkpoints_dir(str(CHECKPOINTS_DIR))\n",
                "ensure_dit_model(DIT_MODEL, checkpoints, prefer_source=\"huggingface_only\")\n",
                "ensure_lm_model(LM_MODEL, checkpoints, prefer_source=\"huggingface_only\")\n",
                "\n",
                "dit_handler = AceStepHandler()\n",
                "dit_handler.initialize_service(project_root=str(PROJECT_ROOT), config_path=DIT_MODEL, device=DEVICE, compile_model=True)\n",
                "\n",
                "llm_handler = LLMHandler()\n",
                "llm_handler.initialize(checkpoint_dir=str(CHECKPOINTS_DIR), lm_model_path=LM_MODEL, backend=BACKEND, device=DEVICE)\n",
                "\n",
                "print(\"✅ ACE-Step A100 servisi hazir ve tum araclar yuklendi.\")"
            ]
            cell['source'] = [line + ("" if line.endswith("\n") else "\n") for line in new_source]
            print("Cell 4 updated.")

        # Tüm hücrelerde normalizasyon değerini güncelle
        if "normalization_db=-1.0" in source:
            cell['source'] = [line.replace("normalization_db=-1.0", "normalization_db=-3.0") for line in cell['source']]
            print(f"Normalization updated in a cell.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook update complete.")

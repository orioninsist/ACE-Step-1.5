import json
from pathlib import Path

notebook_path = Path("ACE_Step_1_5_A100_Colab.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        modified = False
        for line in cell['source']:
            if "https://github.com/ACE-Step/ACE-Step-1.5.git" in line:
                new_line = line.replace("https://github.com/ACE-Step/ACE-Step-1.5.git", "https://github.com/orioninsist/ACE-Step-1.5.git")
                new_source.append(new_line)
                modified = True
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("GitHub URL updated to fork successfully!")

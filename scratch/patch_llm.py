import os

filepath = "/mnt/samsung/orion-backup-local/projects/ACE-Step-1.5/acestep/llm_inference.py"
with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "if current_key and current_value_lines:" in line and "def save_current_field" in lines[i-3]:
        new_lines.append(line.replace("if current_key and current_value_lines:", "if current_key:"))
    else:
        new_lines.append(line)

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print("Patch applied successfully via script.")

import os

filepath = "/mnt/samsung/orion-backup-local/projects/ACE-Step-1.5/acestep/core/generation/handler/generate_music_execute.py"
with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'print(f"\\n--- [DEBUG] GENERATION INPUTS ---")' in line:
        new_lines.append("            " + line.strip() + "\n")
    elif 'print(f"   ' in line and '[DEBUG]' not in line:
        new_lines.append("            " + line.strip() + "\n")
    elif 'print(f"--- [DEBUG] END ---\\n")' in line:
        new_lines.append("            " + line.strip() + "\n")
    elif 'if service_inputs[\'audio_code_hints_batch\']:' in line and 'print(' not in line:
         new_lines.append("            " + line.strip() + "\n")
    elif 'print(f"   Audio Codes Length:' in line:
         new_lines.append("                " + line.strip() + "\n")
    else:
        new_lines.append(line)

# Clean up any messed up indentation from the previous run
# Actually, I'll just write it correctly.
fixed_code = """
        def _service_target():
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
            try:
"""

# Re-read and replace the whole block
with open(filepath, 'r') as f:
    full_content = f.read()

import re
# Match from def _service_target(): until try:
pattern = r'def _service_target\(\):\n.*?try:'
fixed_content = re.sub(pattern, fixed_code, full_content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(fixed_content)

print("Indentation fixed.")

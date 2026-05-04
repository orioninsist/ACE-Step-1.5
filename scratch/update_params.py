import json

with open("ACE_Step_1_5_A100_Colab.ipynb", "r") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "enable_normalization=True," in line or "normalization_db=-3.0," in line:
                # We need to find the close parenthesis
                pass
        
        # A safer way is to just replace the whole text in the cell
        cell_text = "".join(source)
        if "params = GenerationParams(" in cell_text:
            if "dcw_enabled=False" not in cell_text:
                # Find the closing parenthesis of GenerationParams(
                # Because we know it ends with normalization_db=-3.0,\n)
                # Let's just do a string replace
                old_str = "    normalization_db=-3.0,\n)"
                new_str = "    normalization_db=-3.0,\n    dcw_enabled=False,\n    guidance_scale=5.0,\n)"
                cell_text = cell_text.replace(old_str, new_str)
                
                # Update the cell source
                # we have to split it back into lines keeping the newline characters
                new_source = []
                lines = cell_text.splitlines(True) # Keep \n
                cell['source'] = lines

with open("ACE_Step_1_5_A100_Colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook GenerationParams updated successfully.")

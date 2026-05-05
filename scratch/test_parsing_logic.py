import re
from typing import Dict, Any, Tuple

def postprocess_caption(caption: str) -> str:
    # Minimal mock
    return caption.strip()

def parse_lm_output(output_text: str) -> Tuple[Dict[str, Any], str]:
    metadata = {}
    audio_codes = ""

    # Extract audio codes - find all <|audio_code_XXX|> patterns
    code_pattern = r'<\|audio_code_\d+\|>'
    code_matches = re.findall(code_pattern, output_text)
    if code_matches:
        audio_codes = "".join(code_matches)

    # Extract metadata from reasoning section
    reasoning_patterns = [
        r'<think>(.*?)</think>',
        r'<reasoning>(.*?)</reasoning>',
    ]

    reasoning_text = None
    for pattern in reasoning_patterns:
        match = re.search(pattern, output_text, re.DOTALL)
        if match:
            reasoning_text = match.group(1).strip()
            break

    if not reasoning_text:
        lines_before_codes = output_text.split('<|audio_code_')[0] if '<|audio_code_' in output_text else output_text
        reasoning_text = lines_before_codes.strip()

    if reasoning_text:
        lines = reasoning_text.split('\n')
        current_key = None
        current_value_lines = []

        def save_current_field():
            nonlocal current_key, current_value_lines
            if current_key:
                value = '\n'.join(current_value_lines)
                if current_key == 'bpm':
                    try: metadata['bpm'] = int(value.strip())
                    except: metadata['bpm'] = value.strip()
                elif current_key == 'caption':
                    metadata['caption'] = postprocess_caption(value)
                elif current_key == 'duration':
                    try: metadata['duration'] = int(value.strip())
                    except: metadata['duration'] = value.strip()
                elif current_key == 'genres':
                    metadata['genres'] = value.strip()
                elif current_key == 'keyscale':
                    metadata['keyscale'] = value.strip()
                elif current_key == 'language':
                    metadata['language'] = value.strip()
                elif current_key == 'timesignature':
                    metadata['timesignature'] = value.strip()
                elif current_key == 'lyrics':
                    metadata['lyrics'] = value.strip()
            current_key = None
            current_value_lines = []

        meta_keys = ['bpm', 'caption', 'duration', 'genres', 'keyscale', 'language', 'timesignature', 'lyrics']
        
        for line in lines:
            if line.strip().startswith('<'): continue
            if line and not line[0].isspace() and ':' in line:
                remaining_line = line
                while ':' in remaining_line:
                    colon_idx = remaining_line.find(':')
                    potential_key = remaining_line[:colon_idx].strip().lower()
                    matched_key = None
                    for k in meta_keys:
                        if potential_key == k or potential_key.endswith(' ' + k):
                            matched_key = k
                            break
                    if matched_key:
                        save_current_field()
                        current_key = matched_key
                        remaining_line = remaining_line[colon_idx + 1:].strip()
                        next_key_idx = -1
                        for k in meta_keys:
                            search_pattern = k + ":"
                            pos = remaining_line.find(search_pattern)
                            if pos != -1 and (pos == 0 or remaining_line[pos-1].isspace()):
                                if next_key_idx == -1 or pos < next_key_idx:
                                    next_key_idx = pos
                        if next_key_idx != -1:
                            val = remaining_line[:next_key_idx].strip()
                            if val: current_value_lines.append(val)
                            remaining_line = remaining_line[next_key_idx:]
                        else:
                            if remaining_line: current_value_lines.append(remaining_line)
                            remaining_line = ""
                    else:
                        if current_key: current_value_lines.append(remaining_line)
                        remaining_line = ""
            else:
                if current_key: current_value_lines.append(line)
        save_current_field()

    return metadata, audio_codes

# Test
output_text = """<think>
bpm: 120
caption: A fast energetic song
lyrics: [Verse 1]
Hello world
This is a test
</think>
<|audio_code_123|><|audio_code_456|>"""

metadata, codes = parse_lm_output(output_text)
print(f"Extracted metadata: {metadata}")
assert metadata.get('bpm') == 120
assert metadata.get('caption') == "A fast energetic song"
assert metadata.get('lyrics') == "[Verse 1]\nHello world\nThis is a test"
assert codes == "<|audio_code_123|><|audio_code_456|>"
print("✅ Standalone lyrics parsing test passed!")

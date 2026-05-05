
import sys
import os

# Standalone test for the parser logic
def test_parser_with_merged_fields():
    # problematic output from user log
    output_text = """<think>
bpm: 45
caption:duration: 95
keyscale: E minor
timesignature: 2
</think>
<|audio_code_123|>"""

    metadata = {
        'bpm': 120,
        'caption': "",
        'duration': 30,
        'genres': "",
        'keyscale': "C major",
        'language': "unknown",
        'timesignature': "4"
    }
    
    # Extract reasoning text
    reasoning_text = output_text.split('</think>')[0].split('<think>')[-1].strip()
    
    # The logic we added:
    lines = reasoning_text.split('\n')
    current_key = None
    current_value_lines = []
    
    meta_keys = ['bpm', 'caption', 'duration', 'genres', 'keyscale', 'language', 'timesignature']

    def save_current_field():
        nonlocal current_key, current_value_lines
        if current_key and current_value_lines:
            value = '\n'.join(current_value_lines)
            if current_key == 'bpm':
                try: metadata['bpm'] = int(value.strip())
                except: metadata['bpm'] = value.strip()
            elif current_key == 'caption':
                # Simplified postprocess
                metadata['caption'] = ' '.join([l.strip() for l in value.split('\n') if l.strip()])
            elif current_key == 'duration':
                try: metadata['duration'] = int(value.strip())
                except: metadata['duration'] = value.strip()
            elif current_key == 'keyscale':
                metadata['keyscale'] = value.strip()
            elif current_key == 'timesignature':
                metadata['timesignature'] = value.strip()
        current_key = None
        current_value_lines = []

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
                        if pos != -1:
                            if pos == 0 or remaining_line[pos-1].isspace():
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
        elif line.startswith(' ') or line.startswith('\t'):
            if current_key: current_value_lines.append(line)
            
    save_current_field()
    
    print(f"Parsed Metadata: {metadata}")
    assert metadata['bpm'] == 45
    assert metadata['duration'] == 95
    assert metadata['keyscale'] == "E minor"
    assert metadata['timesignature'] == "2"
    print("Test passed!")

if __name__ == "__main__":
    test_parser_with_merged_fields()

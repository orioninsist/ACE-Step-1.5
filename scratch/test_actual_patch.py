import re
from typing import Dict, Any, Tuple
import sys

# Add the project path to sys.path to import from acestep
sys.path.append("/mnt/samsung/orion-backup-local/projects/ACE-Step-1.5")

from acestep.llm_inference import LLMHandler

def test_actual_code():
    test_input = """<think>
bpm: 45
caption:duration: 95
keyscale: E minor
timesignature: 2
</think>"""

    # We need to mock some things or just call parse_lm_output directly if it's static-ish
    # Actually, it's an instance method. Let's create a dummy handler.
    handler = LLMHandler.__new__(LLMHandler)
    
    metadata, _ = handler.parse_lm_output(test_input)
    print(f"Metadata: {metadata}")

if __name__ == "__main__":
    test_actual_code()

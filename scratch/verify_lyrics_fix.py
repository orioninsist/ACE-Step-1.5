from acestep.llm_inference import LLMHandler
import unittest
from unittest.mock import MagicMock

class TestLyricsParsing(unittest.TestCase):
    def test_parse_lm_output_with_lyrics(self):
        handler = LLMHandler()
        output_text = """<think>
bpm: 120
caption: A fast energetic song
lyrics: [Verse 1]
Hello world
This is a test
</think>
<|audio_code_123|><|audio_code_456|>"""
        
        metadata, codes = handler.parse_lm_output(output_text)
        
        self.assertEqual(metadata.get('bpm'), 120)
        self.assertEqual(metadata.get('caption'), "A fast energetic song")
        self.assertEqual(metadata.get('lyrics'), "[Verse 1]\nHello world\nThis is a test")
        self.assertEqual(codes, "<|audio_code_123|><|audio_code_456|>")
        print("✅ Lyrics parsing test passed!")

if __name__ == "__main__":
    TestLyricsParsing().test_parse_lm_output_with_lyrics()

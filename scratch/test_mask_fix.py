import torch
from unittest.mock import MagicMock
from acestep.core.generation.handler.conditioning_masks import ConditioningMaskMixin

class MockHandler(ConditioningMaskMixin):
    def __init__(self):
        self.device = "cpu"
        self.sample_rate = 48000

def test_auto_mask_fix():
    handler = MockHandler()
    batch_size = 2
    max_latent_length = 10
    instructions = ["test"] * batch_size
    audio_code_hints = [None] * batch_size
    target_wavs = None
    target_latents = None
    repainting_start = None
    repainting_end = None
    silence_latent_tiled = torch.zeros((max_latent_length, 128))
    
    # Original buggy behavior test (simulated)
    chunk_mask_modes = ["auto", "auto"]
    
    chunk_masks_tensor, spans, is_covers_tensor, src_latents, repaint_mask = handler._build_chunk_masks_and_src_latents(
        batch_size, max_latent_length, instructions, audio_code_hints,
        target_wavs, target_latents, repainting_start, repainting_end,
        silence_latent_tiled, chunk_mask_modes, task_type="text2music"
    )
    
    print(f"Mask tensor dtype: {chunk_masks_tensor.dtype}")
    print(f"Mask tensor values:\n{chunk_masks_tensor}")
    
    # Check if any value is 2.0
    has_two = (chunk_masks_tensor == 2.0).any().item()
    if has_two:
        print("SUCCESS: Found 2.0 in masks")
    else:
        print("FAILURE: 2.0 was cast to 1.0 (bool bug still present)")

if __name__ == "__main__":
    test_auto_mask_fix()

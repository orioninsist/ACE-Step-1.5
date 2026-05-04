import torch
import math

def test_ola():
    bsz, channels = 1, 2
    latent_frames = 100
    chunk_size = 30
    overlap = 5
    stride = chunk_size - 2 * overlap
    upsample_factor = 32.768
    
    total_audio_length = int(round(latent_frames * upsample_factor))
    final_audio = torch.zeros(bsz, channels, total_audio_length)
    weight_sum = torch.zeros(1, 1, total_audio_length)
    
    num_steps = math.ceil(latent_frames / stride)
    
    for i in range(num_steps):
        core_start = i * stride
        core_end = min(core_start + stride, latent_frames)
        win_start = max(0, core_start - overlap)
        win_end = min(latent_frames, core_end + overlap)
        
        latent_len = win_end - win_start
        audio_len = int(round(latent_len * upsample_factor))
        
        audio_chunk = torch.ones(bsz, channels, audio_len)
        
        win_start_audio = int(round(win_start * upsample_factor))
        
        window = torch.ones(1, 1, audio_len)
        
        fade_in_latent = core_start - win_start
        fade_out_latent = win_end - core_end
        
        fade_in_audio = int(round(fade_in_latent * upsample_factor))
        fade_out_audio = int(round(fade_out_latent * upsample_factor))
        
        if fade_in_audio > 0:
            window[..., :fade_in_audio] = torch.linspace(0, 1, fade_in_audio)
        if fade_out_audio > 0:
            window[..., -fade_out_audio:] = torch.linspace(1, 0, fade_out_audio)
            
        final_audio[:, :, win_start_audio:win_start_audio+audio_len] += audio_chunk * window
        weight_sum[:, :, win_start_audio:win_start_audio+audio_len] += window

    print("Weight sum min:", weight_sum.min().item())
    print("Weight sum max:", weight_sum.max().item())

    # Some weights might be 0 if the math doesn't cover all samples
    if weight_sum.min().item() < 1e-5:
        print("WARNING: Some samples have 0 weight!")

test_ola()

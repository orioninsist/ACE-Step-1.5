import torch
import numpy as np

def simulate(overlap=8, latent_frames=48, chunk_size=32):
    stride = chunk_size - 2 * overlap
    num_steps = int(np.ceil(latent_frames / stride))
    
    upsample_factor = 10
    total_audio_length = latent_frames * upsample_factor
    
    final_audio = torch.zeros(total_audio_length)
    weight_sum = torch.zeros(total_audio_length)
    
    for i in range(num_steps):
        core_start = i * stride
        core_end = min(core_start + stride, latent_frames)
        win_start = max(0, core_start - overlap)
        win_end = min(latent_frames, core_end + overlap)
        
        actual_len = (win_end - win_start) * upsample_factor
        window = torch.ones(actual_len)
        
        fade_in_latent = core_start - win_start
        fade_out_latent = win_end - core_end
        
        if fade_in_latent > 0:
            cf_half = fade_in_latent // 2
            fade_in_start = fade_in_latent - cf_half
            fade_in_end = fade_in_latent + cf_half
            
            start_audio = int(round(fade_in_start * upsample_factor))
            end_audio = int(round(fade_in_end * upsample_factor))
            cf_len = end_audio - start_audio
            
            if start_audio > 0:
                window[:start_audio] = 0.0
            if cf_len > 0:
                window[start_audio:end_audio] = torch.linspace(0, 1, cf_len)
                
        if fade_out_latent > 0:
            cf_half = fade_out_latent // 2
            fade_out_start = fade_out_latent + cf_half
            fade_out_end = fade_out_latent - cf_half
            
            start_audio = int(round(fade_out_start * upsample_factor))
            end_audio = int(round(fade_out_end * upsample_factor))
            cf_len = start_audio - end_audio
            
            idx_start = actual_len - start_audio
            idx_end = actual_len - end_audio
            
            if cf_len > 0:
                window[idx_start:idx_end] = torch.linspace(1, 0, cf_len)
            if end_audio > 0:
                window[idx_end:] = 0.0
                
        win_start_audio = win_start * upsample_factor
        end_idx = min(win_start_audio + actual_len, total_audio_length)
        actual_len_clamped = end_idx - win_start_audio
        
        # Audio is just 1.0 to check weights
        audio_chunk = torch.ones(actual_len_clamped)
        
        final_audio[win_start_audio:end_idx] += audio_chunk * window[:actual_len_clamped]
        weight_sum[win_start_audio:end_idx] += window[:actual_len_clamped]
        
    print(f"Weight sum min: {weight_sum.min().item()}, max: {weight_sum.max().item()}")
    
simulate()

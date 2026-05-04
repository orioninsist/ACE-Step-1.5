import torch
import math

bsz, channels = 1, 2
latent_frames = 100
chunk_size = 30
overlap = 5
stride = chunk_size - 2 * overlap
upsample_factor = 4

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
    
    audio_chunk = torch.ones(bsz, channels, audio_len) # dummy
    
    win_start_audio = int(round(win_start * upsample_factor))
    
    window = torch.ones(1, 1, audio_len)
    fade_len = int(round(overlap * upsample_factor))
    if win_start > 0:
        window[..., :fade_len] = torch.linspace(0, 1, fade_len)
    if win_end < latent_frames:
        window[..., -fade_len:] = torch.linspace(1, 0, fade_len)
        
    final_audio[:, :, win_start_audio:win_start_audio+audio_len] += audio_chunk * window
    weight_sum[:, :, win_start_audio:win_start_audio+audio_len] += window

# Check weight sum
print("Weight sum min:", weight_sum.min().item())
print("Weight sum max:", weight_sum.max().item())

final_audio = final_audio / weight_sum.clamp(min=1e-6)

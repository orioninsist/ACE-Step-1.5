"""Chunk-level VAE decode helpers used by tiled decode orchestration."""

import math

import torch
from loguru import logger
from tqdm import tqdm


class VaeDecodeChunksMixin:
    """Implement chunked decode strategies for GPU and CPU-offload modes."""

    def _tiled_decode_inner(self, latents, chunk_size, overlap, offload_wav_to_cpu):
        """Run tiled decode with adaptive overlap and OOM fallbacks."""
        bsz, _channels, latent_frames = latents.shape

        # Batch-sequential decode keeps peak VRAM stable across batch sizes.
        if bsz > 1:
            logger.info(f"[tiled_decode] Batch size {bsz} > 1; decoding samples sequentially to save VRAM")
            per_sample_results = []
            for b_idx in range(bsz):
                single = latents[b_idx : b_idx + 1]
                decoded = self._tiled_decode_inner(single, chunk_size, overlap, offload_wav_to_cpu)
                per_sample_results.append(decoded.cpu() if decoded.device.type != "cpu" else decoded)
                self._empty_cache()
            result = torch.cat(per_sample_results, dim=0)
            if latents.device.type != "cpu" and not offload_wav_to_cpu:
                result = result.to(latents.device)
            return result

        min_overlap = 4  # Minimum floor to prevent audio artifacts at chunk boundaries
        effective_overlap = overlap
        while chunk_size - 2 * effective_overlap <= 0 and effective_overlap > min_overlap:
            effective_overlap = effective_overlap // 2
        # Enforce minimum overlap floor to avoid near-zero values that cause corruption
        if effective_overlap < min_overlap and overlap >= min_overlap:
            effective_overlap = min_overlap
        if effective_overlap != overlap:
            logger.warning(
                f"[tiled_decode] Reduced overlap from {overlap} to {effective_overlap} for chunk_size={chunk_size}"
            )
        overlap = effective_overlap

        if latent_frames <= chunk_size:
            try:
                decoder_output = self.vae.decode(latents)
                result = decoder_output.sample
                del decoder_output
                return result
            except torch.cuda.OutOfMemoryError:
                logger.warning("[tiled_decode] OOM on direct decode, falling back to CPU VAE decode")
                self._empty_cache()
                return self._decode_on_cpu(latents)

        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")

        num_steps = math.ceil(latent_frames / stride)

        if offload_wav_to_cpu:
            try:
                return self._tiled_decode_offload_cpu(latents, bsz, latent_frames, stride, overlap, num_steps)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"[tiled_decode] OOM during offload_cpu decode with chunk_size={chunk_size}, "
                    "falling back to CPU VAE decode"
                )
                self._empty_cache()
                return self._decode_on_cpu(latents)

        try:
            return self._tiled_decode_gpu(latents, stride, overlap, num_steps)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                f"[tiled_decode] OOM during GPU decode with chunk_size={chunk_size}, "
                "falling back to CPU offload path"
            )
            self._empty_cache()
            try:
                return self._tiled_decode_offload_cpu(latents, bsz, latent_frames, stride, overlap, num_steps)
            except torch.cuda.OutOfMemoryError:
                logger.warning("[tiled_decode] OOM even with offload path, falling back to full CPU VAE decode")
                self._empty_cache()
                return self._decode_on_cpu(latents)

    def _tiled_decode_gpu(self, latents, stride, overlap, num_steps):
        """Decode chunks and perform Overlap-Add (OLA) on GPU."""
        bsz, _channels, latent_frames = latents.shape
        upsample_factor = None
        final_audio = None
        weight_sum = None

        disable_tqdm = getattr(self, "disable_tqdm", False)
        for i in tqdm(range(num_steps), desc="Decoding audio chunks", disable=disable_tqdm):
            core_start = i * stride
            core_end = min(core_start + stride, latent_frames)
            win_start = max(0, core_start - overlap)
            win_end = min(latent_frames, core_end + overlap)

            latent_chunk = latents[:, :, win_start:win_end]
            decoder_output = self.vae.decode(latent_chunk)
            audio_chunk = decoder_output.sample
            del decoder_output

            if upsample_factor is None:
                upsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]
                audio_channels = audio_chunk.shape[1]
                total_audio_length = int(round(latent_frames * upsample_factor))
                final_audio = torch.zeros(bsz, audio_channels, total_audio_length, dtype=audio_chunk.dtype, device=audio_chunk.device)
                weight_sum = torch.zeros(1, 1, total_audio_length, dtype=audio_chunk.dtype, device=audio_chunk.device)

            win_start_audio = int(round(win_start * upsample_factor))
            
            audio_len = audio_chunk.shape[-1]
            end_idx = min(win_start_audio + audio_len, total_audio_length)
            actual_len = end_idx - win_start_audio
            audio_chunk = audio_chunk[:, :, :actual_len]

            window = torch.ones(1, 1, actual_len, dtype=audio_chunk.dtype, device=audio_chunk.device)
            fade_in_latent = core_start - win_start
            fade_out_latent = win_end - core_end
            fade_in_audio = int(round(fade_in_latent * upsample_factor))
            fade_out_audio = int(round(fade_out_latent * upsample_factor))

            if fade_in_audio > 0:
                window[..., :fade_in_audio] = torch.linspace(0, 1, fade_in_audio, dtype=audio_chunk.dtype, device=audio_chunk.device)
            if fade_out_audio > 0:
                window[..., -fade_out_audio:] = torch.linspace(1, 0, fade_out_audio, dtype=audio_chunk.dtype, device=audio_chunk.device)

            final_audio[:, :, win_start_audio:end_idx] += audio_chunk * window
            weight_sum[:, :, win_start_audio:end_idx] += window

        return final_audio / weight_sum.clamp(min=1e-6)

    def _tiled_decode_offload_cpu(self, latents, bsz, latent_frames, stride, overlap, num_steps):
        """Decode chunks on GPU and perform Overlap-Add (OLA) on CPU."""
        upsample_factor = None
        final_audio = None
        weight_sum = None

        disable_tqdm = getattr(self, "disable_tqdm", False)
        for i in tqdm(range(num_steps), desc="Decoding audio chunks", disable=disable_tqdm):
            core_start = i * stride
            core_end = min(core_start + stride, latent_frames)
            win_start = max(0, core_start - overlap)
            win_end = min(latent_frames, core_end + overlap)

            latent_chunk = latents[:, :, win_start:win_end]
            decoder_output = self.vae.decode(latent_chunk)
            audio_chunk = decoder_output.sample
            del decoder_output

            if upsample_factor is None:
                upsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]
                audio_channels = audio_chunk.shape[1]
                total_audio_length = int(round(latent_frames * upsample_factor))
                final_audio = torch.zeros(bsz, audio_channels, total_audio_length, dtype=audio_chunk.dtype, device="cpu")
                weight_sum = torch.zeros(1, 1, total_audio_length, dtype=audio_chunk.dtype, device="cpu")

            win_start_audio = int(round(win_start * upsample_factor))
            
            audio_len = audio_chunk.shape[-1]
            end_idx = min(win_start_audio + audio_len, total_audio_length)
            actual_len = end_idx - win_start_audio
            audio_chunk = audio_chunk[:, :, :actual_len].cpu()

            window = torch.ones(1, 1, actual_len, dtype=audio_chunk.dtype, device="cpu")
            fade_in_latent = core_start - win_start
            fade_out_latent = win_end - core_end
            fade_in_audio = int(round(fade_in_latent * upsample_factor))
            fade_out_audio = int(round(fade_out_latent * upsample_factor))

            if fade_in_audio > 0:
                window[..., :fade_in_audio] = torch.linspace(0, 1, fade_in_audio, dtype=audio_chunk.dtype, device="cpu")
            if fade_out_audio > 0:
                window[..., -fade_out_audio:] = torch.linspace(1, 0, fade_out_audio, dtype=audio_chunk.dtype, device="cpu")

            final_audio[:, :, win_start_audio:end_idx] += audio_chunk * window
            weight_sum[:, :, win_start_audio:end_idx] += window

        return final_audio / weight_sum.clamp(min=1e-6)

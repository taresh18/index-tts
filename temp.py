import os
import sys
import time
import torch
import numpy as np
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indextts.infer_v2 import IndexTTS2

# Configuration
MODEL_DIR = "checkpoints"
CFG_PATH = os.path.join(MODEL_DIR, "config.yaml")
REFERENCE_AUDIO = "../maya_ref.wav"  # Reference speaker audio
OUTPUT_FILE = "output_streamed_audio.wav"
SAMPLE_RATE = 22050

# Test text
input_text = "Wow. This place looks even better than I imagined. How did they set all this up so perfectly? The lights, the music, everything feels magical. I can't stop smiling right now."

# =============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================
# Streaming parameters (affect TTFB)
QUICK_STREAMING_TOKENS = 80  # Creates smaller first segment for faster TTFB (0 = disabled, 80 recommended)
MAX_TEXT_TOKENS_PER_SEGMENT = 50  # Smaller = faster TTFB but more chunks (default: 120)

# Model optimization parameters
USE_FP16 = True           # Half precision for faster GPU inference
USE_CUDA_KERNEL = True    # Custom CUDA kernels for BigVGAN (significant speedup)
USE_TORCH_COMPILE = False # torch.compile optimization (slower startup but faster inference)
USE_DEEPSPEED = True     # DeepSpeed optimization (requires installation)
# =============================================================================

# Initialize TTS engine
print("Initializing IndexTTS2 engine...")
print(f"  use_fp16={USE_FP16}, use_cuda_kernel={USE_CUDA_KERNEL}, use_torch_compile={USE_TORCH_COMPILE}")
tts = IndexTTS2(
    cfg_path=CFG_PATH,
    model_dir=MODEL_DIR,
    use_fp16=USE_FP16,
    use_cuda_kernel=USE_CUDA_KERNEL,
    use_deepspeed=USE_DEEPSPEED,
    use_torch_compile=USE_TORCH_COMPILE
)
print("Engine initialized!\n")


def warmup_run():
    """
    Warmup run - consumes all chunks without measuring or saving anything.
    This helps warm up the model and caches.
    """
    wav_generator = tts.infer(
        stream_return=True,
        spk_audio_prompt=REFERENCE_AUDIO,
        output_path=None,  # Don't save during warmup
        text=input_text,
        verbose=False,
        max_text_tokens_per_segment=MAX_TEXT_TOKENS_PER_SEGMENT,
        more_segment_before=QUICK_STREAMING_TOKENS  # This maps to quick_streaming_tokens in infer_generator
    )
    
    # Consume all chunks
    for wav in wav_generator:
        pass


def benchmark_run():
    """
    Benchmark run - measures TTFB latency accurately.
    TTFB (Time To First Byte) is measured from when we start the generator
    until we receive the first audio chunk.
    
    Note: IndexTTS2 streaming is SEGMENT-LEVEL, not token-level.
    Each segment goes through: GPT → s2mel → bigvgan before yielding.
    To reduce TTFB:
    - Use smaller max_text_tokens_per_segment
    - Set quick_streaming_tokens > 0 for smaller first segment
    """
    # Start timing right before we start the generator
    ttfb_start = time.time()
    first_chunk_received = False
    chunk_count = 0
    total_audio_samples = 0
    all_audio_chunks = []
    ttfb_latency = None
    
    print(f"Streaming config: max_tokens_per_segment={MAX_TEXT_TOKENS_PER_SEGMENT}, quick_streaming_tokens={QUICK_STREAMING_TOKENS}")
    
    # Start the generator - TTFB timing starts here
    wav_generator = tts.infer(
        stream_return=True,
        spk_audio_prompt=REFERENCE_AUDIO,
        output_path=None,  # Don't save to file during benchmark (we'll do it manually)
        text=input_text,
        verbose=False,
        max_text_tokens_per_segment=MAX_TEXT_TOKENS_PER_SEGMENT,
        more_segment_before=QUICK_STREAMING_TOKENS  # This maps to quick_streaming_tokens in infer_generator
    )
    
    for wav in wav_generator:
        if not first_chunk_received:
            # TTFB is measured when we receive the first audio chunk
            # This is pure audio data, no file I/O involved
            ttfb_latency = time.time() - ttfb_start
            print(f"🎯 TTFB (Time To First Byte) latency: {ttfb_latency*1000:.2f} ms")
            first_chunk_received = True
        
        chunk_count += 1
        
        # Convert tensor to numpy if needed
        if isinstance(wav, torch.Tensor):
            wav_np = wav.numpy().flatten()
        else:
            wav_np = wav.flatten()
        
        total_audio_samples += len(wav_np)
        all_audio_chunks.append(wav_np)
        print(f"Received chunk {chunk_count}: {len(wav_np)} samples ({len(wav_np)/SAMPLE_RATE:.3f} seconds)")
    
    # All streaming is complete, now calculate statistics
    total_time = time.time() - ttfb_start
    total_audio_duration = total_audio_samples / SAMPLE_RATE
    
    print(f"\n📊 Streaming Statistics:")
    print(f"  Total chunks received: {chunk_count}")
    print(f"  Total audio duration: {total_audio_duration:.3f} seconds")
    print(f"  Total streaming time: {total_time:.3f} seconds")
    if total_time > 0:
        print(f"  Real-time factor: {total_audio_duration/total_time:.2f}x")
    if chunk_count > 0:
        print(f"  Average chunk size: {total_audio_samples/chunk_count:.0f} samples")
    
    # Save audio AFTER all measurements are complete
    if all_audio_chunks:
        full_audio = np.concatenate(all_audio_chunks)
        
        # The audio is already in int16 format (scaled by 32767)
        # Convert to float32 for soundfile
        full_audio_float = full_audio.astype(np.float32) / 32767.0
        
        # Clip to prevent any overflow
        full_audio_float = np.clip(full_audio_float, -1.0, 1.0)
        
        # Save to WAV file
        sf.write(OUTPUT_FILE, full_audio_float, SAMPLE_RATE)
        print(f"\n💾 Saved audio to: {OUTPUT_FILE}")
        print(f"   Audio length: {len(full_audio)/SAMPLE_RATE:.3f} seconds")
    
    return ttfb_latency


def run_benchmark():
    """
    Run 5 warmup runs followed by the actual benchmark.
    """
    print(f"=" * 60)
    print(f"IndexTTS2 Streaming TTFB Benchmark")
    print(f"=" * 60)
    print(f"Input text length: {len(input_text)} characters")
    print(f"Reference audio: {REFERENCE_AUDIO}")
    print(f"\n⚙️  Optimization Settings:")
    print(f"  - use_fp16: {USE_FP16}")
    print(f"  - use_cuda_kernel: {USE_CUDA_KERNEL}")
    print(f"  - use_torch_compile: {USE_TORCH_COMPILE}")
    print(f"  - max_text_tokens_per_segment: {MAX_TEXT_TOKENS_PER_SEGMENT}")
    print(f"  - quick_streaming_tokens: {QUICK_STREAMING_TOKENS}")
    print(f"\n🔥 Running 5 warmup runs...")
    
    # Warmup runs
    for i in range(5):
        print(f"  Warmup run {i+1}/5...", end=" ", flush=True)
        warmup_run()
        print("✓")
    
    print(f"\n📊 Starting benchmark run...\n")
    
    # Actual benchmark run
    ttfb_latency = benchmark_run()
    
    if ttfb_latency:
        print(f"\n✅ Benchmark complete! TTFB: {ttfb_latency*1000:.2f} ms")


if __name__ == "__main__":
    run_benchmark()

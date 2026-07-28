import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from fla.layers import DeltaNet
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from flash_attn import flash_attn_func
from time import time

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class SwiGLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Standard LLaMA-style hidden dimension scaling
        intermediate_size = int(8 * hidden_size / 3)
        # Round up to multiple of 256 for optimal Tensor Core usage
        intermediate_size = (intermediate_size + 255) // 256 * 256
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class FlashAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Output shape: [Batch, Seq, Heads, Head_dim] 
        # No transpose needed!
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        # flash_attn_func handles the exact causal scaling internally
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        
        # out shape is [Batch, Seq, Heads, Head_dim]
        # Just collapse the last two dimensions back to Hidden_dim
        out = out.reshape(B, L, D)
        
        return self.o_proj(out)
    
class HybridTransformerBlock(nn.Module):
    def __init__(self, config, is_flash=False):
        super().__init__()
        self.is_flash = is_flash
        self.norm1 = RMSNorm(config.hidden_size)
        
        if is_flash:
            self.attn = FlashAttention(config.hidden_size, config.num_heads)
        else:
            self.attn = DeltaNet(
                hidden_size=config.hidden_size, 
                num_heads=config.num_heads,
                use_gate=True
            )
            
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        if not self.is_flash and isinstance(attn_out, tuple): 
            attn_out = attn_out[0]
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class DummyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 3 DeltaNet : 1 FlashAttention (Every 4th layer is FlashAttention)
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(config, is_flash=((i + 1) % 4 == 0)) 
            for i in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward_features(self, x):
        x = self.embed(x)
        for block in self.blocks:
            if self.training:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(self, x):
        return self.lm_head(self.forward_features(x))

def resolve_precision():
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def build_loss_fn(args):
    if LigerFusedLinearCrossEntropyLoss is None:
        raise ImportError("liger-kernel is not available. Please install it to run this benchmark.")

    liger_loss = LigerFusedLinearCrossEntropyLoss()

    def compute_loss(model, x, targets):
        hidden = model.forward_features(x)
        return liger_loss(
            model.lm_head.weight,
            hidden.reshape(-1, args.hidden_size),
            targets.reshape(-1),
        )

    return compute_loss

def run_benchmark(args):
    device = torch.device("cuda")
    dtype = resolve_precision()
    
    print(f"Initializing {args.num_layers}-layer DeltaNet Transformer ({dtype})...")
    model = DummyTransformer(args).to(device=device, dtype=dtype)
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params / 1e6:.2f}M")
    model.train()
    compute_loss = build_loss_fn(args)
    print("Loss kernel: liger_fused")
    print("Gradient checkpointing: enabled")

    # Generate dummy input tokens and shifted targets
    # Shapes: Inputs [B, L], Targets [B, L]
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    
    print(f"Warming up (Seq Len: {args.seq_len}, Vocab: {args.vocab_size})...")
    for _ in range(args.warmup):
        loss = compute_loss(model, x, targets)
        loss.backward()
        model.zero_grad()

    # Reset metrics
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    
    print(f"Running benchmark for {args.iters} iterations...")
    start_time = time()
    for i in range(args.iters):
        start_events[i].record()
        
        loss = compute_loss(model, x, targets)
        loss.backward()
        
        end_events[i].record()
        model.zero_grad()
        
    torch.cuda.synchronize()
    end_time = time()

    print(f"Total time for {args.iters} iterations: {end_time - start_time:.2f} seconds")

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time_ms = sum(times) / len(times)
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    print("=" * 50)
    print(f"RESULTS: DELTANET | {args.num_layers} Layers | {args.seq_len} Tokens | liger_fused")
    print("=" * 50)
    print(f"Avg Time per Step (Fwd+Bwd): {avg_time_ms:.2f} ms")
    print(f"Peak VRAM Allocated:         {peak_mem_gb:.4f} GB")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=60000)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    
    run_benchmark(args)

import torch


@torch.no_grad()
def encode_long_sequence(model, x: torch.Tensor, max_patches_per_forward: int = 256) -> torch.Tensor:
    """
    Encodes an arbitrarily long sequence by chunking it to prevent VRAM OOM.
    x shape: [B, P, T, 5]
    """
    model.eval()
    B, P, T, num_attr = x.shape
    
    all_indices = []
    
    # Process in chunks
    for i in range(0, P, max_patches_per_forward):
        x_chunk = x[:, i : i + max_patches_per_forward, :, :]
        indices_chunk = model.encode(x_chunk) 
        all_indices.append(indices_chunk)
        
    return torch.cat(all_indices, dim=1)
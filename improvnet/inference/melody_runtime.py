"""Fixed inference defaults and a persistent precision/attention context."""
from contextlib import contextmanager
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

INFERENCE_SEED = 42
TOP_K = 50


@contextmanager
def inference_context(device):
    """Keep BF16 weight caching and avoid cuDNN's dynamic-shape decode overhead.

    The coordinator owns this context when workers proxy their GPU requests.
    Backend flags are restored on exit; other processes, including training,
    have independent settings.
    """
    if device.type != 'cuda':
        yield
        return
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]), torch.amp.autocast('cuda', dtype=dtype):
        yield

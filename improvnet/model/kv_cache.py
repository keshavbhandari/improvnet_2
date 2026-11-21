"""Request-owned, bounded inference buffers; never parameters or checkpoint state."""
import torch


class StaticKVCache:
    """Append new keys/values in place while exposing only the occupied prefix."""
    def __init__(self, key, value, capacity):
        if torch.is_grad_enabled():
            raise RuntimeError('Static KV caches require no_grad/inference_mode')
        if key.shape != value.shape or key.ndim != 4 or capacity < key.size(1):
            raise ValueError('Invalid KV cache shape or capacity')
        shape = (key.size(0), capacity, *key.shape[2:])
        self.storage = key.new_empty((2, *shape))
        self.key, self.value = self.storage.unbind(0)
        self.length = 0
        self.append(key, value)

    def append(self, key, value):
        if torch.is_grad_enabled():
            raise RuntimeError('Static KV caches require no_grad/inference_mode')
        end = self.length + key.size(1)
        if (key.shape != value.shape or key.size(0) != self.key.size(0)
                or key.shape[2:] != self.key.shape[2:] or end > self.key.size(1)):
            raise ValueError('KV cache capacity exceeded or append shape differs')
        torch.stack((key, value), out=self.storage[:, :, self.length:end])
        self.length = end
        return self[0], self[1]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.key, self.value)[index][:, :self.length]

    def __iter__(self):
        return iter((self[0], self[1]))


def reserve_kv_cache(cache, capacity):
    """Convert each prefilled layer once; empty scripted caches remain supported."""
    return [layer if isinstance(layer, StaticKVCache) else StaticKVCache(*layer, capacity)
            for layer in cache]

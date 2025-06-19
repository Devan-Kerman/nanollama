"""Async GPU iterator for JAX that prefetches and copies data asynchronously."""

import jax
from typing import Iterator, Any, Optional, Generator
import numpy as np
from collections import deque
import time


def device_prefetch(iterator: Iterator[Any],
                      buffer_size: int = 1,
                      device: Optional[jax.Device] = None) -> Generator[Any, None, None]:
    """Create an async GPU iterator that prefetches and copies numpy arrays to GPU.
    
    Args:
        iterator: Parent iterator to fetch from
        buffer_size: Number of items to prefetch (default 1)
        device: JAX device to copy to (default: default device)
        
    Yields:
        Data with numpy arrays copied to GPU
    """
    device = device or jax.devices()[0]
    buffer = deque()
    
    def copy_to_device(data):
        """Schedule async copy of numpy arrays to device.
        
        JAX will automatically do async transfers when possible.
        The key is to not call block_until_ready() immediately.
        """
        # This creates DeviceArrays but doesn't block
        return jax.tree.map(
            lambda x: jax.device_put(x, device) if isinstance(x, np.ndarray) else x,
            data,
            is_leaf=lambda x: isinstance(x, np.ndarray)
        )
    
    for i in range(buffer_size):
        try:
            buffer.append(copy_to_device(next(iterator)))
        except StopIteration:
            break
    
    while buffer:
        # Pop the oldest copy from buffer
        result = buffer.popleft()

        _ = yield result

        # Schedule a new async copy if iterator not exhausted
        try:
            buffer.append(copy_to_device(next(iterator)))
        except StopIteration:
            pass

        yield None


if __name__ == "__main__":
    import time

    # Pre-generate data to avoid numpy generation overhead
    print("Pre-generating test data...")
    pre_gen_start = time.time()
    test_data = {
        'input_ids': np.random.randn(8, 4096).astype(np.float32),
        'attention_mask': np.random.randn(8, 4096, 4096).astype(np.float32),
    }
    print(f"Test data generated in {time.time() - pre_gen_start:.2f}s")
    print(f"  input_ids size: {test_data['input_ids'].nbytes / 1e6:.1f} MB")
    print(f"  attention_mask size: {test_data['attention_mask'].nbytes / 1e9:.1f} GB")
    
    # Test with large buffers and data
    def generate_large_batches(n_batches=100):
        """Generate batches similar to transformer training data."""
        for i in range(n_batches):
            # Reuse same arrays to test async behavior
            yield {
                'input_ids': test_data['input_ids'],
                'attention_mask': test_data['attention_mask'],
                'batch_idx': i
            }
    
    print("Testing async GPU iterator with large buffers...")
    print(f"Device: {jax.devices()[0]}")
    
    # Also run the original timing test
    print("\nTiming test with different buffer sizes:")
    # Process batches
    iterator = device_prefetch(generate_large_batches(), buffer_size=5)
    import jax.numpy as jnp

    @jax.jit
    def expensive_jax(x, attention_mask):
        for _ in range(1000):  # Reduced iterations for testing
            x = jnp.einsum("bn,bmn->bm", x, attention_mask)
            x = jnp.einsum("bm,bmn->bn", x, attention_mask)
        return jnp.square(x.sum(axis=1)).sum()

    start_time = time.time()
    for batch in iterator:
        print(f"================ Batch: {time.time() - start_time:.2f}s ================")

        value = expensive_jax(batch['input_ids'], batch['attention_mask'])

        iterator.send(None)

        start_waiting = time.time()
        value.block_until_ready()
        print(f"Exec Waiting time: {time.time() - start_waiting:.2f}s")

        start_time = time.time()

    print("\nTest completed!")
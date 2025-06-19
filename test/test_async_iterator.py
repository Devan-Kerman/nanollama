"""Tests for async GPU iterator."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from ueaj.data.prefetch import device_prefetch


def test_simple_array_copy():
    """Test basic numpy array copying to GPU."""
    data = [np.ones((10, 10)) for _ in range(3)]
    
    results = []
    for item in device_prefetch(iter(data)):
        assert isinstance(item, jax.Array)
        assert item.device() == jax.devices()[0]
        results.append(item)
    
    assert len(results) == 3
    assert all(jnp.allclose(r, 1.0) for r in results)


def test_nested_structure():
    """Test copying nested structures with mixed types."""
    data = [
        {
            'x': np.ones((5, 5)),
            'y': np.zeros(10),
            'metadata': {'id': i, 'name': f'batch_{i}'},
            'list': [np.array([1, 2, 3]), 'text', 42]
        }
        for i in range(2)
    ]
    
    results = []
    for batch in device_prefetch(iter(data)):
        # Check numpy arrays are converted
        assert isinstance(batch['x'], jax.Array)
        assert isinstance(batch['y'], jax.Array)
        assert isinstance(batch['list'][0], jax.Array)
        
        # Check non-arrays are preserved
        assert isinstance(batch['metadata']['id'], int)
        assert isinstance(batch['metadata']['name'], str)
        assert batch['list'][1] == 'text'
        assert batch['list'][2] == 42
        
        results.append(batch)
    
    assert len(results) == 2


def test_tuple_input():
    """Test with tuples instead of dicts."""
    data = [(np.ones(5), np.zeros(3), i) for i in range(3)]
    
    for i, (x, y, idx) in enumerate(device_prefetch(iter(data))):
        assert isinstance(x, jax.Array)
        assert isinstance(y, jax.Array)
        assert idx == i
        assert jnp.allclose(x, 1.0)
        assert jnp.allclose(y, 0.0)


def test_buffer_prefetch():
    """Test that buffering works correctly."""
    # Create iterator that tracks when next() is called
    call_count = 0
    
    def counting_iterator():
        nonlocal call_count
        for i in range(5):
            call_count += 1
            yield {'data': np.array([i]), 'count': call_count}
    
    # When we create the async iterator, it should immediately prefetch buffer_size items
    results = list(device_prefetch(counting_iterator(), buffer_size=2))
    
    assert len(results) == 5
    assert all(isinstance(r['data'], jax.Array) for r in results)


def test_empty_iterator():
    """Test handling of empty iterator."""
    data = []
    result = list(device_prefetch(iter(data)))
    assert result == []


def test_single_item():
    """Test with single item iterator."""
    data = [np.ones((3, 3))]
    result = list(device_prefetch(iter(data)))
    assert len(result) == 1
    assert isinstance(result[0], jax.Array)


def test_no_numpy_arrays():
    """Test with data containing no numpy arrays."""
    data = [{'a': 1, 'b': [2, 3], 'c': 'text'} for _ in range(2)]
    
    results = list(device_prefetch(iter(data)))
    assert len(results) == 2
    assert results == data  # Should be unchanged


def test_mixed_array_types():
    """Test that JAX arrays are not re-copied."""
    jax_array = jnp.ones((2, 2))
    numpy_array = np.zeros((2, 2))
    
    data = [{'jax': jax_array, 'numpy': numpy_array}]
    
    for batch in device_prefetch(iter(data)):
        # JAX array should be unchanged
        assert batch['jax'] is jax_array
        # Numpy array should be converted
        assert isinstance(batch['numpy'], jax.Array)
        assert jnp.allclose(batch['numpy'], 0.0)


if __name__ == "__main__":
    pytest.main([__file__])
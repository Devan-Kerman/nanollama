"""Debug slicing issues."""

import jax.numpy as jnp

# Test array shape (4, 8, 16, 2)
arr = jnp.ones((4, 8, 16, 2))

print("Original shape:", arr.shape)
print()

# Test 1: MLP pattern [:, batch[:], batch[:], :]
print("Test 1: MLP pattern")
print("Expected: 8 slices of shape (8, 16)")
for i in range(4):
    for j in range(2):
        sliced = arr[i, :, :, j]
        print(f"  arr[{i}, :, :, {j}].shape = {sliced.shape}")
print()

# Test 2: batch[:, :] pattern
print("Test 2: batch[:, :], :, :] pattern")
print("Expected: 32 slices of shape (4, 8)")
print("Pattern means: keep first 2 dims together, split last 2")

# The pattern batch[:, :] means treat first 2 dimensions as a unit
# Then split the remaining dimensions
for i in range(16):  # dim 2
    for j in range(2):   # dim 3
        sliced = arr[:, :, i, j]
        print(f"  arr[:, :, {i}, {j}].shape = {sliced.shape}")
        if i == 2 and j == 0:
            print("  ... (showing first few)")
            break
print()

# What's happening in the code?
print("Current implementation issue:")
print("- For MLP: it's slicing arr[i, :, :, j] but something's wrong")
print("- For batch[:,:]: it's not creating enough optimizers")
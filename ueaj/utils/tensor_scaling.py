import jax
import jax.numpy as jnp

def sq_norm(x):
	return jnp.sum(jnp.square(x.astype(jnp.float32)))

def precision_aware_update(params, update, target_lr, rtol=1e-2, max_iters=32):
	"""
	Apply update scaled to achieve target norm reduction despite precision limits.

	Args:
		params: Current parameters
		update: Normalized update direction (e.g., from muon)
		target_lr: Desired learning rate (norm reduction factor)
		rtol: Relative tolerance for achieving target norm
		max_iters: Maximum binary search iterations

	Returns:
		new_params, actual_norm_change
	"""
	# Binary search for scaling factor
	alpha_low = jnp.float32(0.0)
	alpha_high = jnp.float32(1.0)  # Assuming normalized update

	target_norm = target_lr * sq_norm(update)

	assert max_iters > 0
	for i in range(max_iters):
		alpha = (alpha_low + alpha_high) / 2

		# Apply update and measure actual change
		new_params = (params.astype(update.dtype) + alpha * update).astype(params.dtype)
		actual_norm = sq_norm(new_params - params)

		# Check if we're close enough
		relative_error = jnp.abs(actual_norm - target_norm) / target_norm

		print(f"===== {i} ====")
		print(f"alpha:\t{alpha:.5f}")
		print(f"error:\t{relative_error*100:.1f}%")
		print(actual_norm, target_norm)

		if relative_error < rtol:
			break

		# Adjust search bounds
		if actual_norm < target_norm:
			alpha_low = alpha
		else:
			alpha_high = alpha

	return new_params, actual_norm, target_norm

if __name__ == "__main__":
	# Example usage
	rng = jax.random.PRNGKey(0)
	params = jax.random.normal(rng, (1024, 1024), jnp.float32).astype(jnp.float8_e4m3fn)
	update = jax.random.normal(rng, (1024, 1024), jnp.float32)

	target_lr = 2**-32
	new_params, update_norm, target_norm = precision_aware_update(params, update, target_lr)

	default_norm = sq_norm((params.astype(jnp.float32) + target_lr*update).astype(jnp.float8_e4m3fn) - params)

	print("============== Results ==============")
	print(f"Iterative error:\t{(update_norm - target_norm) / target_norm * 100:.1f}%")
	print(f"Default error:\t\t{(default_norm - target_norm) / target_norm * 100:.1f}%")

	print(f"Update norm:\t\t{update_norm:.3f}")
	print(f"Target norm:\t\t{target_norm:.3f}")
	print(f"Default norm:\t\t{default_norm:.3f}")
import itertools
import os
from operator import mul

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import functools
from typing import Optional, Tuple, Any

import jax

import wandb
import uuid

run_id = uuid.uuid4()
wandb.init(project="nanogpt-euclidian", name=f"run-{run_id}")

import numpy as np
from flax import nnx
from flax.nnx import rnglib as rng
from transformer_engine.jax.attention import SequenceDescriptor
import time

from ueaj import data, model
from ueaj.opt import chunked_softmax_cross_entropy
import gc
from jax import numpy as jnp
import optax

@nnx.jit
def test(
	model: model.LlamaModel,
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None
):
	kwargs: dict[str, Any] = {}  # {'implementation': 'cudnn'}
	kwargs['sequence_descriptor'] = SequenceDescriptor.from_seqlens((inputs != pad_token_id).sum(axis=1))
	activations = model.get_activations(inputs, **kwargs)

	token_loss, loss_mask = chunked_softmax_cross_entropy(
		inputs,
		activations,
		model.get_logits,
		document_ids=document_ids,
		pad_token_id=pad_token_id,
		return_loss_mask=True
	)
	mask = loss_mask.sum(dtype=jnp.float32)
	return token_loss.sum() / mask.sum()

@nnx.value_and_grad(has_aux=True)
def grad(
	model: model.LlamaModel,
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None
) -> Tuple[jax.Array, Any]:
	kwargs: dict[str, Any] = {}  # {'implementation': 'cudnn'}
	kwargs['sequence_descriptor'] = SequenceDescriptor.from_seqlens((inputs != pad_token_id).sum(axis=1))
	activations = model.get_activations(inputs, **kwargs)

	token_loss, loss_mask = chunked_softmax_cross_entropy(
		inputs,
		activations,
		model.get_logits,
		document_ids=document_ids,
		pad_token_id=pad_token_id,
		return_loss_mask=True
	)
	mask = loss_mask.sum(dtype=jnp.float32)
	return token_loss.sum() / jnp.sqrt(mask), (token_loss.mean(), token_loss.std())

main_optimizer = optax.lion(learning_rate=1e-3, b1=1, b2=.9, weight_decay=1e-2)

def k_eff(W):
	if W.ndim < 2:
		return 1
	n, m = W.shape
	# careful with this one
	# squaring number only requires slightly more exponent bits
	W = W.astype(jnp.bfloat16)
	# need more mantissa for accum
	f2 = jnp.sum(jnp.square(W), dtype=jnp.float32)
	s4 = jnp.sum(jnp.square(W @ W.T), dtype=jnp.float32)
	return m * s4 / (f2**2)

def compute_k_eff(W: jax.Array):
	idx = W.shape.index(768)

	new_shape = (W.shape[0],)

	pre = functools.reduce(mul, W.shape[1:idx], 1)
	if pre > 1:
		new_shape += (pre,)

	new_shape += (768,)

	post = functools.reduce(mul, W.shape[idx+1:], 1)
	if post > 1:
		new_shape += (post,)

	W = W.reshape(new_shape)
	return jax.lax.scan(lambda _, x: (None, k_eff(x)), None, W)[1]

@functools.partial(jax.jit, donate_argnums=(1, 2))
def train_step(
	g_def: nnx.GraphDef[model.LlamaModel],
	state: nnx.State,
	opt_state: optax.OptState,
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None,
	lr: float = 1e-4,
	mom: float = .995
):
	model_instance = nnx.merge(g_def, state)
	(loss, (mean_loss, std_loss)), dmodel = grad(model_instance, inputs, document_ids, pad_token_id)
	grad_norm = jax.tree.map(lambda dt: dt.std(axis=range(1, dt.ndim)), dmodel.layers)

	params = nnx.state(model_instance, nnx.Param)
	k_eff_par = jax.tree.map(compute_k_eff, params.layers)

	opt = optax.lion(learning_rate=lr, b1=2*mom-1, b2=mom, weight_decay=1e-3)
	dmodel, opt_state = opt.update(dmodel, opt_state, params)
	params = optax.apply_updates(params, dmodel)
	state = nnx.merge_state(nnx.state(model_instance, nnx.Not(nnx.Param)), params)

	k_eff_mom = jax.tree.map(compute_k_eff, opt_state[0].mu.layers)
	return state, opt_state, (mean_loss, std_loss, grad_norm, k_eff_mom, k_eff_par)


import datasets
import transformers

model_name = "meta-llama/Llama-3.2-1B"

print("Loading tokenizer...")
tokenizer = transformers.GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.max_length = 1024*100
pad_token = 65535

print("Vocab Size:", tokenizer.vocab_size)

print("Loading dataset...")
dataset = datasets.load_dataset(
	"HuggingFaceFW/fineweb-edu",
	name="sample-10BT",
	split="train",
	streaming=True,
)

print("Setting up train iterator...")

def tokenize(examples):
	tokenized = tokenizer(examples["text"])['input_ids']
	return {'tokens': tokenized}


def tokens(dataset):
	for ex in dataset:
		yield np.array(ex["tokens"], dtype=np.int32)

batch_size, seq_len = 12, 4096

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.select_columns("tokens")
test_ds = itertools.cycle(dataset.take(1000))
def chunkify(dataset):
	dataset = tokens(dataset)
	dataset = data.pack_documents(dataset, max_length=seq_len, min_fill_ratio=.99, pad_token_id=pad_token)
	dataset = data.batch_iterator(dataset, batch_size=batch_size, drop_last=True, collate_fn=data.tuple_collate)
	dataset = data.device_prefetch(dataset, buffer_size=100)
	return dataset

dataset = chunkify(dataset)
test_data = next(chunkify(test_ds))
del test_ds

tokens = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)
document_ids = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

print("Loading model...")
tensor_config = model.ParamConfig("", group=nnx.Param).with_dtype(jnp.bfloat16)
def leaky_relu_sq(x):
	return jnp.where(x < 0, -.0625, 1) * jnp.square(x)

config = model.LlamaConfig(
	vocab_size=65536,
	model_d=768,
	num_layers=12,
	tensor_config=tensor_config,
	layer_config=model.TransformerLayerConfig(
		model_d=768,
		use_gated_mlp=False,
		attention_config=model.AttentionConfig(
			_fused=False,
			model_d=768,
			kq_d=64,
			v_head_d=64,
			kv_heads=6,
			kv_q_ratio=2,
			rope_theta=10_000.0,
			param_config=tensor_config,
			act_fn=leaky_relu_sq
		),
		mlp_config=model.MLPConfig(
			model_d=768,
			hidden_d=768*4,
			param_config=tensor_config,
			activation_fn=leaky_relu_sq
		),
		norm_config=model.RMSNormConfig(
			model_d=768,
			scale="centered"
		)
	),
	_norm_config=model.RMSNormConfig(
		model_d=768,
		scale="centered"
	)
)
model = model.LlamaModel(config, rngs=rng.Rngs(0))

graph_def, state = nnx.split(model, nnx.Param)
opt_state = main_optimizer.init(state)

print(config)

print("Compiling train step...")
print("	Tracing..")
step = train_step.trace(
	graph_def,
	state,
	opt_state,
	tokens,
	document_ids=document_ids,
	pad_token_id=pad_token,
	lr=1e-4,
	mom=.995
)

print("	Lowering...")
step = step.lower()

print("	Compiling....")
train_step = step.compile()

cost = train_step.cost_analysis()
if cost and 'flops' in cost:
	print(f"Estimated FLOPs:	{cost['flops'] * 1e-12:.2f} TFLOPs")
	tflops = cost['flops'] * 1e-12
else:
	tflops = None

# opt_state[0].mu
cost = train_step.memory_analysis()
if cost is not None:
	print(f"Peak Training VRAM:	{cost.temp_size_in_bytes * 1e-9:.2f}GB")
	print(f"Parameter VRAM:		{cost.output_size_in_bytes * 1e-9:.2f}GB")
	print(f"Total VRAM:			{cost.temp_size_in_bytes * 1e-9 + cost.output_size_in_bytes * 1e-9:.2f}GB")

lr, n, k_eff, scale = 1e-3, 10, 0, 0
print("Starting training...")
for i, batch in enumerate(dataset):
	tokens, doc_ids = batch
	if i == 0:
		gc.collect()

	start_train = time.time()
	state, opt_state, (mean_loss, std_loss, stats, k_eff_mom, k_eff_par) = train_step(
		graph_def, state, opt_state, tokens, document_ids=doc_ids, pad_token_id=pad_token, lr=lr/n, mom=1 - 1/n
	)

	dataset.send(None)
	gc.collect()

	start_wait = time.time()
	mean_loss.block_until_ready()
	end_wait = time.time()

	train_time, wait_time = end_wait - start_train, end_wait - start_wait

	new_k_eff = k_eff_mom.mlp.up_proj.w_1.value.mean()
	if jnp.isnan(new_k_eff):
		print("[Warn] NaN detected in k_eff!")
	elif i < 10:
		diff = new_k_eff - k_eff
		scale = .9 * scale + .1 * diff**2
	else:
		diff = new_k_eff - k_eff
		scale = .95 * scale + .05 * diff**2
		n = max(n + jnp.clip(diff / jnp.sqrt(max(scale, 1e-3)), -2, 2) + .01, 10.0)
		k_eff = new_k_eff

	if wait_time < .01:
		print("[Warn] Training is outpacing data loading!")

	wandb_dict = {
		"step": i,
		"train_loss": float(mean_loss),
		"std_loss": float(std_loss),
		"train_time": train_time,
		"wait_time": wait_time,
		"effective_batch_size": n,
		"learning_rate": lr / n,
		"momentum": 1 - 1/n
	}

	if i % 25 == 0:
		model = nnx.merge(graph_def, state)
		t_loss = test(model, test_data[0], test_data[1], pad_token)
		wandb_dict["test_loss"] = float(t_loss)
		print(f"Test loss: {float(t_loss):.2f}, Train time: {train_time:.2f}s, Wait time: {wait_time:.2f}s")


	for key, value in nnx.to_flat_state(k_eff_mom):
		wandb_dict["k_eff-mom-" + ".".join(key)] = value.value.mean()

	for key, value in nnx.to_flat_state(k_eff_par):
		wandb_dict["k_eff-par-" + ".".join(key)] = value.value.mean()

	for key, value in nnx.to_flat_state(stats):
		wandb_dict["gnorm-" + ".".join(key)] = value.value.mean()

	wandb.log(wandb_dict)
	print(f"[{i}] Train loss: {float(mean_loss):.2f}, Std loss: {float(std_loss):.2f}, Effective batch size: {n:.1f}, Learning rate: {lr / n:.2e}, Momentum: {1 - 1/n:.2f}")
	if np.isnan(mean_loss):
		print("Loss is NaN, stopping training...")
		break
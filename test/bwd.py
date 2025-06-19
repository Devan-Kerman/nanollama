import jax
from ueaj.model.llama import llama

def test_mlp_bwd():
	config = llama.LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
	print(config)
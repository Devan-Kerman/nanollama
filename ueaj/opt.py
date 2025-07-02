import dataclasses
import re
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import optax
from flax import nnx
from flax.core import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

class OptimizerManager:
    """
    A flexible optimizer manager that allows assigning different optax optimizers
    to different parts of a model's parameter PyTree using NumPy-style slicing.
    """

    def __init__(self):
        self.rules: List[Tuple[Any, optax.GradientTransformation]] = []
        self.optimizer: optax.GradientTransformation | None = None

    def __setitem__(self, key: Any, value: optax.GradientTransformation):
        """
        Defines a rule to assign an optimizer to a part of the model.

        Args:
            key: A NumPy-style slice or path tuple to specify which parameters
                 this optimizer applies to.
            value: The optax.GradientTransformation to apply.
        """
        if not isinstance(value, optax.GradientTransformation):
            raise TypeError(f"Value must be an optax.GradientTransformation, got {type(value)}")
        self.rules.append((key, value))

    def _slice_to_regex(self, key: Any) -> str:
        """Converts a slice key into a regex pattern."""
        if not isinstance(key, tuple):
            key = (key,)

        regex_parts = []
        for item in key:
            if isinstance(item, str):
                regex_parts.append(re.escape(item))
            elif isinstance(item, int):
                regex_parts.append(str(item))
            elif item is Ellipsis:
                regex_parts.append(".*")
            elif isinstance(item, slice) and item == slice(None, None, None):
                regex_parts.append("[^/]+")  # Match any single path component
            elif isinstance(item, list):
                regex_parts.append(f"({'|'.join(re.escape(x) for x in item)})")
            else:
                raise TypeError(f"Unsupported slice type: {type(item)}")
        
        # Match from the start of the path, with '/' as separator
        return "^" + "/".join(regex_parts)

    def build(self, params: nnx.State) -> optax.GradientTransformation:
        """
        Builds the final optax.multi_transform optimizer from the defined rules.

        Args:
            params: The PyTree of model parameters to be optimized.

        Returns:
            A unified optax.GradientTransformation.
        """
        if not self.rules:
            raise ValueError("No optimizer rules have been defined. "
                             "Please set at least a default rule, e.g., `opt[...] = optax.adamw(...)`.")

        partition_optimizers: Dict[str, optax.GradientTransformation] = {}
        partitioner_rules: List[Tuple[str, str]] = []
        fallback = None

        # Process rules, giving priority to more specific (later) rules.
        # We use the rule index as the label.
        for i, (key, optimizer) in enumerate(self.rules):
            label = str(i)
            partition_optimizers[label] = optimizer
            if key is Ellipsis:
                fallback = label
            else:
                regex = self._slice_to_regex(key)
                partitioner_rules.append((label, regex))

        if fallback is None:
            raise ValueError("A default optimizer rule `opt[...] = ...` is required.")

        # More specific regexes (longer) should be checked first.
        partitioner_rules.sort(key=lambda x: len(x[1]), reverse=True)

        partitioner = nnx.Partitioner(partitioner_rules, fallback=fallback)
        self.optimizer = optax.multi_transform(partition_optimizers, partitioner)
        return self.optimizer

    def init(self, params: nnx.State) -> optax.OptState:
        """Initializes the optimizer state."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been built. Call `build(params)` first.")
        return self.optimizer.init(params)

    def update(self, grads, opt_state, params=None) -> Tuple[Any, Any]:
        """Updates the optimizer state and computes parameter updates."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been built. Call `build(params)` first.")
        return self.optimizer.update(grads, opt_state, params)

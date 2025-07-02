"""Tree-based optimizer state management."""

from typing import Any, Dict, List, Tuple, Union, Optional, Set
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
from dataclasses import dataclass, field
import itertools

from ueaj.utils import either


@dataclass(frozen=True)
class BatchMarker:
    """Marks a dimension as batched."""
    slice_spec: Any = slice(None)

    def __repr__(self):
        return f"batch[{self.slice_spec}]"

    def __hash__(self):
        # Make it hashable
        if isinstance(self.slice_spec, slice):
            return hash((self.slice_spec.start, self.slice_spec.stop, self.slice_spec.step))
        elif isinstance(self.slice_spec, tuple):
            return hash(self.slice_spec)
        else:
            return hash(self.slice_spec)


class BatchAccessor:
    """Helper to create batch markers with slice syntax."""
    def __getitem__(self, key):
        return BatchMarker(key)


@dataclass(frozen=True)
class OptNode:
    """Immutable node in the optimizer tree."""
    # todo batched tracking
    opt_fn: Optional[Any] = None
    set_id: Optional[int] = None  # Tracks which set() call created this
    children: Dict[Any, 'OptNode'] = field(default_factory=lambda: {})

def intersect(keys: set, key):
    if isinstance(key, (tuple, list, set)):
        return keys.intersection(key)
    elif key is ...:
        return keys
    elif isinstance(key, slice) and key == slice(None, None, None):
        return keys
    elif isinstance(key, slice):
        return keys.intersection(range(either(key.start, min(keys)), either(key.stop, max(keys)), key.step or 1))
    else:
        return keys.intersection([key])

def access(state, keys):
    if isinstance(state, jax.ShapeDtypeStruct):
        return [(keys, jax.ShapeDtypeStruct(state.shape[1:], state.dtype))]
    elif isinstance(state, (dict, nnx.State)):
        return [({key}, state[key]) for key in keys]
    elif isinstance(state, (list, tuple)):
        return [({key}, state[key]) for key in keys]
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")

class OptimizerManager:
    """Tree-based optimizer configuration."""

    def __init__(self, model):
        self.root = OptNode()
        self.batch = BatchAccessor()
        self.model = model
        self._next_set_id = 0
        self._patterns = []  # Store patterns for validation

    def __setitem__(self, key: Union[Any, Tuple[Any, ...]], opt_fn: Any):
        """Set optimizer for a pattern."""
        if not isinstance(key, tuple):
            key = (key,)

        # todo expand batch tuple

        # Store pattern for validation
        self._patterns.append((key, opt_fn, self._next_set_id))

        # Insert into tree
        state = nnx.state(self.model, nnx.Param)
        state = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)
        self.root = self._insert_pattern(self.root, state, key, opt_fn, self._next_set_id)
        self._next_set_id += 1

    def _insert_pattern(
        self, node: OptNode, state, pattern: Tuple,
        opt_fn: Any, set_id: int
    ) -> OptNode:
        """Insert pattern into tree, splitting nodes as needed."""
        if len(pattern) == 0:
            # End of pattern - set optimizer
            return OptNode(opt_fn=opt_fn, set_id=set_id)

        key = pattern[0]
        ellipsis_ = False
        if key is ...:
            key = pattern[1:]

            # todo ellipsis handling, iterate over children first
            ellipsis_ = True

        if isinstance(state, jax.ShapeDtypeStruct):
            keys = set(range(state.shape[0]))
        elif isinstance(state, (dict, nnx.State)):
            keys = set(state.keys())
        elif isinstance(state, (list, tuple, np.ndarray)):
            keys = set(range(len(state)))
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

        subtract = intersect(keys, key)
        if len(subtract) == 0:
            return node

        new_children = {}
        if node.children is None:
            remaining = keys - subtract
            new_children[remaining] = OptNode(opt_fn=node.opt_fn, set_id=node.set_id)
        else:
            for k, v in list(node.children.items()):
                new_k = k - subtract
                if len(new_k) > 0:
                    new_children[new_k] = v

        # if subtract is strings then just do them separately
        for key, substate in access(state, subtract):
            new_children[key] = self._insert_pattern(
                OptNode(opt_fn=node.opt_fn, set_id=node.set_id),
                substate,
                pattern[1:],
                opt_fn,
                set_id
            )

        return OptNode(
            opt_fn=None,
            set_id=None,
            children=new_children
        )
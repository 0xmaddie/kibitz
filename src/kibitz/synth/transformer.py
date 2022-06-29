from jax.nn import elu
from jax.nn import relu
from jax.nn import softmax
from jax.nn import normalize

from kibitz.synth import Tensor
from kibitz.synth import use_param
from kibitz.synth import use_state

def transformer(
    stream: Tensor,
    depth: int = 2,
) -> Tensor:
  for _ in range(0, depth):
    stream += _attention(stream)
    stream = _layer_norm(stream)
    stream += _dense(stream)
    stream = _layer_norm(stream)
  return softmax(stream)

# This is the "fast weight" layer with normalization from "Linear
# Transformers Are Secretly Fast Weight Programmers"
# https://arxiv.org/abs/2102.11174
def _attention(stream: Tensor) -> Tensor:
  to_query = use_param(grade=2)
  to_key = use_param(grade=2)
  to_value = use_param(grade=2)
  attn_state = use_state(grade=2)
  norm_state = use_state(grade=1)

  query = stream@to_query
  key = stream@to_key
  value = stream@to_value

  query = elu(query)+1
  key = elu(key)+1

  attn = attn_state.add_mut(key.T@value)
  norm = norm_state.add_mut(key)

  return (query@attn)/(query*norm)

def _dense(stream: Tensor) -> Tensor:
  linear = use_param(grade=2)
  bias = use_param(grade=1)
  return relu(stream@linear+bias)

def _layer_norm(stream: Tensor) -> Tensor:
  gain = use_param(grade=1)
  bias = use_param(grade=1)
  return gain*normalize(stream)+bias

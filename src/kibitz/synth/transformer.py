from jax.nn import elu
from jax.nn import relu
from jax.nn import softmax
from jax.nn import normalize

from kibitz.synth import Tensor
from kibitz.synth import use_param
from kibitz.synth import use_state

def transformer(stream: Tensor) -> Tensor:
  depth = 2
  for _ in range(0, depth):
    to_query = use_param(grade=2)
    to_key = use_param(grade=2)
    to_value = use_param(grade=2)

    residual_gain = use_param(grade=1)
    residual_bias = use_param(grade=1)
    residual_scale_state = use_state(grade=1)

    attention_state = use_state(grade=2)

    query = stream@to_query
    key = stream@to_key
    value = stream@to_value

    # Attention normalization.
    query = elu(query)+1
    key = elu(key)+1

    attention = attention_state.add_mut(key.T@value)
    residual_scale = residual_scale_state.add_mut(key)

    # Attention normalization.
    residual = query@attention
    residual = residual/(query*residual_scale)

    # Layer normalization.
    residual = normalize(residual)
    residual *= residual_gain
    residual += residual_bias

    # Activation.
    residual = relu(residual)

    # todo: should we normalize again here?

    print(f'stream={stream}\nquery={query}\nkey={key}\nvalue={value}\nkey.T@value={key.T@value}\nattention={attention}residual={residual}\nstream+residual={stream+residual}')

    stream += residual

  stream = softmax(stream)
  return stream

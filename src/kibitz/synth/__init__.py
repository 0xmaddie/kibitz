from dataclasses import dataclass

from typing import Sequence, Callable, Any

import jax
import jax.numpy as np

Tensor = np.ndarray
State = Sequence[Tensor]
Param = Sequence[Tensor]
# why doesnt this work?
# Config = tuple[State, Tensor]
Config = Any
Model = Callable

@dataclass(frozen=True)
class Response:
  prompt_latent: State
  continuation_latent: State
  continuation: Sequence[str]

def hot1(index: int, length: int) -> Tensor:
  value = np.zeros((1, length))
  value = value.at[0, index].set(1)
  return value

_magic_tokens = ['__begin__', '__end__']
class Codec:
  tokens: Sequence[str]
  size: int
  
  def __init__(self, tokens: Sequence[str]):
    tokens = _magic_tokens + tokens
    self.tokens = tokens
    self.size = len(tokens)
    self._index_from_token = {
      token: index
      for index, token in enumerate(tokens)
    }
    self._token_from_index = {
      index: token
      for index, token in enumerate(tokens)
    }

  def encode(self, token: str) -> Tensor:
    index = self._index_from_token[token]
    value = hot1(index, self.size)
    return value

  def decode(self, value: Tensor) -> str:
    # For now I'm going to keep this deterministic and pick the most
    # likely one.
    value = value.at[0, 0].set(0)
    value = jax.nn.softmax(value,axis=1)
    index = int(np.argmax(value))
    token = self._token_from_index[index]
    return token

_dynamic_scope = None

class Box:
  index: int
  def __init__(self, index: int):
    self.index = index

  def get(self) -> Tensor:
    global _dynamic_scope
    return _dynamic_scope.state[self.index]

  def set(self, value: Tensor):
    global _dynamic_scope
    _dynamic_scope._state_buf[self.index] = value

  def add_mut(self, residual: Tensor) -> Tensor:
    global _dynamic_scope
    old_value = _dynamic_scope.state[self.index]
    new_value = residual+old_value
    _dynamic_scope._state_buf[self.index] = new_value
    return new_value

def zero(grade: int=1) -> Tensor:
  return _dynamic_scope.zero(grade)

def use_param(grade: int=1) -> Tensor:
  return _dynamic_scope.use_param(grade)

def use_state(grade: int=1) -> Box:
  return _dynamic_scope.use_state(grade)

class RandomScope:
  codec: Codec
  param: Sequence[Tensor]
  state: Sequence[Tensor]
  _state_buf: Sequence[Tensor]
  _state_ptr: int = 0
  _param_ptr: int = 0
  _prng_key: jax.random.PRNGKey

  def __init__(
      self,
      codec: Codec,
      state: State,
      prng_seed: int = 0,
  ):
    self.codec = codec
    self.param = []
    self.state = state
    self._state_buf = state.copy()
    self._prng_key = jax.random.PRNGKey(prng_seed)

  def begin(self):
    global _dynamic_scope
    _dynamic_scope = self

  def end(self):
    global _dynamic_scope
    assert self == _dynamic_scope
    _dynamic_scope = None
    tmp = self.state
    self.state = self._state_buf
    self._state_buf = tmp
    self._state_ptr = 0
    self._param_ptr = 0

  def _next_prng_key(self):
    self._prng_key, subkey = jax.random.split(self._prng_key)
    return subkey

  def _shape_from_grade(self, grade: int):
    size = self.codec.size
    shape_list = [(1, 1), (1, size), (size, size)]
    shape = shape_list[grade]
    return shape

  def zero(
      self,
      grade: int = 1
  ) -> Tensor:
    assert grade == 1 or grade == 2
    shape = self._shape_from_grade(grade)
    value = np.zeros(shape)
    return value
  
  def use_random(
      self,
      grade: int = 1,
  ) -> Tensor:
    assert grade == 1 or grade == 2
    key = self._next_prng_key()
    size = self.codec.size
    shape = self._shape_from_grade(grade)
    value = jax.random.normal(key, shape)
    return value

  def use_param(
      self,
      grade: int = 1,
  ) -> Tensor:
    assert grade == 1 or grade == 2
    assert self._param_ptr <= len(self.param)
    if self._param_ptr == len(self.param):
      random = self.use_random(grade)
      self.param.append(random)
    value = self.param[self._param_ptr]
    expected_shape = self._shape_from_grade(grade)
    assert value.shape == expected_shape
    self._param_ptr += 1
    return value

  def use_state(
      self,
      grade: int = 1,
  ) -> Box:
    assert grade == 1 or grade == 2
    assert self._state_ptr <= len(self.state)
    if self._state_ptr == len(self.state):
      zero = self.zero(grade)
      self.state.append(zero)
      self._state_buf.append(zero)
    value = self.state[self._state_ptr]
    expected_shape = self._shape_from_grade(grade)
    assert value.shape == expected_shape
    box = Box(self._state_ptr)
    self._state_ptr += 1
    return box

# def analyze(model) -> ModelShape:
#   nil = np.zeros(1)
#   scope = AnalyzeScope()
#   scope.begin()
#   _ = model(nil)
#   scope.end()
#   return scope.shape

def evaluate(
    model: Model,
    param: Param,
    codec: Codec,
    config: Config,
    prng_seed:int=0,
):
  state, source = config
  scope = RandomScope(
    codec=codec,
    state=state,
    prng_seed=prng_seed,
  )
  scope.begin()
  target = model(source)
  scope.end()
  return [scope.state, target]

def prompt(
    model: Model,
    param: Param,
    codec: Codec,
    tokens: Sequence[str]=[],
    gas:int=1e2,
    prng_seed:int=0,
) -> Response:
  state = []
  tokens = ['__begin__'] + tokens
  for token in tokens:
    source = codec.encode(token)
    state, target = evaluate(
      model=model,
      param=param,
      codec=codec,
      config=[state, source],
      prng_seed=prng_seed,
    )
  prompt_latent = state
  continuation = []
  token = codec.decode(target)
  while token != '__end__' and gas > 0:
    continuation.append(token)
    print(f'gas={gas}\nparam={param}\nstate={state}\ntarget={target}\ntoken={token}\ncontinuation={continuation}')
    gas -= 1
    state, target = evaluate(
      model=model,
      param=param,
      codec=codec,
      config=[state, target],
      prng_seed=prng_seed,
    )
    token = codec.decode(target)
  if gas == 0:
    print('gas exhausted')
  continuation_latent = state
  response = Response(
    prompt_latent=prompt_latent,
    continuation_latent=continuation_latent,
    continuation=continuation,
  )
  return response

import kibitz.synth.transformer

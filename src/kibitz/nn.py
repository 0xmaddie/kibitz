import math
import numpy as np
from typing import Union, List, Callable

Num = np.ndarray
Exp = Callable[[Num], Num]

_dynamic_scope = None
_nil = np.array([[0.0]])

def use_param(grade:int=1) -> Num:
  return _dynamic_scope._use_param(grade)

def use_state(grade:int=1) -> Num:
  return _dynamic_scope._use_state(grade)

class Box:
  index: int
  value: Num

  def __init__(self, index: int, value: Num):
    self.index = index
    self.value = value

  def add(self, value: Num) -> Num:
    self.value += value
    _dynamic_scope._set_state(self.index, self.value)
    return self.value

class Model:
  dim: int
  exp: Exp
  param: Num
  state: Num

  _param_use_offset: int = 0
  _state_use_index: int = 0
  _state_set_index: int = 0
  _state_use_offset: int = 0

  def __init__(
      self,
      dim:int,
      exp:Exp,
      param:Num,
      state:Num,
  ):
    self.dim = dim
    self.exp = exp
    self.param = param
    self.state = state
    self._state_buf = []
    self._shape_map = [(1, 1), (1, dim), (dim, dim)]

  def _use_param(self, grade:int=1) -> Num:
    _, param_total_len = self.param.shape
    param_shape = self._shape_map[grade]
    param_len = param_shape[0]*param_shape[1]
    lhs = self._param_use_offset
    rhs = self._param_use_offset+param_len
    assert rhs <= param_total_len
    value = self.param[:, lhs:rhs]
    value = np.reshape(value, param_shape)
    self._param_use_offset += param_len
    return value

  def _use_state(self, grade:int=1) -> Box:
    _, state_total_len = self.state.shape
    state_shape = self._shape_map[grade]
    state_len = state_shape[0]*state_shape[1]
    lhs = self._state_use_offset
    rhs = self._state_use_offset+state_len
    value = self.state[:, lhs:rhs]
    value = np.reshape(value, state_shape)
    box = Box(self._state_use_index, value)
    self._state_use_index += 1
    self._state_use_offset += state_len
    return box

  def _set_state(self, index: int, value: Num):
    assert index >= 0
    assert index == self._state_set_index
    self._state_buf.append(value)
    self._state_set_index += 1

  def eval(self, src: Num) -> Num:
    global _dynamic_scope
    assert _dynamic_scope == None
    _dynamic_scope = self
    dst = self.exp(src)
    _dynamic_scope = None
    self._param_use_offset = 0
    self._state_use_offset = 0
    self._state_use_index = 0
    self._state_set_index = 0
    state_shape = self.state.shape
    self.state = np.concatenate(
      self._state_buf, axis=None)
    self.state = np.reshape(self.state, state_shape)
    self._state_buf = []
    return dst

class _Compile:
  dim: int
  param_len: int = 0
  state_len: int = 0

  def __init__(self, dim: int = 1):
    self.dim = dim

  def _use_param(self, grade: int = 1) -> Num:
    self.param_len += self.dim**grade
    return _nil

  def _use_state(self, grade: int = 1) -> Box:
    self.state_len += self.dim**grade
    return Box(0, _nil)

  def _set_state(self, index: int, value: Num):
    pass

def compile(exp: Exp, dim: int) -> Model:
  global _dynamic_scope
  assert _dynamic_scope == None
  _dynamic_scope = _Compile(dim=dim)
  exp(_nil)
  rng = np.random.default_rng()
  param_shape = (1, _dynamic_scope.param_len)
  state_shape = (1, _dynamic_scope.state_len)
  param = rng.standard_normal(param_shape)
  state = np.zeros(state_shape)
  model = Model(
    dim=dim, param=param, state=state, exp=exp)
  _dynamic_scope = None
  return model

def norm(x: Num) -> Num:
  mean = np.mean(x)
  var = np.cov(x,bias=True)
  stddev = np.sqrt(var)
  return (x-mean)/(stddev+1e-5)

def relu(x: Num) -> Num:
  return np.maximum(x, 0)

def attention(x: Num) -> Num:
  Q = use_param(grade=2)
  K = use_param(grade=2)
  V = use_param(grade=2)
  A = use_state(grade=2)
  z = use_state(grade=1)
  # todo: the linear transformer uses `elu` here,
  # not `relu`. does this really matter?
  q, k, v = relu(x@Q)+1, relu(x@K)+1, x@V
  A.add(k.T@v)
  z.add(k)
  return (q@A.value)/(q*z.value)

def dense(x: Num) -> Num:
  W = use_param(grade=2)
  b = use_param(grade=1)
  return relu(x@W+b)

def layer_norm(x: Num) -> Num:
  g = use_param(grade=1)
  b = use_param(grade=1)
  return g*norm(x)+b

def transformer(
    stream: Num,
    depth: int=2,
) -> Num:
  for _ in range(0, depth):
    stream += attention(stream)
    stream = layer_norm(stream)
    stream += dense(stream)
    stream = layer_norm(stream)
  return stream

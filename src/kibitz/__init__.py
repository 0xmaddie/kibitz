from typing import Sequence
import jax.numpy as np

class Alphabet:
  def __init__(self, tokens: Sequence[str]):
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

  def encode(self, token: str) -> np.ndarray:
    index = self._index_from_token[token]
    value = hot1(index, self.size)
    return value

  def decode(self, value: np.ndarray) -> str:
    # For now I'm going to keep this deterministic and pick the most
    # likely one.
    index = int(np.argmax(value))
    token = self._token_from_index[index]
    return token

def hot1(index: int, length: int) -> np.ndarray:
  value = np.zeros(length)
  value = value.at[index].set(1)
  return value

import functools
import sys
from .model import CNNBase
from .model import CNNBaseSmall
from .model import CNNBaseGfootball
from .model import MLPBase

MODEL_MAP = {
    'CNNBase': CNNBase,
    'CNNBaseSmall': CNNBaseSmall,
    'CNNBaseGfootball': CNNBaseGfootball,
    'MLPBase': MLPBase,
}


def get_base(name):
  assert name in MODEL_MAP
  return MODEL_MAP[name]

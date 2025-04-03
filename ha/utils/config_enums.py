from enum import Enum

class Mode(str, Enum):
    TRAIN = "Train"
    EVAL = "Eval"
    PRETRAIN = "Pretrain"
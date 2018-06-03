import enum

class DistanceMetrics(enum.Enum):
    L1 = 1
    L2 = 2
    Cosine = 3

class LossType(enum.Enum):
    SVM = 1
    BinaryCrossEntropy = 2
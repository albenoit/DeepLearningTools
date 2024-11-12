from tensorflow.keras.layers import Layer
from statistics import mean
import numpy as np


class Average_calculator:
    def __init__(self):
        pass

    def mean(self, models):
        avg_layers = []
        dlayer = {}
        for model in models:
            for i, layer in enumerate(model):
                if not i in dlayer:
                    dlayer[i] = []
                dlayer[i].append(layer)

        for i in range(len(dlayer)):
            avg_layers.append(np.mean(dlayer[i], axis=0))
        return avg_layers


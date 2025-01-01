import pandas as pd
import numpy as np


class MyKNNClf:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def __str__(self) -> str:
        return f'MyKNNClf class: k={self.k}'

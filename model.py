import pandas as pd
import numpy as np
from scipy.spatial import distance


class MyKNNClf:
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.train_size = None

    def __str__(self) -> str:
        return f'MyKNNClf class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_size = X.shape[0], X.shape[1]
        self.X = X.copy()  # Сохраняем копию, чтобы не менять исходный DataFrame
        self.y = y.copy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = []

        for i in range(len(X)):
            distances = []
            point1 = X.iloc[i]

            for j in range(len(self.X)):
                point2 = self.X.iloc[j]

                euclid = distance.euclidean(point1, point2)
                distances.append((euclid, self.y[j]))

            k_distances = sorted(distances, key=lambda x: x[0])[:self.k]
            k_classes = [el[1] for el in k_distances]

            if k_classes.count(1) >= k_classes.count(0):
                pred += [1]
            else:
                pred += [0]

        return np.array(pred)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:

        pred = []

        for i in range(len(X)):
            distances = []
            point1 = X.iloc[i]
            for j in range(len(self.X)):
                point2 = self.X.iloc[j]

                euclid = distance.euclidean(point1, point2)
                distances.append((euclid, self.y[j]))

            k_distances = sorted(distances, key=lambda x: x[0])[:self.k]
            k_classes = [el[1] for el in k_distances]

            if k_classes.count(1):
                pred += [k_classes.count(1) / len(k_classes)]
            else:
                pred += [0]

        return np.array(pred)

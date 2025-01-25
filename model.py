import pandas as pd
import numpy as np
from scipy.spatial import distance


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform') -> None:
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight

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
                d = []

                if self.metric == 'euclidean':
                    d = distance.euclidean(point1, point2)
                elif self.metric == 'chebyshev':
                    d = distance.chebyshev(point1, point2)
                elif self.metric == 'manhattan':
                    d = distance.cityblock(point1, point2)
                elif self.metric == 'cosine':
                    d = distance.cosine(point1, point2)
                distances.append((d, self.y[j]))

            k_distances = sorted(distances, key=lambda x: x[0])[:self.k]
            k_classes = [el[1] for el in k_distances]

            if self.weight == 'uniform':
                if k_classes.count(1) >= k_classes.count(0):
                    pred += [1]
                else:
                    pred += [0]

            elif self.weight == 'rank':
                ind0, ind1 = 0, 0

                for i in range(len(k_classes)):
                    if k_classes[i] == 0:
                        ind0 += 1 / (i + 1)
                    else:
                        ind1 += 1 / (i + 1)

                q0 = ind0 / (ind0 + ind1)
                q1 = ind1 / (ind0 + ind1)

                if q1 >= q0:
                    pred += [1]
                else:
                    pred += [0]

            elif self.weight == 'distance':
                d0 = 0
                d1 = 0
                for i in range(len(k_distances)):
                    if k_classes[i] == 0:
                        d0 += 1 / (k_distances[i][0])
                    else:
                        d1 += 1 / (k_distances[i][0])

                q0 = d0 / (d0 + d1)
                q1 = d1 / (d0 + d1)

                if q1 >= q0:
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
                d = []

                if self.metric == 'euclidean':
                    d = distance.euclidean(point1, point2)
                elif self.metric == 'chebyshev':
                    d = distance.chebyshev(point1, point2)
                elif self.metric == 'manhattan':
                    d = distance.cityblock(point1, point2)
                elif self.metric == 'cosine':
                    d = distance.cosine(point1, point2)
                distances.append((d, self.y[j]))

            k_distances = sorted(distances, key=lambda x: x[0])[:self.k]
            k_classes = [el[1] for el in k_distances]

            if self.weight == 'uniform':
                if k_classes.count(1):
                    pred += [k_classes.count(1) / len(k_classes)]
                else:
                    pred += [0]

            elif self.weight == 'rank':
                ind0, ind1 = 0, 0

                for i in range(len(k_classes)):
                    if k_classes[i] == 0:
                        ind0 += 1 / (i + 1)
                    else:
                        ind1 += 1 / (i + 1)

                q1 = ind1 / (ind0 + ind1)
                pred += [q1]

            elif self.weight == 'distance':
                d0 = 0
                d1 = 0
                for i in range(len(k_distances)):
                    if k_classes[i] == 0:
                        d0 += 1 / (k_distances[i][0])
                    else:
                        d1 += 1 / (k_distances[i][0])

                q1 = d1 / (d0 + d1)
                pred += [q1]

        return np.array(pred)

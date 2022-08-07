import numpy as np
class my_scaler:
    def __init__(self):
        self.trans = np.log
        # natural log

    def fit(self, X):
        assert X.ndim == 2
        res = np.empty(X.shape, dtype = np.float32)
        res = self.trans(X)
        return res
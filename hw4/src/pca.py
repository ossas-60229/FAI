import numpy as np


"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None
    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        # shift by the mean
        self.mean = np.mean(X, axis=0)
        tmpX = X - self.mean
        # calculate the eigen vector
        cov = np.cov(tmpX,rowvar=False)
        eig_val,eig_vecs = np.linalg.eigh(cov)
        #eig_val = np.abs(eig_val)
        #eig_vecs = np.real(eig_vecs)
        # sort the eigen vector
        idx_arr = np.argsort(eig_val)
        eig_vecs = eig_vecs[:,idx_arr]
        #print(top_n_eigvec.shape)
        self.components = eig_vecs
        # reduce the dimensions

        return
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        tmpX = X - self.mean
        return np.dot(tmpX, self.components.T)
        raise NotImplementedError

    def reconstruct(self, X) -> np.ndarray:
        #TODO: 2%
        newX = X - self.mean
        newX = np.dot(newX, self.components.T)
        newx = np.dot(newX, self.components) + self.mean
        return newx
        raise NotImplementedError

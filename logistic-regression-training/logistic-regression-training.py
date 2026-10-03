import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    w=np.zeros(X.shape[1])
    b=0.0
    loss=0.0
    for i in range(steps) :
        z=X@w+b
        yp=_sigmoid(z)
        w=w-lr*(X.T@(yp-y))/X.shape[0]
        b=b-lr*(np.mean(yp-y))
    return (w,b)
    pass
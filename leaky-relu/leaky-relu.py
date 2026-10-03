import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """
    x=np.asarray(x,dtype=float)
    y= np.asarray(x,dtype=float)
    for i in range(len(x)):
        if x[i]<0 :
            y[i]=x[i]*alpha
        else : y[i]=x[i]
    return y
    pass
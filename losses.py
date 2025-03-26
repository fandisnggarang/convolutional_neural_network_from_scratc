import numpy as np 

def mse(y_true, y_pred): 
    return np.square(np.subtract(y_true, y_pred)).mean()

def mse_prime(y_true, y_pred): 
    return np.multiply(2, np.divide(np.subtract(y_pred, y_true), y_true.size))

def binary_cross_entropy(y_true, y_pred): 
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred): 
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

# modified from https://github.com/TheIndependentCode/Neural-Network/tree/master
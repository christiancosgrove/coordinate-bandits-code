from sklearn.datasets import load_svmlight_file
import numpy as np
def load_dataset(filename):
    print('Loading dataset ', filename)
    X, y = load_svmlight_file(filename)
    X = X.toarray()
    print('X shape ', X.shape)
    print('Y shape ', y.shape)
    print('X mean ', np.mean(X))#, ' std', np.std(X))
    print('y min', ((y+1)/2).min(), ' max ', ((y+1)/2).max())

    return X, (y+1)/2

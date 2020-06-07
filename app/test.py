import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

if __name__ == "__main__":
    # test model to test web app

    # lin reg example from sklearn
    X = np.array([[1,1], [1,2], [2,2], [3,3]])
    # y = np.dot(X, np.array([1,2])) + 3

    # model = LinearRegression()
    # model.fit(X,y)

    # with open('models/test_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    with open('models/test_model.pkl', 'rb') as f:
        test = pickle.load(f)

    y_hat = test.predict(X)
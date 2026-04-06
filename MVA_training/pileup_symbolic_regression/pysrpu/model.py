import numpy as np
from pysr import PySRRegressor, silence_julia_warning

silence_julia_warning()

def train_pysr(X, y, niterations, population_size, maxsize, seed=1):
    # elementwise_loss="loss(yhat, y) = log(1 + exp(- (2y - 1) * yhat))",

    model = PySRRegressor(
        niterations=niterations,
        population_size=population_size,
        maxsize=maxsize,
        model_selection="best", # options: best, accuracy
        verbosity=1,
        random_state=seed,
        deterministic=True,
        parallelism="multiprocessing",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log1p", "tanh", "log"],
        complexity_of_operators={"/": 3, "log1p": 2, "log": 2, "tanh": 2, "sqrt": 2},
        # elementwise_loss=elementwise_loss,
        elementwise_loss="logistic",
    )

    model.fit(X, y)
    return model


def safe_predict(model, X):
    s = model.predict(X)
    if np.isscalar(s):
        return np.full(len(X), s)
    return s

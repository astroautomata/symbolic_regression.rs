import pickle

import numpy as np

from symbolic_regression_rs import equation_search

X = np.random.default_rng(0).uniform(-2, 2, (64, 2))
y = X[:, 0]**2 + X[:, 1]
options = dict(
    binary_operators=["+", "*"],
    populations=4,
    seed=0,
    deterministic=True,
    parallelism="serial",
    save_to_file=False,
)
state = equation_search(X, y, niterations=40, **options)
for complexity, loss, equation in state.pareto_fronts[0]:
    print(complexity, loss, equation)

restored = pickle.loads(pickle.dumps(state))
continued = equation_search(X, y, saved_state=restored, niterations=10, **options)
print(continued.pareto_fronts[0])

import cProfile
import numpy as np

import google_benchmark as benchmark
from google_benchmark import Counter


def create_matrix(d, nb, nq, seed=1234):
    """create a matrix

    Args:
        d (int): dimension of matrix
        nb (int): database size
        nq (int): size of queries
        seed (int, optional): random seed. Defaults to 1234.

    Returns:
        xb: matrix of database
        xq: matrix of queries
    """
    np.random.seed(seed)
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    return xb, xq

@benchmark.register(name="create_matrix")
def _create_matrix(state):
    while state:
        create_matrix(64, 100000, 10000)


if __name__ == "__main__":
    benchmark.main()

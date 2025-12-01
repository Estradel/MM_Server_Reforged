import math

import pytest
import numpy as np
from numba import njit, prange

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
@pytest.fixture(scope="module")
def grands_tableaux():
    # 1 million de points
    print("\n[Init] Génération...")
    NUMBER = 1_000_000
    # NUMBER = 10_000
    # NUMBER = 1_000
    x = np.random.rand(NUMBER)
    y = np.random.rand(NUMBER)

    # Pré-allocation du tableau de résultats
    result = np.empty(NUMBER, dtype=np.float64)
    return x, y, result

# ---------------------------------------------------------
# 2. Les Fonctions
# ---------------------------------------------------------
def op_python_pur(x_list, y_list, result):
    n = len(x_list)
    for i in range(n):
        result[i] = math.sqrt(x_list[i]**2 + y_list[i]**2) / (1 + math.exp(x_list[i] * y_list[i]))
    return result

@njit
def op_python_pur_numba(x_list, y_list, result):
    n = len(x_list)
    for i in prange(n):
        result[i] = math.sqrt(x_list[i]**2 + y_list[i]**2) / (1 + math.exp(x_list[i] * y_list[i]))
    return result

@njit(parallel=True)
def op_python_pur_numba_parallel(x_list, y_list, result):
    n = len(x_list)
    for i in prange(n):
        result[i] = math.sqrt(x_list[i]**2 + y_list[i]**2) / (1 + math.exp(x_list[i] * y_list[i]))
    return result

def op_numpy_vectorize(x, y, result):
    result[:] = np.sqrt(x**2 + y**2) / (1 + np.exp(x * y))
    return result

@njit
def op_numpy_vectorize_numba(x, y, result):
    result[:] = np.sqrt(x**2 + y**2) / (1 + np.exp(x * y))
    return result

@njit(parallel=True)
def op_numpy_vectorize_numba_parallel(x, y, result):
    result[:] = np.sqrt(x**2 + y**2) / (1 + np.exp(x * y))
    return result

def op_numpy_loop(x, y, result):
    for i in range(len(x)):
        result[i] = np.sqrt(x[i]**2 + y[i]**2) / (1 + np.exp(x[i] * y[i]))
    return result

@njit
def op_numpy_loop_numba(x, y, result):
    for i in range(len(x)):
        result[i] = np.sqrt(x[i]**2 + y[i]**2) / (1 + np.exp(x[i] * y[i]))
    return result

@njit(parallel=True)
def op_numpy_loop_numba_parallel(x, y, result):
    for i in prange(len(x)):
        result[i] = np.sqrt(x[i]**2 + y[i]**2) / (1 + np.exp(x[i] * y[i]))
    return result


@njit(fastmath=True)
def vectorized_math(a, b):
    # Numba essentially calls the C-functions for these numpy ops one by one.
    # 1. Allocate Temp1, Compute cos(a) -> Temp1
    # 2. Allocate Temp2, Compute sin(b) -> Temp2
    # 3. Allocate Result, Compute Temp1 * Temp2 -> Result
    return np.cos(a) * np.sin(b)

@njit(fastmath=True)
def looped_math(a, b, result):
    n = len(a)
    for i in prange(n):
        # Numba fuses this.
        # Loads a[i] and b[i] into registers, computes, stores result[i].
        # ZERO memory allocation during the loop.
        result[i] = math.cos(a[i]) * math.sin(b[i])

# ---------------------------------------------------------
# 3. Les Benchmarks
# ---------------------------------------------------------

def test_python_pur(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    x_list = x_np.tolist()
    y_list = y_np.tolist()
    benchmark(op_python_pur, x_list, y_list, result)

def test_python_pur_with_numba(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    x_list = x_np.tolist()
    y_list = y_np.tolist()
    op_python_pur_numba(x_list, y_list, result)
    benchmark(op_python_pur_numba, x_list, y_list, result)

def test_python_pur_with_numba_parallel(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    x_list = x_np.tolist()
    y_list = y_np.tolist()
    op_python_pur_numba_parallel(x_list, y_list, result)
    benchmark(op_python_pur_numba_parallel, x_list, y_list, result)

def test_numpy_vectorize(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    benchmark(op_numpy_vectorize, x_np, y_np, result)

def test_numpy_vectorize_with_numba(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    op_numpy_vectorize_numba(x_np, y_np, result)
    benchmark(op_numpy_vectorize_numba, x_np, y_np, result)

def test_numpy_vectorize_with_numba_parallel(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    op_numpy_vectorize_numba_parallel(x_np, y_np, result)
    benchmark(op_numpy_vectorize_numba_parallel, x_np, y_np, result)

def test_numpy_loop(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    benchmark(op_numpy_loop, x_np, y_np, result)

def test_numpy_loop_with_numba(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    op_numpy_loop_numba(x_np, y_np, result)
    benchmark(op_numpy_loop_numba, x_np, y_np, result)

def test_numpy_loop_with_numba_parallel(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    op_numpy_loop_numba_parallel(x_np, y_np, result)
    benchmark(op_numpy_loop_numba_parallel, x_np, y_np, result)

def test_vectorized_math(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    vectorized_math(x_np, y_np)
    benchmark(vectorized_math, x_np, y_np)

def test_looped_math(benchmark, grands_tableaux):
    x_np, y_np, result = grands_tableaux
    looped_math(x_np, y_np, result)
    benchmark(looped_math, x_np, y_np, result)
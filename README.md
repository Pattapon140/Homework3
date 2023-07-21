import numpy as np
import time

def getMatrixMinor(arr, i, j):
    c = arr[:]
    c = np.delete(c, (i), axis=0)
    return np.array([np.delete(row, (j), axis=0) for row in c])
def detCofactor(A):
    if A.ndim != 2:
        raise Exception("Input must be a matrix.")
    if A.shape[0] != A.shape[1]:
        raise Exception("Matrix must be square.")
    if A.shape[0] == 2:
        det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
        return det
    elif A.shape[0] == 1:
        return A[0, 0]
    else:
        det = 0
        i = 0
        for j in range(A.shape[1]):
            aij = A[i, j]
            Mij = getMatrixMinor(A, i, j)
            if (i + j) % 2 == 0:
                Cij = detCofactor(Mij)
            else:
                Cij = -detCofactor(Mij)
            det += aij * Cij
        return det

for n in range(5, 11):
    A = np.random.randn(n, n)

    # Using cofactor method
    tic_cofactor = time.perf_counter_ns()
    det_cofactor = detCofactor(A)
    toc_cofactor = time.perf_counter_ns()

    # Using np.linalg.det method
    tic_np = time.perf_counter_ns()
    det_np = np.linalg.det(A)
    toc_np = time.perf_counter_ns()

    det_cofactor_rounded = round(det_cofactor, 3)
    det_np_rounded = round(det_np, 3)

    time_difference_percentage = ((toc_np - tic_np) - (toc_cofactor - tic_cofactor)) / (toc_cofactor - tic_cofactor) * 100

    print(f"\nMatrix Size: {n}x{n}")
    print("Cofactor Method:")
    print("Determinant:", det_cofactor_rounded)
    print("Time: ", (toc_cofactor - tic_cofactor) / 1e6, "milliseconds")

    print("\nnp.linalg.det Method:")
    print("Determinant:", det_np_rounded)
    print("Time: ", (toc_np - tic_np) / 1e6, "milliseconds")

    print("Time Difference Percentage: ", time_difference_percentage, "%")

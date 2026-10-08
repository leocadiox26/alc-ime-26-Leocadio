import numpy as np

def resolve_lu(A, b):
    """Resolve Ax=b por decomposição LU sem pivotamento.

    Retorna (L, U, x). Não utiliza rotinas prontas de álgebra linear.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A deve ser uma matriz quadrada.')
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError('b deve ser um vetor com n elementos.')

    L = np.eye(n)
    U = np.array(A, dtype=float)

    for k in range(n):
        if abs(U[k, k]) < 1e-12:
            raise Exception('Pivô nulo ou muito próximo de zero. Utilize uma função alternativa com pivotamento.')
        for i in range(k + 1, n):
            multiplicador = U[i, k] / U[k, k]
            L[i, k] = multiplicador
            for j in range(k, n):
                U[i, j] = U[i, j] - multiplicador * U[k, j]
            U[i, k] = 0.0

    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = b[i] - soma

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]
    return L, U, x


if __name__ == '__main__':
    A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
    b = [4, 10, 24]
    L, U, x = resolve_lu(A, b)
    print('L =\n', L)
    print('U =\n', U)
    print('x =', x)

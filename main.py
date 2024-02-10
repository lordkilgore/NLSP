import numpy as np

class NLSP:
    def __init__(self):
        pass
    
    @staticmethod
    def backsub(U, y):
        # Initialization
        n = U.shape[0] - 1

        x = np.array([float(i) for i in range(n + 1)])
        x[n] = y[n] / U[n][n]
        
        for i in range(n):
            entry_num = n - 1 - i

            # Inner Product between row of U and x, excluding the last row
            row_sum = 0
            for k in range(n - entry_num):
                row_sum += U[entry_num][entry_num + k + 1] * x[entry_num + k + 1]
            
            # Backwards Substitution
            x[entry_num] = 1 / (U[entry_num][entry_num]) * (y[entry_num] - row_sum)
        
        return x

    @staticmethod
    def forwardsub(L, y):
        # Initialization
        n = L.shape[0] - 1
        x = np.array([float(1) for i in range(n + 1)])
        x[0] = y[0] / L[0][0]

        for i in range(n):
            entry_num = i + 1

            # Inner Product between row of L and x, excluding the first row
            row_sum = 0
            for k in range(entry_num):
                row_sum += L[entry_num][k] * x[k]
            
            # Forward Substitution
            x[entry_num] = 1 / (L[entry_num][entry_num]) * (y[entry_num] - row_sum)
        
        return x

    @staticmethod
    def gaussian(A, y = None):
        n = A.shape[0] - 1

        for i in range(n):
            # Initialization
            e = np.zeros(n+1)
            e[i] = 1
            
            tau = np.zeros(n+1)
            for k in range(n+1):
                if k <= i:
                    pass
                else:
                    tau[k] = A[k][i]
            tau = 1 / A[i][i] * tau

            # Gauss Transform
            L = (np.identity(n + 1) - np.outer(tau, e))

            # Gaussian Elimination
            A = np.matmul(L, A)
            if str(type(y)) != "<class 'NoneType'>":
                y = np.matmul(L, y)
        
        if str(type(y)) != "<class 'NoneType'>":
            return A, y
        else:
            return A
    
    @staticmethod
    def solve(A, b):
        if np.linalg.det(A) != 0:
            U, y = NLSP.gaussian(A, b)
            x = NLSP.backsub(U, y)
            return x
        else:
            return f"{A} is singular."

A = np.array([[1, 1, 1, 1, 1],
              [2, 3, -4, 5, 0],
              [4, 9, 16, 25, 0],
              [8, 27, -64, 125, 0],
              [16, 81, 256, 625, 0]])
b = np.array([2, 1, 4, 2, 1])

print(NLSP().solve(A, b))

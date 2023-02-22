import numpy as np
import cv2

def get_matrix(alpha, beta, gamma, n):
    """Return the matrix for the internal energy minimization.
    # Arguments
        alpha: The alpha parameter.
        beta: The beta parameter.
        gamma: The gamma parameter.
        num_points: The number of points in the curve.


    # Returns
        The matrix for the internal energy minimization. (i.e. A + gamma * I)
    """
    id = np.identity((n))
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = (2*alpha) + (6*beta)
            elif j-i == 1 or i-j == 1:
                A[i][j] = (-1*alpha) + (-4*beta)
            elif i-j == n-1 or j-i == n-1:
                A[i][j] = (-1 * alpha) + (-4 * beta)
            elif j-i == 2 or i-j == 2:
                A[i][j] = beta
            elif j - i == n - 2 or i - j == n - 2:
                A[i][j] = beta
            else:
                A[i][j] = 0

    print(A)
    print(A.shape)
    M_ = A + (gamma*id)
    M = np.linalg.inv(M_)
    cv2.imshow('Internal Energy', M)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return M

    # raise NotImplementedError

import numpy as np
A = np.array[[1,2],
             [3,4]]
A_ = A.transpose()
B = [[1,2],
     [1,2]]
C = (A.dot(B)).dot(A_)
print(C)
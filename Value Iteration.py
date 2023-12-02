import numpy as np

A = 3
S = 5
gamma = 0.5
iterations = 100

T = np.asarray(
    [[[1/2, 1/2,   0,   0,   0], [1/2, 1/2,   0,   0,   0], [2/3, 1/3,   0,   0,   0]],
     [[1/4, 1/2, 1/4,   0,   0], [1/3, 2/3,   0,   0,   0], [  0, 2/3, 1/3,   0,   0]], 
     [[  0, 1/4, 1/2, 1/4,   0], [  0, 1/3, 2/3,   0,   0], [  0,   0, 2/3, 1/3,   0]], 
     [[  0,   0, 1/4, 1/2, 1/4], [  0,   0, 1/3, 2/3,   0], [  0,   0,   0, 2/3, 1/3]],
     [[  0,   0,   0, 1/2, 1/2], [  0,   0,   0, 1/3, 2/3], [  0,   0,   0, 1/2, 1/2]] 
    ])

R = np.array([0, 0, 0, 0, 1])
V = np.zeros((iterations, S))

for k in range(1, iterations):
    for s in range(S):
        V[k, s] = max([sum([T[s, a, s_prime]*(R[s] + gamma*V[k-1, s_prime]) for s_prime in range(S)]) for a in range(A)])
        
print(V[-1, :])

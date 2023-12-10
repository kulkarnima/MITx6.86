import numpy as np

A = 3
S = 5
gamma = 0.5
iterations = 100

"""
Array of state transition probabilities (|S|x|A|x|S'|) Each entry represents 
probability of going from state s to state s' when action a is taken. The sum 
of entries in any row should add up to 1. The rules for state transitions are 
as follows:
    1. At any grid location, the agent can decide to stay at location or move 
    to neighboring one. If agent decides to stay, the action succeeds with 
    probability 1/2 and:
        1.1 

"""

T = np.asarray(
    [[[1/2,1/2,  0,  0,  0], [1/2,1/2,  0,  0,  0], [2/3,1/3,  0,  0,  0]],
     [[1/4,1/2,1/4,  0,  0], [1/3,2/3,  0,  0,  0], [  0,2/3,1/3,  0,  0]], 
     [[  0,1/4,1/2,1/4,  0], [  0,1/3,2/3,  0,  0], [  0,  0,2/3,1/3,  0]], 
     [[  0,  0,1/4,1/2,1/4], [  0,  0,1/3,2/3,  0], [  0,  0,  0,2/3,1/3]],
     [[  0,  0,  0,1/2,1/2], [  0,  0,  0,1/3,2/3], [  0,  0,  0,1/2,1/2]] 
    ])

R = np.array([0, 0, 0, 0, 1])
V = np.zeros(5)

for k in range(0, iterations-1):
    for s in range(S):
        V[s] = max([sum([T[s, a, s_prime]*(R[s] + gamma*V[s_prime]) 
                         for s_prime in range(S)
                        ]) for a in range(A) 
                   ])
        
print(V)

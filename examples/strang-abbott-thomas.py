"""Figure 4 in Strang, Abbott, and Thomas (2022)"""

import numpy as np


# Gradient
G = np.array([[-1, 1, 0, 0], 
              [0, -1, 0, 1], 
              [1, 0, 0, -1],
              [0, -1, 1, 0], 
              [0, 0, -1, 1]], dtype=np.float64)

# Divergence
D = G.T

# Curl
C = np.array([[1, 1, 1, 0, 0], 
              [0, -1, 0, 1, 1]], dtype=np.float64)

print(C @ G)
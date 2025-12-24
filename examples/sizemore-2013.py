"""Small examples for Hodge ranking from Sizemore (2013) pp. 82-86."""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# import scipy.sparse as sp

from HodgeRanking import HodgeRanking


# Example 1: Mostly Consistent Data (pp. 82-83)
teams1 = np.array(['Mercer', 'Wake Forest', 'Furman', 'Davidson', 'New Mexico', 'George Mason'], dtype=np.str_)

# NOTE: For example 1, the code gives the correct ranking when we replace the '2's in the weight matrix with '1's. 
W1_with_1 = np.array([[0,1,1,0,1,1], 
              [1,0,1,0,0,0], 
              [1,1,0,1,0,0], 
              [0,0,1,0,1,0], 
              [1,0,0,1,0,1], 
              [1,0,0,0,1,0]], dtype=np.float64)

Y1 = np.array([[ 0,  -3,   27,     0, -18, -3],
              [  3,   0,   24,     0,   0,  0],
              [-27, -24,    0, -27.5,   0,  0],
              [  0,   0, 27.5,     0,  -5,  0],
              [ 18,   0,    0,     5,   0,  1],
              [  3,   0,    0,     0,  -1,  0]], dtype=np.float64)

hodge_ranking_1 = HodgeRanking(teams1, W1_with_1, Y1)

# rank = HodgeRanking.rank(teams, W, Y)
print(f"Ranking: \n{hodge_ranking_1.rank(output_dir='out/sizemore/ex1/', offset=70.24)}\n")
decomposition = hodge_ranking_1.decompose()

# ------------------------------------
# TODO: visualize the components using networkx and matplotlib
# print(f"Gradient component: \n{decomposition[0]}\n")
# print(f"Inconsistent component: \n{decomposition[1]}\n")
# print(f"Cyclic ratio: \n{decomposition[2]}\n")

# Only get the upper triangular part to avoid duplicate edges
# gradient_G = nx.from_numpy_array(np.triu(decomposition[0]), create_using=nx.DiGraph)
# plt.savefig('examples/ex1/gradient_component.png')

# inconsistent_G = nx.from_numpy_array(np.triu(decomposition[1]), create_using=nx.DiGraph)
# plt.savefig('examples/ex1/inconsistent_component.png')
# ------------------------------------

# Example 2: Inconsistent Data (pp. 84-86)
teams2 = np.array(['Miami', 'NC State', 'Wake Forest', 'Maryland', 'UNC', 'Duke'], dtype=np.str_)
Y2 = np.array([[ 0 ,-1,-15, 0 , 9 , 27], 
               [ 1 , 0,-2 , 0 , 0 , 0], 
               [ 15, 2, 0 ,-26, 0 , 0], 
               [ 0 , 0, 26, 0,-10 , 0], 
               [-9 , 0, 0 , 10, 0 ,-5], 
               [-27, 0, 0 , 0 , 5 , 0]], dtype=np.float64)
W2 = (Y2 != 0).astype(np.float64)

ranking = HodgeRanking(teams2, W2, Y2).rank(output_dir='out/sizemore/ex2/', offset=69.87)
print(f"Ranking: \n{ranking}\n")
# decomposition2 = HodgeRanking(teams2, W2, Y2).decompose()





# def main() -> None: 
#     os.makedirs('examples', exist_ok=True)
#     dirpath = os.path.join('examples', 'ex1')
#     os.makedirs(dirpath, exist_ok=True)

#     np.savetxt(os.path.join(dirpath, 'W.txt'), W, fmt='%.6f')
#     np.savetxt(os.path.join(dirpath, 'Y.txt'), Y, fmt='%.6f')

#     print("W and Y matrix saved to W.txt and Y.txt respectively.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Example script to save W and Y matrices.")
#     parser.add_argument('teams', help="path to team names text file")
#     parser.add_argument('W', help="path to weight / adjacency matrix (n x n) numpy text file")
#     parser.add_argument('Y', help="path to pairwise comparison matrix (n x n) numpy text file")
#     main()
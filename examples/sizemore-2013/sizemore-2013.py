"""Small examples for Hodge ranking from Sizemore (2013) pp. 82-86."""

import argparse
import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from HodgeRanking import y2adj, HodgeRanking


def draw_graph(adj_matrix: NDArray[np.float64], teams: NDArray[np.str_], filepath='out/sizemore-2013/graph.png') -> nx.DiGraph:
    """Draw a directed graph from the adjacency matrix using networkx.

    Parameters
    ----------
    adj_matrix : NDArray[np.float64]
        Adjacency matrix (n x n)
    teams : NDArray[np.str_]
        List of team names sorted by their IDs.
    filename : str, optional
        Filename to save the graph image, by default 'graph.png'

    Returns
    -------
    nx.DiGraph
        Graph object created from the adjacency matrix.
    """
    plt.clf()

    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    pos_for_6 = { 0: (0, 0), 1: (1, 1), 2: (3, 1),
                  3: (4, 0), 4: (3, -1), 5: (1, -1) }
    pos = pos_for_6 if len(teams) == 6 else nx.spring_layout(nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph))

    nx.draw_networkx(G, pos=pos, 
                     labels={i:teams[i] for i in range(len(teams))},
                     node_color='#ccccff', node_size=5000)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels={(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)})
    
    if filepath != '' and filepath is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)

    plt.close()
    return G


def main(args: argparse.Namespace) -> None: 
    if args.example not in [1, 2]:
        raise ValueError("Example # must be 1 or 2.")
    
    print(f"Running example {args.example}...")

    # Load data
    data_dir = os.path.join('examples', 'sizemore-2013', 'data')
    teamfile = os.path.join(data_dir, f'teams{args.example}.csv')
    Wfile = os.path.join(data_dir, f'w{args.example}.csv')
    Yfile = os.path.join(data_dir, f'y{args.example}.csv')

    teams: NDArray[np.str_] = np.loadtxt(teamfile, dtype=np.str_, delimiter=',')
    W: NDArray[np.float64] = np.loadtxt(Wfile, dtype=np.float64, delimiter=',')
    Y: NDArray[np.float64] = np.loadtxt(Yfile, dtype=np.float64, delimiter=',')

    # NOTE: Skipping error checking for now...

    out_dir = os.path.join('out', 'sizemore-2013', f'ex{args.example}')

    if args.graph_y:
        adj_y = y2adj(Y)
        filepath = os.path.join(out_dir, 'y_graph.png')
        draw_graph(adj_y, teams, filepath=filepath)
        print(f"Directed graph from of pairwise comparison matrix saved to {filepath}.")

    # Ranking
    offset = 70.24 if args.example == 1 else 69.87  # precomputed offsets by me to align results with Sizemore
    hodge = HodgeRanking(teams, W, Y)
    ranking = hodge.rank(output_dir=out_dir, offset=offset)
    print(f"Ranking: \n{ranking}\n")

    ranking.to_csv(os.path.join(out_dir, 'ranking.csv'), index=False)
    print(f"Ranking saved to {os.path.join(out_dir, 'ranking.csv')}\n")

    # Decomposition
    grad_comp, inconst_comp, cyclicity_ratio = hodge.decompose(output_dir=out_dir)
    print(f"Cyclicity Ratio: {cyclicity_ratio}\n")

    grad_filepath = os.path.join(out_dir, 'grad_graph.png')
    inconst_filepath = os.path.join(out_dir, 'inconsistent_graph.png')

    draw_graph(grad_comp, teams, filepath=grad_filepath)
    draw_graph(inconst_comp, teams, filepath=inconst_filepath)

    print(f"Directed graph of gradient component saved to {grad_filepath}.")
    print(f"Directed graph of inconsistent component saved to {inconst_filepath}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and view an example from Sizemore (2013).")
    parser.add_argument('example', type=int, help="Which example to run (1 or 2)")
    parser.add_argument('--graph-y', action='store_true', help="Whether to draw the directed graph from Y matrix")

    args = parser.parse_args()
    main(args)
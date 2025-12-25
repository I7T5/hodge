from collections.abc import MutableSequence
import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd
np.set_printoptions(precision=3, suppress=True)


def y2adj(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Helper function for converting pairwise comparison matrix Y to adjacency matrix.

    Parameters
    ----------
    Y : NDArray[np.float64]
        Pairwise comparison matrix (n x n)

    Returns
    -------
    NDArray[np.float64]
        Adjacency matrix (n x n)
    """
    n = Y.shape[0]
    adj_matrix = np.zeros((n, n), dtype=np.float64)

    # Only keep negative weights in the pairwise comparison matrix
    for i in range(n):
        for j in range(n):
            if Y[i, j] < 0:
                adj_matrix[i, j] = Y[i, j]
    
    # Absorb negative sign to make weights positive
    return np.abs(adj_matrix)


class HodgeRanking:
    """HodgeRank class for computing rankings based on pairwise comparisons (Sizemore, 2013, p. 81). Naming of variables follows the notation in Sizemore.
    
    Attributes
    ----------
    teams : MutableSequence|NDArray[np.str_]
        MutableSequence or numpy array of team names in ascending order of their IDs.
    W : NDArray[np.float64]
        Weight / adjacency matrix (n x n). W needs be a float matrix for proper division operations. 
    Y : NDArray[np.float64]
        Pairwise comparison matrix (n x n)
    """
    
    def __init__(self, teams: MutableSequence|NDArray[np.str_], W: NDArray[np.float64], Y: NDArray[np.float64]) -> None:
        self.teams = teams
        self.W = W
        self.Y = Y
    
    @property
    def teams(self) -> MutableSequence|NDArray[np.str_]:
        """Get the MutableSequence or numpy array of team names.
        
        Returns
        -------
        MutableSequence or NDArray[np.str_]
            team names in ascending order of their IDs.
        """
        return self._teams
    
    @teams.setter
    def teams(self, value: MutableSequence|NDArray[np.str_]) -> None:
        """Set the MutableSequence or NDArray of team names.
        
        Parameters
        ----------
        value : MutableSequence or NDArray[np.str_]
            List of team names in ascending order of their IDs.
        """
        if not isinstance(value, MutableSequence) and not isinstance(value, np.ndarray):
            raise ValueError("teams must be a MutableSequence or NDArray")
        self._teams = value
    
    @property
    def W(self) -> NDArray[np.float64]:
        """Get the weight / adjacency matrix.
        
        Returns
        -------
        NDArray[np.float64]
            Weight / adjacency matrix (n x n)
        """
        return self._W
    
    @W.setter
    def W(self, value: NDArray[np.float64]) -> None:
        """Set the weight / adjacency matrix.
        
        Parameters
        ----------
        value : NDArray[np.float64]
            Weight / adjacency matrix (n x n)
        """
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise ValueError("W must be a NDArray of float64")
        if value.ndim != 2:
            raise ValueError("W must be a 2D array")
        self._W = value
    
    @property
    def Y(self) -> NDArray[np.float64]:
        """Get the pairwise comparison matrix.
        
        Returns
        -------
        NDArray[np.float64]
            Pairwise comparison matrix (n x n)
        """
        return self._Y
    
    @Y.setter
    def Y(self, value: NDArray[np.float64]) -> None:
        """Set the pairwise comparison matrix.
        
        Parameters
        ----------
        value : NDArray[np.float64]
            Pairwise comparison matrix (n x n)
        """
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise ValueError("Y must be a NDArray of float64")
        if value.ndim != 2:
            raise ValueError("Y must be a 2D array")
        self._Y = value


    def rank(self, *, output_dir=None, offset=0.0) -> pd.DataFrame:
        """Compute the HodgeRank using the weight matrix W and pairwise comparison matrix Y. 
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save the ranking CSV file. Default is None (no saving).
        offset : float, optional
            Offset to adjust the baseline score. Default is 0.0.

        Returns
        -------
        pd.DataFrame
            Ranking scores for each team
        """

        # Laplacian matrix (Sizemore, 2013, p. 56 last definition, also p. 81)
        L = np.diag(np.sum(self.W, axis=0)) - self.W

        # Compute rankings
        div_Y = np.sum(self.Y, axis=1)
        r = np.linalg.pinv(L) @ div_Y.T

        # Adjust baseline score if needed (Sizemore, 2013, p. 75)
        r += offset

        # Format the rankings into a DataFrame and save to CSV
        teams = np.array(self.teams).reshape(-1, 1)
        ranking = r.reshape(-1, 1)
        ranking_df = pd.DataFrame(np.concatenate([teams, ranking], axis=1), columns=['team_name', 'score'])
        ranking_df.sort_values(by='score', axis=0, ascending=False, inplace=True)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            ranking_df.to_csv(os.path.join(output_dir, 'ranking.csv'), index=False)

        return ranking_df
    

    def decompose(self, output_dir=None) -> tuple[NDArray[np.float64], NDArray[np.float64], float]: 
        """Decompose the ranking into consistent and inconsistent components (Sizemore 2013, p. 75) as adjacency matrices

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64], float]
            List containing, in order, consistent component, inconsistent component, and the cyclicity ratio. 
        """

        def grad(s: NDArray[np.float64], W: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute the gradient flow of a potential function (vector) s as an adjacency matrix for graphing purposes. Might not be Sizemore's notation.
            
            Parameters
            ----------
            s : NDArray[np.float64]
                Potential function (n x 1 vector) $V -> \\mathbb{R}$ on nodes, where V stands for vertex. 
            W : NDArray[np.float64]
                Weight / adjacency matrix (n x n)
            
            Returns
            -------
            NDArray[np.float64]
                Gradient flow of the ranking vector (n x n)
            """
            n = s.shape[0]
            grad_matrix = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(i, n):  # remove duplicate edges
                    if W[i, j] != 0:   # only compute gradient where there is an edge
                        val = s[j] - s[i]
                        if val < 0:    # if flow from i to j is negative, reverse direction
                            grad_matrix[j, i] = abs(val)
                        else:
                            grad_matrix[i, j] = s[j] - s[i]
            return grad_matrix
        
        s = self.rank().sort_index()['score'].to_numpy(dtype=np.float64)
        gradient = grad(s, self.W)  # gradient component

        # inconsistent component
        residual: NDArray[np.float64] = y2adj(self.Y) - gradient  
        # to solve the problem of subtractin (adding) arrows in different directions (R_ij and R_ji)
        n = residual.shape[0]
        for i in range(n): 
            for j in range(i, n): 
                # if both directions have non-zero weights
                if residual[i, j] != 0 and residual[j, i] != 0:
                    if residual[i, j] > residual[j, i]:
                        residual[i, j] -= residual[j, i]  # -= == + (-1) *
                        residual[j, i] = 0.0
                    else:
                        residual[j, i] -= residual[i, j]
                        residual[i, j] = 0.0

        cyclicity_ratio = float(np.linalg.norm(residual, ord='fro') /  np.linalg.norm(gradient, ord='fro'))  # cyclicity ratio using Frobenius norm

        # TODO: Save the components to output_dir if needed

        return gradient, residual, cyclicity_ratio
    

# # ----------- Binary Comparisons -----------
# s = np.diag(B @ W.T)   # Diagonal of B * W^T
# D = np.linalg.pinv(L)  # Pseudoinverse of Laplacian
# r = -D @ s             # Compute rankings
# np.savetxt('score_binary.csv', r, fmt='%7.4f', delimiter=',')

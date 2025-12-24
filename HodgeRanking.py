from collections.abc import MutableSequence
import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd
np.set_printoptions(precision=3, suppress=True)


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
        MutableSequence or NDArray of team names in ascending order of their IDs.
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
        NDArray[np.float64]: Weight / adjacency matrix (n x n)
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
        NDArray[np.float64]: Pairwise comparison matrix (n x n)
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
        output_dir (str): Directory to save the ranking CSV file.
        offset (int): Offset to adjust the baseline score.

        Returns
        -------
        pd.DataFrame: Ranking scores for each team
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
            ranking_df.to_csv(os.path.join(output_dir, 'ranking.csv'), index=False)

        return ranking_df
    

    def decompose(self, output_dir=None) -> tuple[NDArray[np.float64], NDArray[np.float64], float]: 
        """Decompose the ranking into consistent and inconsistent components. (Sizemore 2013, p. 75)

        Returns
        -------
        list[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: List containing, in order, consistent component, inconsistent component, and the cyclic ratio. 
        """

        def grad(s: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute the gradient flow of a potential function (vector) s.
            
            Parameters
            ----------
            s : NDArray[np.float64]
                Potential function (n x 1 vector) $V -> \mathbb{R}$ on nodes, where V stands for vertex. 
            
            Returns
            -------
            NDArray[np.float64]: Gradient flow of the ranking vector (n x n)
            """
            n = s.shape[0]
            grad_matrix = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    grad_matrix[i, j] = s[j] - s[i]
            return grad_matrix
        
        consistent_comp: NDArray[np.float64] = grad(self.rank()['score'].to_numpy(dtype=np.float64))  # gradient component
        residual: NDArray[np.float64] = self.Y - consistent_comp  # inconsistent component
        cyclic_ratio: float = float(np.linalg.norm(residual, ord='fro') /  np.linalg.norm(consistent_comp, ord='fro'))  # cyclic ratio

        # TODO: Save the components to output_dir if needed

        return consistent_comp, residual, cyclic_ratio
    

# # ----------- Binary Comparisons -----------
# s = np.diag(B @ W.T)   # Diagonal of B * W^T
# D = np.linalg.pinv(L)  # Pseudoinverse of Laplacian
# r = -D @ s             # Compute rankings
# np.savetxt('score_binary.csv', r, fmt='%7.4f', delimiter=',')

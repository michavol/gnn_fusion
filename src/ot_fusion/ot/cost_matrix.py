from .ground_cost import GroundCost
import numpy as np
import tqdm 
import jax.numpy as jnp

class CostMatrix:
    """ 
    Ground cost class for computing the ground cost between two points on a manifold.
    """
    def __init__(self, cfg):
        self.args = cfg
        self.ground_cost_fn = GroundCost(cfg).get_cost_fn()

    def get_cost_matrix(self, X, Y):
        """
        X: list of lists of dgl graphs - OT source support
        Y: list of lists of dgl graphs - OT target support
        return cost matrix between X and Y as jnp array
        """
        # Compute entries of cost matrix using pairwise ground costs
        cost_matrix = np.zeros((len(X),len(Y)))
        for i in tqdm.tqdm(range(len(X))):
            for j in range(len(Y)):
                cost_matrix[i][j] = self.ground_cost_fn(X[i], Y[j])

        # Normalize cost matrix
        cost_matrix = self._normalize(cost_matrix)

        # Sanity check
        self._sanity_check(cost_matrix)

        return jnp.array(cost_matrix)
    
    def _normalize(self, cost_matrix):
        """
        Normalize the ground cost matrix by the specified method.
        """
        normalization = self.args.cost_matrix_normalization
        if normalization == "log":
            cost_matrix = torch.log1p(cost_matrix)
        elif normalization == "max":
            print("Normalizing by max of ground cost and which is ", cost_matrix.max())
            cost_matrix = cost_matrix / cost_matrix.max()
        elif normalization == "median":
            print("Normalizing by median of ground cost and which is ", cost_matrix.median())
            cost_matrix = cost_matrix / cost_matrix.median()
        elif normalization == "mean":
            print("Normalizing by mean of ground cost and which is ", cost_matrix.mean())
            cost_matrix = cost_matrix / cost_matrix.mean()
        elif normalization == "none":
            return cost_matrix
        else:
            raise NotImplementedError

        return cost_matrix
    
    def _isnan(self, x):
        return x != x

    def _sanity_check(self, cost_matrix):
        """ 
        Check that the cost matrix is non-negative and does not contain NaNs.
        """
        assert not (cost_matrix < 0).any()
        assert not (self._isnan(cost_matrix)).any()
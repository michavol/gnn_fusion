from .graph_cost import GraphCost

import jax.numpy as jnp
import numpy as np
import tqdm


class GroundCostGcn:
    """ 
    Ground cost class for computing the ground cost between source and target support for GNN OT fusion problem.
    In the case of activation-based fusion, the source and target support are the activations of the GNNs in the form of a list of lists of dgl graphs.
    """

    def __init__(self, cfg):
        # # Get ground cost config parameters
        # self.args = cfg.ground_costs
        self.args = cfg.ground_cost_gcn
        self.graph_cost = GraphCost(cfg)

    def get_cost_fn(self):
        """
        Return the ground cost function.
        """
        # Return ground cost
        return self._ground_cost

    def get_cost_matrix(self, X, Y):
        """
        X: list of lists of dgl graphs - OT source support
        Y: list of lists of dgl graphs - OT target support
        return cost matrix between X and Y as jnp array
        """
        # Compute entries of cost matrix using pairwise ground costs
        cost_matrix = np.zeros((len(X), len(Y)))

        # Disable progress bar if verbose is False
        disable = True
        if self.args.progress_bar:
            print("=============================================")
            print("Computing Cost Matrix...")
            disable = False

        for i in tqdm.tqdm(range(len(X)), disable=disable):
            for j in range(len(Y)):
                cost_matrix[i][j] = self._ground_cost(X[i], Y[j])

        # Normalize cost matrix
        normalization_method = self.args.cost_matrix_normalization
        cost_matrix = self._normalize(cost_matrix, normalization_method)

        # Sanity check
        self._sanity_check(cost_matrix)

        return jnp.array(cost_matrix)

    def _ground_cost(self, x, y):
        """
        Cost function for OT problem with sets of dgl graphs as support.
        x, y are sets of dgl graphs - i.e. the GNN activations
        """
        # Check dimensions
        num_graphs = len(x)
        assert num_graphs == len(y)

        graph_cost_fn = self.graph_cost.get_graph_cost_fn()

        cost = 0
        for i in range(num_graphs):
            cost += graph_cost_fn(x[i], y[i])
        return cost

    def _normalize(self, cost_matrix, normalization_method="none"):
        """
        Normalize the ground cost matrix by the specified method.
        """
        if normalization_method == "log":
            cost_matrix = torch.log1p(cost_matrix)
        elif normalization_method == "max":
            print("Normalizing by max of ground cost and which is ", cost_matrix.max())
            cost_matrix = cost_matrix / cost_matrix.max()
        elif normalization_method == "median":
            print("Normalizing by median of ground cost and which is ", cost_matrix.median())
            cost_matrix = cost_matrix / cost_matrix.median()
        elif normalization_method == "mean":
            print("Normalizing by mean of ground cost and which is ", cost_matrix.mean())
            cost_matrix = cost_matrix / cost_matrix.mean()
        elif normalization_method == "none":
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

from .graph_cost import GraphCost
from ott.math import utils as mu
import tqdm
import jax.numpy as jnp

class GroundCost:
    """ 
    Ground cost class for computing the ground cost between source and target support for GNN OT fusion problem.
    In the case of activation-based fusion, the source and target support are the activations of the GNNs in the form of a list of lists of dgl graphs.
    """
    def __init__(self, cfg):
        # Get ground cost config parameters
        self.args = cfg.ground_costs
        self.cfg = cfg

    def _ground_cost_feature_based(self, x, y):
        """
        Cost function for l2-cost based OT problem
        x, y are sets of dgl graphs - i.e. the GNN activations
        """

        # Check dimensions
        num_graphs = len(x)
        assert num_graphs == len(y)

        # Compute and sum pairwise costs
        cost = 0
        for i in range(num_graphs):
            # Compute the lp-norm between the features of the two graphs
            cost += mu.norm(jnp.array(x[i].ndata["Feature"]) - jnp.array(y[i].ndata["Feature"]), ord=self.args.lp_norm_ord)
        return cost
    
    def _ground_cost_graph_based(self, x, y):
        """
        Cost function for QE-metric (quadratic energy) based OT problem
        x, y are sets of dgl graphs - i.e. the GNN activations
        """
        # Check dimensions
        num_graphs = len(x)
        assert num_graphs == len(y)

        graph_costs = GraphCost(self.cfg)
        graph_cost_fn = graph_costs.get_graph_cost_fn()

        cost = 0
        for i in range(num_graphs):
            cost += graph_cost_fn(x[i], y[i])
        return cost
    
    def get_cost_fn(self):
        """
        Return the ground cost function.
        """
        # Return ground cost
        if self.args.ground_cost_type == "feature_based":
            return self._ground_cost_feature_based
        
        elif self.args.ground_cost_type == "graph_based":
            return self._ground_cost_graph_based
        
        else:
            raise NotImplementedError
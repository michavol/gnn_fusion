import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import tqdm
import torch
import jax
import jax.numpy as jnp

from ott.math import utils as mu
from ott.geometry import geometry, pointcloud
from ott.geometry.graph import Graph
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
import tqdm

class CostMatrix:
    """ 
    Ground cost class for computing the ground cost between two points on a manifold.
    """
    def __init__(self, args):
        self.args = args
        self.ground_costs = GroundCosts(args)
        self.ground_cost_fn = self.ground_costs._get_cost_fn()
        self.cost_matrix_normalize = self.args.cost_matrix_normalize
        self.cost_matrix_normalize_method = self.args.cost_matrix_normalize_method

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
        if self.cost_matrix_normalize:
            cost_matrix = self._normalize(cost_matrix)

        # Sanity check
        self._sanity_check(cost_matrix)

        return jnp.array(cost_matrix)
    
    def _normalize(self, cost_matrix):
        """
        Normalize the ground cost matrix by the specified method.
        """

        if self.ground_metric_normalize == "log":
            cost_matrix = torch.log1p(cost_matrix)
        elif self.ground_metric_normalize == "max":
            print("Normalizing by max of ground cost and which is ", cost_matrix.max())
            cost_matrix = cost_matrix / cost_matrix.max()
        elif self.ground_metric_normalize == "median":
            print("Normalizing by median of ground cost and which is ", cost_matrix.median())
            cost_matrix = cost_matrix / cost_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print("Normalizing by mean of ground cost and which is ", cost_matrix.mean())
            cost_matrix = cost_matrix / cost_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return cost_matrix
        else:
            raise NotImplementedError

        return cost_matrix
    
    def _isnan(x):
        return x != x

    def _sanity_check(self, cost_matrix):
        """ 
        Check that the cost matrix is non-negative and does not contain NaNs.
        """
        assert not (cost_matrix < 0).any()
        assert not (isnan(cost_matrix).any())


class GroundCosts:
    """ 
    Ground cost class for computing the ground cost between source and target support for GNN OT fusion problem.
    In the case of activation-based fusion, the source and target support are the activations of the GNNs in the form of a list of lists of dgl graphs.
    """
    def __init__(self, args):
        self.args = args
        self.ground_metric_type = self.args.ground_metric_type

        # Get graph cost if graph-based
        if self.ground_metric_type == "graph_based":
            self.graph_costs = GraphCosts(args)
            self.graph_cost_fn = self.graph_costs.get_graph_cost_fn()

        elif self.ground_metric_type == "feature_based":
            self.lp_norm_ord = self.args.ground_metric_lp_norm_ord
            assert self.lp_norm_ord != 0

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
            cost += mu.norm(jnp.array(x[i].ndata["Feature"]) - jnp.array(y[i].ndata["Feature"]), ord=self.lp_norm_ord)
        return cost
    
    def _ground_cost_graph_based(self, x, y):
        """
        Cost function for QE-metric (quadratic energy) based OT problem
        x, y are sets of dgl graphs - i.e. the GNN activations
        """
        # Check dimensions
        num_graphs = len(x)
        assert num_graphs == len(y)

        # Compute pairwise costs
        cost = 0
        for i in range(num_graphs):
            cost += self.graph_cost_fn(x[i], y[i])
        return cost
    
    def _get_cost_fn(self):
        """
        Return the ground cost function.
        """
        # Return ground cost
        if self.ground_metric_type == "feature_based":
            return self._ground_cost_feature_based
        
        elif self.ground_metric_type == "graph_based":
            return self._ground_cost_graph_based
        
        else:
            raise NotImplementedError
        

class GraphCosts:
    """ 
    Class for computing the cost between two entities
    """
    def __init__(self, args):
        self.args = args
        self.graph_metric_type = self.args.graph_cost_type

        if self.graph_metric_type == "quadratic_energy":
            self.alpha = self.args.QE_graph_cost_alpha
        
        elif self.graph_metric_type == "fused_GW":
            self.loss = self.args.fused_GW_graph_cost_loss
            self.output_cost = self.args.fused_GW_graph_cost_output_cost
            self.fused_penalty = self.args.fused_GW_graph_cost_fused_penalty
            self.epsilon = self.args.fused_GW_graph_cost_epsilon
            self.tau_a = self.args.fused_GW_graph_cost_tau_a
            self.tau_b = self.args.fused_GW_graph_cost_tau_b
            self.max_iterations = self.args.fused_GW_graph_cost_max_iterations
            self.directed = self.args.fused_GW_graph_cost_directed
            self.normalize = self.args.fused_GW_graph_cost_normalize
        
        else:
            raise NotImplementedError
            
    def _graph_cost_quadratic_energy(self, graph_x, graph_y, alpha):
        """ 
        Compute the quadratic energy ground cost between two dgl graphs.
        """
        # Get features
        x_features = graph_x.ndata["Feature"]
        y_features = graph_y.ndata["Feature"]

        # Get number of edges (here graph structure of x and y are assumed to be the same)
        num_edges = graph_x.num_edges()
        assert num_edges == graph_y.num_edges()
        num_nodes = graph_x.num_nodes()
        assert num_nodes == graph_y.num_nodes()

        # Compute energy functional
        edge_energy = 0
        node_energy = 0

        # Compute edge energy
        for i in range(num_edges):
            # Get indeces of connected nodes
            out_node = graph_x.edges()[0][i]
            in_node = graph_x.edges()[1][i]

            # Get features of connected nodes of graphs x and y
            x_out_feature = x_features[out_node]
            y_in_feature = y_features[in_node]

            # Compute cost
            edge_energy += (x_out_feature - y_in_feature)**2 

        # Compute node energy
        for i in range(num_nodes):
            # Get feature of node
            x_feature = x_features[i]
            y_feature = y_features[i]

            # Compute cost
            node_energy += (x_feature - y_feature)**2

        # Compute weighted total energy
        total_energy = alpha*edge_energy + (1-alpha)*node_energy

        return total_energy
    
    def _graph_cost_fused_GW(self, graph_x, graph_y, loss='sqeucl', output_cost='primal', fused_penalty=1.0, epsilon=100, tau_a=1.0, tau_b=1.0, max_iterations=20, directed=False, normalize=True):
        """
        Compute GW cost between two dgl graphs.
        """
        # Extract features
        x_features = graph_x.ndata["Feature"]
        y_features = graph_y.ndata["Feature"]

        # Extract adjacency matrices
        adj_mat_x = graph_x.adjacency_matrix().to_dense()
        adj_mat_y = graph_y.adjacency_matrix().to_dense()

        # Create geometries
        geom_xy = pointcloud.PointCloud(jnp.array(x_features), jnp.array(y_features), cost_fn=None)
        geom_xx = Graph.from_graph(jnp.array(adj_mat_x), directed=directed, normalize=normalize)
        geom_yy = Graph.from_graph(jnp.array(adj_mat_y), directed=directed, normalize=normalize)

        # Create quadratic problem
        prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, geom_xy, 
                                                loss=loss,
                                                fused_penalty=fused_penalty,
                                                tau_a=tau_a, tau_b=tau_b,
                                                ranks=-1)
        
        # Instantiate a jitt'ed Gromov-Wasserstein solver
        solver = jax.jit(
            gromov_wasserstein.GromovWasserstein(
                epsilon=epsilon, max_iterations=max_iterations, store_inner_errors=True
            )
        )   

        # Solve the problem
        ot = solver(prob)

        # Extract costs
        if output_cost == 'primal':
            primal_cost = ot.primal_cost
            return primal_cost
        elif output_cost == 'reg_gw':
            reg_gw_cost = ot.reg_gw_cost
            return reg_gw_cost
    
    def get_graph_cost_fn(self):
        """
        Compute the graph cost between two graphs.
        """
        # Return graph cost
        if self.graph_metric_type == "quadratic_energy":
            graph_cost_fn = lambda x, y: self._graph_cost_quadratic_energy(x, y, self.alpha)
            return graph_cost_fn
        
        elif self.graph_metric_type == "fused_GW":
            graph_cost_fn = lambda x, y: self._graph_cost_fused_GW(x, y, 
                                                                  loss=self.loss, 
                                                                  output_cost=self.output_cost, 
                                                                  fused_penalty=self.fused_penalty, 
                                                                  epsilon=self.epsilon, 
                                                                  tau_a=self.tau_a, tau_b=self.tau_b, 
                                                                  max_iterations=self.max_iterations, 
                                                                  directed=self.directed, 
                                                                  normalize=self.normalize)
            return graph_cost_fn
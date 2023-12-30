# Imports
from ott.math import utils as mu
from ott.geometry import pointcloud
from ott.geometry.graph import Graph
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
import numpy as np

import jax
import jax.numpy as jnp

import tqdm


class GraphCost:
    """ 
    Class for computing the cost between two graphs.
    """

    def __init__(self, cfg):
        # Get graph cost config parameters
        self.args = cfg.graph_cost

        # Instantiate solver for fused_gw version for better performance
        if self.args.graph_cost_type == "fused_gw":
            self.solver = jax.jit(
                gromov_wasserstein.GromovWasserstein(
                    epsilon=self.args.epsilon,
                    max_iterations=self.args.max_iterations
                )
            )

    def _graph_cost_quadratic_energy(self, graph_x, graph_y, alpha):
        """ 
        Compute the quadratic energy ground cost between two dgl graphs.
        """
        # Get features
        x_features = jnp.array(graph_x.ndata["Feature"])
        y_features = jnp.array(graph_y.ndata["Feature"])

        # Get number of edges (here graph structure of x and y are assumed to be the same)
        num_edges = graph_x.num_edges()
        assert num_edges == graph_y.num_edges()
        num_nodes = graph_x.num_nodes()
        assert num_nodes == graph_y.num_nodes()

        # Compute energy functional
        # Compute edge cost
        out_nodes = jnp.array(graph_x.edges()[0])
        in_nodes = jnp.array(graph_x.edges()[1])
        x_out_features = x_features[out_nodes]
        y_in_features = y_features[in_nodes]

        edge_energy = mu.norm(x_out_features - y_in_features, ord=2) ** 2

        # Compute node cost
        node_energy = mu.norm(x_features - y_features, ord=2) ** 2
        # Compute weighted total energy
        alpha = self.args.alpha
        total_energy = alpha * edge_energy + (1 - alpha) * node_energy

        return total_energy

    def _graph_cost_fused_gw(self, graph_x, graph_y, loss='sqeucl', output_cost='primal', fused_penalty=1.0,
                             epsilon=100, tau_a=1.0, tau_b=1.0, max_iterations=20, directed=False, normalize=True):
        """
        Compute GW cost between two dgl graphs.
        """
        # Extract features
        x_features = graph_x.ndata["Feature"]
        y_features = graph_y.ndata["Feature"]

        print('features')
        print(x_features)
        print(y_features)

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

        # Solve the problem
        ot = self.solver(prob)

        # Extract costs
        if output_cost == 'primal':
            primal_cost = ot.primal_cost
            return primal_cost

        elif output_cost == 'reg_gw':
            reg_gw_cost = ot.reg_gw_cost
            return reg_gw_cost

        else:
            raise NotImplementedError

    def _graph_cost_feature_lp(self, graph_x, graph_y, lp_norm_ord=2):
        """
        Compute the L2 distance between the features of two graphs.
        """
        # Get features
        x_features = jnp.array(graph_x.ndata["Feature"])
        y_features = jnp.array(graph_y.ndata["Feature"])

        # Compute L2 distance
        lp_dist = mu.norm(x_features - y_features, ord=lp_norm_ord)

        return lp_dist

    def get_graph_cost_fn(self):
        """
        Compute the graph cost between two graphs.
        """
        # Return graph cost
        if self.args.graph_cost_type == "quadratic_energy":
            graph_cost_fn = lambda x, y: self._graph_cost_quadratic_energy(x, y, self.args.alpha)
            return graph_cost_fn

        elif self.args.graph_cost_type == "fused_gw":
            graph_cost_fn = lambda x, y: self._graph_cost_fused_gw(x, y,
                                                                   loss=self.args.loss,
                                                                   output_cost=self.args.output_cost,
                                                                   fused_penalty=self.args.fused_penalty,
                                                                   epsilon=self.args.epsilon,
                                                                   tau_a=self.args.tau_a, tau_b=self.args.tau_b,
                                                                   max_iterations=self.args.max_iterations,
                                                                   directed=self.args.directed,
                                                                   normalize=self.args.normalize)
            return graph_cost_fn

        elif self.args.graph_cost_type == "feature_lp":
            graph_cost_fn = lambda x, y: self._graph_cost_feature_lp(x, y, self.args.lp_norm_ord)
            return graph_cost_fn

        else:
            raise NotImplementedError

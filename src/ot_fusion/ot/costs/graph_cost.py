# Imports
from ott.math import utils as mu
from ott.geometry import pointcloud
from ott.geometry.graph import Graph
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
import numpy as np

import jax
import jax.numpy as jnp
from utils.data_operations import torch2jnp
import tqdm
import networkx as nx


class GraphCost:
    """ 
    Class for computing the cost between two graphs.
    """

    def __init__(self, cfg):
        # Get graph cost config parameters
        self.args = cfg.graph_cost

        # Instantiate solver for fused_gw version for better performance
        if self.args.graph_cost_type == "fused_gw":
            self.relative_epsilon = self.args.relative_epsilon
            self.epsilon = self.args.epsilon
            self.solver = jax.jit(
                gromov_wasserstein.GromovWasserstein(
                    warm_start=self.args.warm_start,
                    max_iterations=self.args.max_iterations,
                    store_inner_errors=True,
                    epsilon=self.args.epsilon,
                    relative_epsilon=self.args.relative_epsilon
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
                             epsilon=None, tau_a=1.0, tau_b=1.0, max_iterations=200, directed=False, normalize=True):
        """
        Compute GW cost between two dgl graphs.
        """
        # Extract features
        x_features = graph_x.ndata["Feature"]
        y_features = graph_y.ndata["Feature"]

        # Reshape features
        x_features = x_features.reshape(-1, 1)
        y_features = y_features.reshape(-1, 1)

        # Extract adjacency matrices
        # Use networkx to compute adjacency matrix, since dgl sparse matrix does not work with gpu setup at the moment
        graph_x_netx = graph_x.cpu().to_networkx()
        graph_y_netx = graph_y.cpu().to_networkx()

        adj_mat_x = nx.adjacency_matrix(graph_x_netx).todense()
        adj_mat_y = nx.adjacency_matrix(graph_y_netx).todense()

        # adj_mat_x = graph_x.adjacency_matrix().to_dense()
        # adj_mat_y = graph_y.adjacency_matrix().to_dense()

        # Create geometries
        geom_xy = pointcloud.PointCloud(torch2jnp(x_features),
                                        torch2jnp(y_features))

        geom_xx = Graph.from_graph(jnp.array(adj_mat_x),
                                   directed=directed,
                                   normalize=normalize)

        geom_yy = Graph.from_graph(jnp.array(adj_mat_y),
                                   directed=directed,
                                   normalize=normalize)

        # Create quadratic problem
        prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, geom_xy,
                                                  loss=loss,
                                                  fused_penalty=fused_penalty,
                                                  tau_a=tau_a, tau_b=tau_b,
                                                  scale_cost=True,
                                                  ranks=-1)

        # Solve the problem
        ot = self.solver(prob)

        if self.args.verbose:
            has_converged = ot.converged
            if ot.errors is None:
                n_iters = -1
            else:
                n_iters = jnp.sum(ot.errors[:, 0] != -1)
            print(f"{n_iters} outer iterations were needed.")
            print(f"The last Sinkhorn iteration has converged: {has_converged}")
            print(f"The outer loop of Gromov Wasserstein has converged: {ot.converged}")
            print(f"The final regularized GW cost is: {ot.reg_gw_cost:.3f}")

        # Check ot matrices
        # plt.imshow(ot.matrix)
        # plt.show()
        # exit()

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

        x_features = torch2jnp(graph_x.ndata["Feature"])
        y_features = torch2jnp(graph_y.ndata["Feature"])

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

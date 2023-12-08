import sys

# from .cost_matrix import CostMatrix
from .costs.ground_cost_gcn import GroundCostGcn
from .costs.ground_cost_mlp import GroundCostMlp

from ott import utils
from ott.geometry import geometry
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.problems.linear import linear_problem

import jax
import jax.numpy as jnp

import tqdm

sys.path.append('src')
from utils.layer_operations import LayerType

class OptimalTransport:
    def __init__(self, cfg):
        self.cfg = cfg
        self.args = cfg.optimal_transport

    def get_current_transport_map(self, X, Y, a, b, layer_type=LayerType.gcn):
        """
        Solve optimal transport problem for activation support for GNN Fusion
        """
        if layer_type == LayerType.gcn:
            # Compute cost matrix
            cost_matrix = GroundCostGcn(self.cfg).get_cost_matrix(X, Y)

        elif layer_type in [LayerType.mlp, LayerType.embedding]:
            # Compute cost matrix
            cost_matrix = GroundCostMlp(self.cfg).get_cost_matrix(X, Y)

        elif layer_type == LayerType.bn:
            # Compute cost matrix
            cost_matrix = None

        else:
            raise NotImplementedError

        # Define Geometry
        geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=self.args.epsilon) 

        # Define Problem
        ot_prob = linear_problem.LinearProblem(geom, tau_a=self.args.tau_a, tau_b=self.args.tau_b)

        # Solve Problem
        # Disable progress bar if verbose is False
        disable = True
        if self.args.progress_bar:
            print("=============================================")  
            print("Solving OT problem...")
            disable = False

        with tqdm.tqdm(disable=disable) as pbar:
            progress_fn = utils.tqdm_progress_fn(pbar)

            if self.args.solver_type == "sinkhorn":
                if self.args.low_rank == True:
                    if self.args.rank == "auto":
                        solve_fn = sinkhorn_lr.LRSinkhorn(rank=int(min(len(X), len(Y)) / 2), progress_fn=progress_fn)
                    else:
                        solve_fn = sinkhorn_lr.LRSinkhorn(rank=self.args.rank, progress_fn=progress_fn)

                else:
                    solve_fn = sinkhorn.Sinkhorn(progress_fn=progress_fn)
                    
                ot = jax.jit(solve_fn)(ot_prob)
            
            else:
                raise NotImplementedError

        if self.args.verbose:
            print(
            "\nSinkhorn has converged: ",
            ot.converged,
            "\n",
            "-Error upon last iteration: ",
            ot.errors[(ot.errors > -1)][-1],
            "\n",
            "-Sinkhorn required ",
            jnp.sum(ot.errors > -1),
            " iterations to converge. \n",
            "-Entropy regularized OT cost: ",
            ot.reg_ot_cost,
            "\n",
            "-OT cost (without entropy): ",
            jnp.sum(ot.matrix * ot.geom.cost_matrix),
            )

        return ot.matrix.__array__()
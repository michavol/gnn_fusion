import sys
import wandb

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


# TODO: Invetsigate if we don't have a bug while passing a cost function
class OptimalTransport:
    def __init__(self, cfg):
        self.cfg = cfg
        self.args = cfg.optimal_transport

        # Initialize Solver
        with tqdm.tqdm(disable=self.args.disable_tqdm) as pbar:
            progress_fn = utils.tqdm_progress_fn(pbar)

        if self.args.solver_type == "sinkhorn":

            if self.args.low_rank == True:
                if self.args.rank == "auto":
                    self.solver = jax.jit(
                        sinkhorn_lr.LRSinkhorn(
                            progress_fn=progress_fn
                        )
                    )
                else:
                    self.solver = jax.jit(
                        sinkhorn_lr.LRSinkhorn(
                            rank=self.args.rank,
                            progress_fn=progress_fn
                        )
                    )

            else:
                self.solver = jax.jit(
                    sinkhorn.Sinkhorn(
                        max_iterations=self.args.max_iterations,
                        progress_fn=progress_fn)
                )

        else:
            raise NotImplementedError

    def get_current_transport_map(self, X, Y, a, b, layer_type="gcn", mode='acts'):
        """
        Solve optimal transport problem for activation support for GNN Fusion
        """

        if mode == 'wts' or layer_type in [LayerType.mlp, LayerType.embedding]:
            # Compute cost matrix
            cost_matrix = GroundCostMlp(self.cfg).get_cost_matrix(X, Y)
        elif layer_type == LayerType.gcn:
            # Compute cost matrix
            cost_matrix = GroundCostGcn(self.cfg).get_cost_matrix(X, Y)
        else:
            raise NotImplementedError

        print('cm', cost_matrix)

        # Define Geometry
        geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=self.args.relative_epsilon, relative_epsilon=True)

        # Define Problem
        ot_prob = linear_problem.LinearProblem(geom, tau_a=self.args.tau_a, tau_b=self.args.tau_b)

        # Solve Problem
        # Progress
        if self.args.verbose:
            print("=============================================")
            print("Solving OT problem...")

        # Solver OT problem
        ot = self.solver(ot_prob)

        if self.args.verbose:
            print(
                "\nSinkhorn has converged: ",
                ot.converged,
                "\n",
                "-Error upon last iteration: ",
                ot.errors[(ot.errors > -1)][-1] if len(ot.errors[(ot.errors > -1)]) else -1,
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

        if self.cfg.wandb:
            wandb.log({"converged": ot.converged})
            wandb.log({"error": ot.errors[(ot.errors > -1)][-1]})
            wandb.log({"iterations": jnp.sum(ot.errors > -1)})
            wandb.log({"reg_ot_cost": ot.reg_ot_cost})
            wandb.log({"ot_cost": jnp.sum(ot.matrix * ot.geom.cost_matrix)})

        return ot.matrix.__array__()

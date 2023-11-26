from .cost_matrix import CostMatrix
import ott
from ott import utils
from ott.math import utils as mu
from ott.geometry import geometry, pointcloud
from ott.geometry.graph import Graph
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
import tqdm 
import jax
import jax.numpy as jnp

class TransportMap:
    def __init__(self, cfg):
        self.cfg = cfg
        self.args = cfg.transport_map

    def get_current_transport_map(self, X, Y, a, b):
        """
        Solve optimal transport problem for activation support for GNN Fusion
        """
        # Compute cost matrix
        cost_matrix = CostMatrix(self.cfg).get_cost_matrix(X, Y)

        # Define Geometry
        geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=self.args.epsilon) 

        # Define Problem
        ot_prob = linear_problem.LinearProblem(geom, tau_a=self.args.tau_a, tau_b=self.args.tau_b)

        # Solve Problem       
        with tqdm.tqdm() as pbar:
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

        return ot.matrix
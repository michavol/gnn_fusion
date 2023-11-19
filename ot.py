import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import torch
import jax
import jax.numpy as jnp
import pandas as pd 
import networkx as nx

import ott
from ott import utils
from ott.math import utils as mu
from ott import problems
from ott.geometry import geometry, pointcloud, costs
from ott.solvers import linear
from ott.solvers.linear import acceleration, sinkhorn, sinkhorn_lr
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from ott.problems.linear import linear_problem
import tqdm

def _cost_fn_l1(x, y):
    n = len(x)
    assert n == len(y)
    cost = 0
    for i in range(n):
        cost += torch.norm(x[i].ndata["Feature"] - y[i].ndata["Feature"], dim=0)
    return cost

def _get_cost_matrix(X, Y, cost_fn):
    """
    X: list of dgl graphs
    Y: list of dgl graphs
    """
    cost_matrix = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            cost_matrix[i][j] = cost_fn(X[i], Y[j])
    return jnp.array(cost_matrix)

def _get_current_transport_map(a, b, X, Y, cost_fn, epsilon = None, tau_a = 1, tau_b = 1, low_rank = False, verbose = False):
    """
    Solve optimal transport problem for activation support for GNN Fusion
    """
    # Define Geometry
    cost_matrix = _get_cost_matrix(X, Y, cost_fn)
    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon) 

    # Define Problem
    ot_prob = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)

    # Solve Problem       
    with tqdm.tqdm() as pbar:
        progress_fn = utils.tqdm_progress_fn(pbar)

        if low_rank == True:
            solve_fn = sinkhorn_lr.LRSinkhorn(rank=int(min(len(x), len(y)) / 2), progress_fn=progress_fn)
        else:
            solve_fn = sinkhorn.Sinkhorn(progress_fn=progress_fn)
            
        ot = jax.jit(solve_fn)(ot_prob)

    if verbose:
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

    return ot
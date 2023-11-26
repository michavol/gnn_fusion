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
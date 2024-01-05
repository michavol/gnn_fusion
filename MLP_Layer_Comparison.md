# Observations
- He does take the square root.
- He clamps the distance with min = 0 - he sets everything to max(0,x).
- He normalizes the distances by the number of samples.
- There are multiple ways to normalize the cost matrix, but our parameter sets it such that we don't do it.
- He provides the option to clip the values of the cost matrix by 10 percent, but we don't do it currently.

# Comparison
- Neither of our ground_costs do scaling of the cost matrix. I added it in the optimal transport. 

# Changes
- I added scaling to our outer sinkhorn solver

# Further works / TODO
- Analysis of trend of finetuning with best config.
- Benchmark on MNIST, to see whether similar results on other dataset.
- Make gromov wassertsein cost more efficient.


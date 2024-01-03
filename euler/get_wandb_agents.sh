cd ..
echo "EMD:"
wandb sweep src/conf/experiments/optimization/sweep_emd.yaml

echo "Sinkhorn:"
wandb sweep src/conf/experiments/optimization/sweep_sinkhorn.yaml

# export WANDB_START_METHOD=thread
echo "Debug:"
wandb sweep src/conf/experiments/optimization/sweep_debug.yaml


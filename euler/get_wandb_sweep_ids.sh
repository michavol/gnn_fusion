# export WANDB_START_METHOD=thread
cd ..
echo "EMD:"
wandb sweep src/conf/experiments/optimization/sweep_emd.yaml

echo "Sinkhorn:"
wandb sweep src/conf/experiments/optimization/sweep_sinkhorn.yaml

echo "Debug:"
wandb sweep src/conf/experiments/optimization/sweep_debug.yaml


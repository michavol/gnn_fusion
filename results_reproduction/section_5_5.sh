# Table 3 - Fusing MLPs vs fusing GCNs
mkdir src/conf/models/mlp_models
mkdir models/mlp_models
mkdir src/conf/models/gcn_models
mkdir models/gcn_models

python src/model_fusion.py graph_cost=feature_lp acts_from_bn=False batch_size=340 fine_tune=False optimal_transport=emd experiment_models_dir=/gcn_models/ models_conf_dir=../../../src/conf/models/gcn_models/ models/individual_models=[GCN_ZINC_GPU-1_19h03m35s_on_Jan_06_2024,GCN_ZINC_GPU-1_19h18m15s_on_Jan_06_2024]
python src/model_fusion.py graph_cost=feature_lp acts_from_bn=False batch_size=340 fine_tune=False optimal_transport=emd experiment_models_dir=/mlp_models/ models_conf_dir=../../../src/conf/models/mlp_models/ models/individual_models=[MLP_ZINC_GPU-1_16h54m39s_on_Jan_07_2024,MLP_ZINC_GPU-1_16h57m50s_on_Jan_07_2024]

echo "Results from Table 3"
python src/optimization_experiment.py config_dir=../../../src/conf/models/gcn_models/ evaluate_in_place=True write_to_csv=False ensemble=False
python src/optimization_experiment.py config_dir=../../../src/conf/models/mlp_models/ evaluate_in_place=True write_to_csv=False ensemble=False

rm -r src/conf/models/mlp_models
rm -r models/mlp_models
rm -r src/conf/models/gcn_models
rm -r models/gcn_models

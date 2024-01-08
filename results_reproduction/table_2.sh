# Table 2 - Best hyperparameter selection for each cost
mkdir src/conf/models/best_models
mkdir models/best_models

python src/model_fusion.py graph_cost=feature_lp acts_from_bn=True batch_size=340 optimal_transport=emd experiment_models_dir=/best_models/ models_conf_dir=../../../src/conf/models/best_models/ fine_tune=True

echo "Results from Table 2"
python src/optimization_experiment.py config_dir=../../../src/conf/models/best_models/ evaluate_in_place=True write_to_csv=False

rm -r src/conf/models/best_models
rm -r models/best_models

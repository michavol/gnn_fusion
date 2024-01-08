# Table 3 - Best hyperparameter selection for each cost
mkdir src/conf/models/best_models
mkdir models/best_models

python src/model_fusion.py graph_cost=feature_lp acts_from_bn=True batch_size=340 optimal_transport=emd experiment_models_dir=/best_models/ models_conf_dir=../../../src/conf/models/best_models/ fine_tune=True

echo "Results from Table 2"
python src/optimization_experiment.py config_dir=../../../src/conf/models/best_models/ evaluate_in_place=True write_to_csv=False

rm -r src/conf/models/best_models
rm -r models/best_models

# Table 2 - Comparison with taking activation pre BN
mkdir src/conf/models/prebn_models
mkdir models/prebn_models

python src/model_fusion.py graph_cost=feature_lp acts_from_bn=True batch_size=340 optimal_transport=emd experiment_models_dir=/prebn_models/ models_conf_dir=../../../src/conf/models/prebn_models/ fine_tune=True acts_from_bn=False

echo "Results from section 5.3"
python src/optimization_experiment.py config_dir=../../../src/conf/models/prebn_models/ evaluate_in_place=True write_to_csv=False

rm -r src/conf/models/prebn_models
rm -r models/prebn_models
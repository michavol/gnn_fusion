# Table 1 - Best hyperparameter selection for each cost
# EFD cost
python src/model_fusion.py graph_cost=feature_lp acts_from_bn=True batch_size=100 optimal_transport=emd
python src/model_fusion.py graph_cost=feature_lp acts_from_bn=True batch_size=100 optimal_transport=sinkhorn optimal_transport.epsilon=0.0005
# QE cost
python src/model_fusion.py graph_cost=quadratic_energy_alpha_02 acts_from_bn=True batch_size=100 optimal_transport=emd
python src/model_fusion.py graph_cost=quadratic_energy_alpha_02 acts_from_bn=True batch_size=100 optimal_transport=sinkhorn optimal_transport.epsilon=0.00005
# FGW cost
python src/model_fusion.py graph_cost=quadratic_energy_alpha_02 acts_from_bn=True batch_size=2 optimal_transport=emd
python src/model_fusion.py graph_cost=quadratic_energy_alpha_02 acts_from_bn=True batch_size=2 optimal_transport=sinkhorn optimal_transport.epsilon=0.00005
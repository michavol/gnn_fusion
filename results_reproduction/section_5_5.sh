# Table 4 - Effects of using activations from before and after BN
# Activations taken after BN
python src/model_fusion.py graph_cost=feature_lp acts_from_bn=True batch_size=340
# Activations taken before BN
python src/model_fusion.py graph_cost=feature_lp acts_from_bn=False batch_size=340

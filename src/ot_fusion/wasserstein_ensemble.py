import argparse
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.append('./src')
from ot_fusion.ot.optimal_transport import OptimalTransport
from utils import activation_operations
from utils import layer_operations
from utils import model_operations
from utils.layer_operations import LayerType

def _reduce_layer_name(layer_name):
    # print("layer0_name is ", layer0_name) It was features.0.weight
    # previous way assumed only one dot, so now I replace the stuff after last dot
    return layer_name.replace('.' + layer_name.split('.')[-1], '')


def _get_histogram(args, cardinality):
    # returns a uniform measure
    if not args.unbalanced:
        print("returns a uniform measure of cardinality: ", cardinality)
        return np.ones(cardinality) / cardinality
    else:
        return np.ones(cardinality)

def _compute_marginals(args, T_var, device, eps=1e-7):
    if args.proper_marginals:
        # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
        marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

        marginals = (1 / (marginals_beta + eps))
        print("shape of inverse marginals beta is ", marginals_beta.shape)
        print("inverse marginals beta is ", marginals_beta)

        T_var = T_var * marginals
        # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
        # this should all be ones, and number equal to number of neurons in 2nd model
        print(T_var.sum(dim=0))
        # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()
    else:
        # think of it as m x 1, scaling weights for m linear combinations of points in X
        marginals = torch.ones(T_var.shape)
        if args.gpu_id != -1:
            marginals = marginals.cuda(args.gpu_id)

        print(T_var.shape, "T_var")
        print(marginals.shape, "marginals")
        marginals = torch.matmul(T_var, marginals)
        marginals = 1 / (marginals + eps)
        print("marginals are ", marginals)

        T_var = T_var * marginals

    print("T_var after correction ", T_var)
    print("T_var stats: max {}, min {}, mean {}, std {} ".format(T_var.max(), T_var.min(), T_var.mean(),
                                                                 T_var.std()))

    return T_var, marginals

def _process_ground_metric_from_acts(layer_name, ground_metric_object, activations):
    layer_type = layer_operations.get_layer_type(layer_name)
    print(layer_type)
    if layer_type in [LayerType.mlp, LayerType.embedding]:
        return ground_metric_object['MLP'].get_cost_matrix(activations[0], activations[1])
    return ground_metric_object[layer_type].get_cost_matrix(activations[0], activations[1])

def _is_bias(layer_name):
    return 'bias' in layer_name


def _get_acts_wassersteinized_layers_modularized(cfg, networks, eps=1e-7, train_loader=None):
    '''
    Average based on the activation vector over data samples. Obtain the transport map,
    and then align the nodes based on it and average the weights!
    Like before: two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    avg_aligned_layers = {}

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))

    # Initialize OT object
    ot = OptimalTransport(cfg.ot)

    # Initialize activations
    activations = activation_operations.compute_selective_activation(cfg, networks, train_loader)
    print('act', activations[0].keys())
    # TODO: Think if this is necessary
    # if cfg.update_acts or cfg.eval_aligned:
    #     model0_aligned_layers = []

    if cfg.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(cfg.gpu_id))

    idx = 0
    T_var = None
    while idx < num_layers:
        print('idx', idx)
        if idx in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
            idx += 1
            continue
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
        print("\n--------------- At layer index {} ------------- \n ".format(idx))
        print('l0', layer0_name, fc_layer0_weight.shape)
        # print('l0', layer0_name, fc_layer0_weight)
        # print('l1', layer1_name, fc_layer1_weight.shape)
        # print('l1', layer0_name, fc_layer1_weight)
        # TODO: Check if this is necessary
        # # layer shape is out x in
        # # assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # assert _check_layer_sizes(cfg, idx, fc_layer0_weight.shape, fc_layer1_weight.shape, num_layers)
        # print("Previous layer shape is ", previous_layer_shape)
        # previous_layer_shape = fc_layer1_weight.shape
        #
        layer0_name_reduced = _reduce_layer_name(layer0_name)
        layer1_name_reduced = _reduce_layer_name(layer1_name)
        layer_type = layer_operations.get_layer_type(layer0_name_reduced)

        # Embedding layer weights are transposed
        if layer_type == LayerType.embedding:
            print('transposed')
            fc_layer0_weight = fc_layer0_weight.T
            fc_layer1_weight = fc_layer1_weight.T

        # if 'batchnorm' in layer0_name_reduced:
        #     print('batch_norm act')
        #     print(activations[0][layer0_name_reduced][0].shape)
        #     print('batch layer')
        #     print(fc_layer0_weight)
        #     print('layers')
        #
        #     print(list(networks[0].modules())[0])
        #     print('batch norm')
        #     print(list(networks[0].modules())[0].layers[0].batchnorm_h.bias.shape)
        #     print(dir(list(networks[0].modules())[0].layers[0].batchnorm_h))
        #     print('m0', list(networks[0].modules())[0].layers[0].batchnorm_h.running_mean)
        #     print('m1', list(networks[1].modules())[0].layers[0].batchnorm_h.running_mean)
        #     print(list(networks[0].modules())[0].layers[0].batchnorm_h.running_var)
        #     print(list(networks[0].modules())[0].layers[0].batchnorm_h.bias.shape)
        #     print(list(networks[0].modules())[0]['layers'])
        #     return
        # else:
        #     idx += 1
        #     continue

        #
        # print("let's see the difference in layer names", layer0_name.replace('.' + layer0_name.split('.')[-1], ''), layer0_name_reduced)
        # print(activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].shape, 'shape of activations generally')
        # # for conv layer I need to make the act_num_samples dimension the last one, but it has the intermediate dimensions for
        # # height and width of channels, so that won't work.
        # # So convert (num_samples, layer_size, ht, wt) -> (layer_size, ht, wt, num_samples)

        activations_0 = activations[0][layer0_name_reduced]
        activations_1 = activations[1][layer1_name_reduced]
        print("a0 shape", activations_0)
        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]


        # TODO: Think if we have to transform the weights anyhow
        # fc_layer0_weight_data = _get_layer_weights(fc_layer0_weight, is_conv)
        # fc_layer1_weight_data = _get_layer_weights(fc_layer1_weight, is_conv)
        fc_layer0_weight_data = fc_layer0_weight
        fc_layer1_weight_data = fc_layer1_weight

        # If we process bias or batch-norm layer we just later on multiply it from the left by the same matrix as in previous step.
        if idx == 0 or layer_type == LayerType.bn or _is_bias(layer0_name):
            aligned_wt = fc_layer0_weight_data

        else:

            print("shape of layer: model 0", fc_layer0_weight_data.shape)
            print("shape of layer: model 1", fc_layer1_weight_data.shape)

            print("shape of previous transport map", T_var.shape)

            # TODO: Think if we should always be able to multiply it this easily (for instance between GNNs and MLPs)
            aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)

        #### Refactored ####

        # if cfg.update_acts:
        #     assert cfg.second_model_name is None
        #     activations_0, activations_1 = _get_updated_acts_v0(cfg, layer_shape, aligned_wt,
        #                                                         model0_aligned_layers, networks,
        #                                                         test_loader, [layer0_name, layer1_name])

        # TODO: Why for the last layer we do this and not the importance histogram
        if not cfg.importance or (idx == num_layers - 1):
            mu = _get_histogram(cfg, mu_cardinality)
            nu = _get_histogram(cfg, nu_cardinality)
        # else:
        #     # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
        #     mu = _get_neuron_importance_histogram(cfg, fc_layer0_weight_data, is_conv)
        #     nu = _get_neuron_importance_histogram(cfg, fc_layer1_weight_data, is_conv)
        #     # print(mu, nu)
        #     assert cfg.proper_marginals
        #

        # TODO: Think if this shouldn't happen sooner
        if cfg.skip_last_layer and idx == (num_layers - 1):

            if cfg.skip_last_layer_type == 'average':
                avg_aligned_layers[layer1_name] = (1 - cfg.ensemble_step) * aligned_wt + cfg.ensemble_step * fc_layer1_weight
            elif cfg.skip_last_layer_type == 'second':
                print("Just giving the weights of the second model. NO transport map needs to be computed")
                avg_aligned_layers[layer1_name] = fc_layer1_weight
            else:
                raise NotImplementedError(f"skip_last_layer_type: {cfg.skip_last_layer_type}. Value not known!")

            return avg_aligned_layers
        print(layer_type)
        if not layer_type == LayerType.bn and not _is_bias(layer0_name):
            T_var = torch.tensor(ot.get_current_transport_map(activations_0, activations_1, mu, nu,
                                             layer_type=layer_type))

            # TODO: What is this correction?
            if cfg.correction:
                T_var, marginals = _compute_marginals(cfg, T_var, device, eps=eps)

            if cfg.debug:
                if idx == (num_layers - 1):
                    print("there goes the last transport map: \n ", T_var)
                    print("and before marginals it is ", T_var / marginals)
                else:
                    print("there goes the transport map at layer {}: \n ".format(idx), T_var)

            print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
            print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))

        # TODO: Why do we have this if?
        if cfg.past_correction:
            print("Shape of aligned wt is ", aligned_wt.shape)
            print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt)
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data)

        print(t_fc0_model.shape)
        print(fc_layer1_weight_data.shape)
        # Average the weights of aligned first layers
        geometric_fc = (1 - cfg.ensemble_step) * t_fc0_model + \
                           cfg.ensemble_step * fc_layer1_weight_data

        # Embedding layer weights are transposed
        if layer_type == LayerType.embedding:
            geometric_fc = geometric_fc.T

        avg_aligned_layers[layer1_name] = geometric_fc


        print("The averaged parameters are :", geometric_fc)
        print("The model0 and model1 parameters were :", fc_layer0_weight.data, fc_layer1_weight.data)

        idx += 1

    return avg_aligned_layers


def _get_network_and_performance_from_param_list(cfg, avg_aligned_layers, test_loader=None):
    print("using independent method")
    # TODO: Change it to some unified way of getting the model
    models_conf = OmegaConf.to_container(
        cfg.models.individual_models, resolve=True, throw_on_missing=True
    )
    new_network = model_operations.get_models(models_conf,cfg.individual_models_dir)[1]
    if cfg.gpu_id != -1:
        new_network = new_network.cuda(cfg.gpu_id)

    if test_loader is not None:
        pass
        # # check the test performance of the network before
        # log_dict = {}
        # log_dict['test_losses'] = []
        # routines.test(cfg, new_network, test_loader, log_dict)



    # Set new network parameters
    model_state_dict = new_network.state_dict()

    print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of param_list is ", len(avg_aligned_layers))

    for key, value in avg_aligned_layers.items():
        model_state_dict[key] = avg_aligned_layers[key]

    new_network.load_state_dict(model_state_dict)

    if test_loader is not None:
        pass
        # # check the test performance of the network after
        # log_dict = {}
        # log_dict['test_losses'] = []
        # acc = routines.test(cfg, new_network, test_loader, log_dict)
        # print(log_dict)

    return new_network


def compose_models(args: argparse.Namespace, models: List, train_loader: DataLoader, test_loader: DataLoader):
    if args.geom_ensemble_type == 'wts':
        pass
    elif args.geom_ensemble_type == 'acts':
        avg_aligned_layers = _get_acts_wassersteinized_layers_modularized(args, models, train_loader=train_loader)

    return _get_network_and_performance_from_param_list(args, avg_aligned_layers, test_loader)

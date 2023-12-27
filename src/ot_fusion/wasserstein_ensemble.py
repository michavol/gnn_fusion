import argparse
import copy
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


def _compute_marginals(cfg, T_var, device, eps=1e-7):
    marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

    marginals = (1 / (marginals_beta + eps))
    print("shape of inverse marginals beta is ", marginals_beta.shape)
    print("inverse marginals beta is ", marginals_beta)

    T_var = T_var * marginals
    # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
    # this should all be ones, and number equal to number of neurons in 2nd model
    print(T_var.sum(dim=0))

    return T_var, marginals


def _process_ground_metric_from_acts(layer_name, ground_metric_object, activations):
    layer_type = layer_operations.get_layer_type(layer_name)
    print(layer_type)
    if layer_type in [LayerType.mlp, LayerType.embedding]:
        return ground_metric_object['MLP'].get_cost_matrix(activations[0], activations[1])
    return ground_metric_object[layer_type].get_cost_matrix(activations[0], activations[1])


def _is_bias(layer_name):
    return 'bias' in layer_name


def _adjust_weights(layer_type: LayerType, weights):
    """Transposes weights to the same setting as MLP."""
    if layer_type in [LayerType.embedding, LayerType.gcn]:
        return weights.T

    return weights


def _get_network_from_param_list(cfg, aligned_layers, model=None):
    if model is None:
        new_model = model_operations.get_models_from_raw_config(cfg)[0]
    else:
        new_model = copy.deepcopy(model)
    if cfg.gpu_id != -1:
        new_model = new_model.cuda(cfg.gpu_id)

    # Set new network parameters
    model_state_dict = new_model.state_dict()

    print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of param_list is ", len(aligned_layers))

    for key, value in aligned_layers.items():
        model_state_dict[key] = aligned_layers[key]

    new_model.load_state_dict(model_state_dict)

    return new_model


def _get_updated_acts(cfg, aligned_layers, networks, train_loader):
    """Updates the activations based on the aligned model."""
    new_model = _get_network_from_param_list(cfg, aligned_layers, networks[0])
    activations = activation_operations.compute_activations(cfg, [new_model] + networks[1:], train_loader)
    return activations


def _check_layer_sizes(args, shape1, shape2):
    if args.width_ratio == 1:
        return shape1 == shape2
    else:

        return (shape1[0] / shape2[0]) == args.width_ratio


def _get_acts_wassersteinized_layers(cfg, networks, eps=1e-7, train_loader=None):
    '''
    Average based on the activation vector over data samples. Obtain the transport map,
    and then align the nodes based on it and average the weights!
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.

    :param cfg: global config
    :param networks: list of networks
    :param eps: constant needed for numerical stability while computing marginals
    :param train_loader: data loader for computing pre-activations

    :return: list of layer weights 'wassersteinized'
    '''

    avg_aligned_layers = {}

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))

    # Initialize OT object
    ot = OptimalTransport(cfg)

    # Initialize activations
    activations = activation_operations.compute_activations(cfg, networks, train_loader)

    if cfg.update_acts:
        model0_aligned_layers = {}

    if cfg.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(cfg.gpu_id))

    idx = 0
    while idx < num_layers:
        # # Uncomment below lines for debugging
        # if idx in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        #     idx += 1
        #     continue
        ((layer0_name, layer0_weight), (layer1_name, layer1_weight)) = networks_named_params[idx]
        print(f"\n--------------- At layer index {idx} ------------- \n ")
        print('l0', layer0_name, layer0_weight.shape)

        assert _check_layer_sizes(cfg, layer0_weight.shape, layer1_weight.shape)

        layer0_name_reduced = _reduce_layer_name(layer0_name)
        layer1_name_reduced = _reduce_layer_name(layer1_name)
        layer_type = layer_operations.get_layer_type(layer0_name_reduced)

        # Some layer types have transposed weights
        layer0_weight = _adjust_weights(layer_type, layer0_weight)
        layer1_weight = _adjust_weights(layer_type, layer1_weight)

        a_cardinality = layer0_weight.shape[0]
        b_cardinality = layer1_weight.shape[0]

        fc_layer0_weight_data = layer0_weight
        fc_layer1_weight_data = layer1_weight

        # Align the weights with a matrix from the previous step. The first layer is already aligned.
        if idx == 0:
            aligned_wt = fc_layer0_weight_data

        else:
            # TODO: Think if we should always be able to multiply it this easily (for instance between GNNs and MLPs)
            aligned_wt = torch.matmul(layer0_weight.data, T_var)

        if cfg.skip_last_layer and idx == (num_layers - 1):
            if cfg.skip_last_layer_type == 'average':
                avg_aligned_layers[layer1_name] = (1 - cfg.ensemble_step) * aligned_wt + cfg.ensemble_step * layer1_weight
            elif cfg.skip_last_layer_type == 'second':
                print("Just giving the weights of the second model. NO transport map needs to be computed")
                avg_aligned_layers[layer1_name] = layer1_weight
            else:
                raise NotImplementedError(f"skip_last_layer_type: {cfg.skip_last_layer_type}. Value not known!")

            return avg_aligned_layers

        if cfg.update_acts:
            activations = _get_updated_acts(cfg, model0_aligned_layers, networks, train_loader)

        activations_0 = activations[0][layer0_name_reduced]
        activations_1 = activations[1][layer1_name_reduced]
        print("a0 shape", activations_0)

        # Probability masses (weight vectors) for source (a - expected sum of rows of T_var) and target
        # (b - expected sum of rows of T_var). Chosen uniformly (the first branch of the if) or through
        # weight importance (the second branch of the if).
        if not cfg.importance or (idx == num_layers - 1):
            a = _get_histogram(cfg, a_cardinality)
            b = _get_histogram(cfg, b_cardinality)
        else:
            raise NotImplementedError
        # TODO: Implement the support for importance histogram
        #     # a = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
        #     a = _get_neuron_importance_histogram(cfg, fc_layer0_weight_data, is_conv)
        #     b = _get_neuron_importance_histogram(cfg, fc_layer1_weight_data, is_conv)
        #

        if not layer_type == LayerType.bn and not _is_bias(layer0_name):
            T_var = torch.tensor(ot.get_current_transport_map(activations_0, activations_1, a, b,
                                                              layer_type=layer_type))

            # This makes sure that the transport matrix performs a convex combination of the source.
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

        if cfg.past_correction:
            print("Shape of aligned wt is ", aligned_wt.shape)
            print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt)
        else:
            # We probably won't use this. This would only make sense if we don't update the activations.
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data)

        if cfg.update_acts:
            model0_aligned_layers[layer1_name] = _adjust_weights(layer_type, t_fc0_model)

        # Average the weights of aligned first layers
        geometric_fc = (1 - cfg.ensemble_step) * t_fc0_model + cfg.ensemble_step * fc_layer1_weight_data

        # Some layer types have transposed weights
        geometric_fc = _adjust_weights(layer_type, geometric_fc)

        avg_aligned_layers[layer1_name] = geometric_fc

        print("The averaged parameters are :", geometric_fc)
        print("The model0 and model1 parameters were :", layer0_weight.data, layer1_weight.data)

        idx += 1

    return avg_aligned_layers


def compose_models(args: argparse.Namespace, models: List, train_loader: DataLoader, test_loader: DataLoader) -> float:
    if args.geom_ensemble_type == 'wts':
        pass
    elif args.geom_ensemble_type == 'acts':
        avg_aligned_layers = _get_acts_wassersteinized_layers(args, models, train_loader=train_loader)

    return _get_network_from_param_list(args, avg_aligned_layers)

# TODO: What to do with the batch-norm statistics

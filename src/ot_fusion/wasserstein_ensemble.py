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
        return weights.t()

    return weights


def _get_network_from_param_list(cfg, aligned_layers, model=None):
    if model is None:
        new_model = model_operations.get_models_from_raw_config(cfg)[0]
    else:
        new_model = copy.deepcopy(model)
    # TODO: Check what does this param do
    if cfg.gpu_id != -1:
        new_model = new_model.cuda(cfg.gpu_id)

    # Set new network parameters
    model_state_dict = new_model.state_dict()

    print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of param_list is ", len(aligned_layers))

    for key, value in aligned_layers.items():
        model_state_dict[key] = aligned_layers[key]

    new_model.load_state_dict(model_state_dict)

    # for key, value in new_model.state_dict().items():
    #     try:
    #         print(key, value.shape, value[0])
    #     except:
    #         pass
    return new_model


def _get_updated_acts(cfg, aligned_layers, network, target_network, train_loader, processed_layer_name):
    """Updates the activations based on the aligned model."""
    new_model = _get_network_from_param_list(cfg, aligned_layers, network)
    activations = activation_operations.compute_activations(cfg, [new_model, target_network], train_loader, layer_to_break_after='.'.join(processed_layer_name.split('.')[:-1]))
    return activations


def _check_layer_sizes(args, shape1, shape2):
    if args.width_ratio == 1:
        return shape1 == shape2
    else:
        return (shape1[0] / shape2[0]) == args.width_ratio


def _get_neuron_importance_histogram(args, layer_weight, eps=1e-9):
    layer = layer_weight.cpu().detach().numpy()

    if args.importance == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
            np.float64) + eps
    elif args.importance == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
            np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = (importance_hist / importance_hist.sum())
        print('sum of importance hist is ', importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist


def _is_last_layer(weights):
    return weights.shape[0] == 1


def _get_acts_wassersteinized_layers_for_single_model(cfg, network, target_network, train_loader, eps=1e-7):
    """
    Average based on the activation vector over data samples. Obtain the transport map,
    and then align the nodes based on it and average the weights!
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.

    :param cfg: global config
    :param network: network to be aligned
    :param target_network: network to which we align
    :param train_loader: data loader for computing pre-activations
    :param eps: constant needed for numerical stability while computing marginals

    :return: list of aligned weights of network, list of layer weights 'wassersteinized' (returned just as sanity check)
    """
    # For consistent performance, since we have batch norm layer
    original_mode = target_network.training
    network.eval()
    target_network.eval()

    avg_aligned_layers = {}

    networks_named_params = list(zip(network.state_dict().items(), target_network.state_dict().items()))
    num_layers = len(networks_named_params)

    # Initialize OT object
    ot = OptimalTransport(cfg)

    # Initialize activations
    activations = activation_operations.compute_activations(cfg, [network, target_network], train_loader)

    model0_aligned_layers = {}

    if cfg.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(cfg.gpu_id))

    idx = 0
    while idx < num_layers:
        # Uncomment below lines for debugging
        # if idx in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        #     idx += 1
        #     continue
        ((layer0_name, layer0_weight), (target_layer_name, target_layer_weight)) = networks_named_params[idx]
        print(f"\n--------------- At layer index {idx} ------------- \n ")
        print('l0', layer0_name, layer0_weight.shape)
        print('l0', target_layer_name, target_layer_weight.shape)
        # This layer in batch nom is just a single number
        if layer0_name.endswith('num_batches_tracked'):
            idx += 1
            continue

        # assert _check_layer_sizes(cfg, layer0_weight.shape, target_layer_weight.shape)

        layer0_name_reduced = _reduce_layer_name(layer0_name)
        target_layer_name_reduced = _reduce_layer_name(target_layer_name)
        layer_type = layer_operations.get_layer_type(layer0_name_reduced)

        # Some layer types have transposed weights
        layer0_weight = _adjust_weights(layer_type, layer0_weight)
        target_layer_weight = _adjust_weights(layer_type, target_layer_weight)

        a_cardinality = layer0_weight.shape[0]
        b_cardinality = target_layer_weight.shape[0]

        # Align the weights with a matrix from the previous step. The first layer is already aligned.
        if idx == 0 or layer_type == LayerType.bn or _is_bias(layer0_name):
            aligned_wt = layer0_weight
        else:
            # TODO: Think if we should always be able to multiply it this easily (for instance between GNNs and MLPs)
            aligned_wt = torch.matmul(layer0_weight, T_var)

        if cfg.skip_last_layer and _is_last_layer(layer0_weight):
            print(f"Skipping layer {layer0_name}")
            if cfg.skip_last_layer_type == 'average':
                avg_aligned_layers[target_layer_name] = (
                                                                1 - cfg.ensemble_step) * aligned_wt + cfg.ensemble_step * target_layer_weight
                model0_aligned_layers[target_layer_name] = aligned_wt
            elif cfg.skip_last_layer_type == 'second':
                print("Just giving the weights of the second model. No transport map needs to be computed")
                avg_aligned_layers[target_layer_name] = target_layer_weight
                model0_aligned_layers[target_layer_name] = target_layer_weight
            else:
                raise NotImplementedError(f"skip_last_layer_type: {cfg.skip_last_layer_type}. Value not known!")

            idx += 1
            continue

        # Probability masses (weight vectors) for source (a - expected sum of rows of T_var) and target
        # (b - expected sum of rows of T_var). Chosen uniformly (the first branch of the if) or through
        # weight importance (the second branch of the if).
        if not cfg.importance or (idx == num_layers - 1):
            a = _get_histogram(cfg, a_cardinality)
            b = _get_histogram(cfg, b_cardinality)
        else:
            a = _get_neuron_importance_histogram(cfg, layer0_weight)
            b = _get_neuron_importance_histogram(cfg, target_layer_weight)

        if not layer_type == LayerType.bn and not _is_bias(layer0_name):

            if cfg.update_acts:
                # We correct the activations also for the last aligned layer
                # TODO: Figure out how to do this for models with varying size
                model0_aligned_layers[target_layer_name] = _adjust_weights(layer_type, aligned_wt)
                activations = _get_updated_acts(cfg, model0_aligned_layers, network, target_network, train_loader, layer0_name)

            activations_0 = activations[0][layer0_name_reduced]
            activations_1 = activations[1][target_layer_name_reduced]

            T_var = torch.tensor(ot.get_current_transport_map(activations_0, activations_1, a, b,
                                                              layer_type=layer_type))
            # Random permutation matrix for debugging
            # T_var = torch.eye(a.shape[0], b.shape[0])
            # T_var = T_var[torch.randperm(T_var.size()[0])]


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
            print("Shape of fc_layer0_weight_data is ", layer0_weight.shape)

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt)
        else:
            # We probably won't use this. This would only make sense if we don't update the activations.
            t_fc0_model = torch.matmul(T_var.t(), layer0_weight)

        model0_aligned_layers[target_layer_name] = _adjust_weights(layer_type, t_fc0_model)

        # Average the weights of aligned first layers
        geometric_fc = (1 - cfg.ensemble_step) * t_fc0_model + cfg.ensemble_step * target_layer_weight

        # Some layer types have transposed weights
        geometric_fc = _adjust_weights(layer_type, geometric_fc)

        avg_aligned_layers[target_layer_name] = geometric_fc

        print("The averaged parameters are :", geometric_fc)
        print("The model0 and model1 parameters were :", layer0_weight, target_layer_weight)

        idx += 1
    print('model0_aligned_layers', model0_aligned_layers)
    print('model0_aligned_layers', type(model0_aligned_layers))

    # Setting the mode back to the original state
    network.train(mode=original_mode)
    target_network.train()

    return model0_aligned_layers, avg_aligned_layers


def _get_wts_wassersteinized_layers_for_single_model(cfg, network, target_network, eps=1e-7):
    """
    Average based on the weight vector over data samples. Obtain the transport map,
    and then align the nodes based on it and average the weights!
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.

    :param cfg: global config
    :param network: network to be aligned
    :param target_network: network to which we align
    :param eps: constant needed for numerical stability while computing marginals

    :return: list of aligned weights of network, list of layer weights 'wassersteinized' (returned just as sanity check)
    """
    # For consistent performance, since we have batch norm layer
    original_mode = target_network.training
    network.eval()
    target_network.eval()

    avg_aligned_layers = {}

    # We take state_dict instead of _named_params, because it also contains the
    networks_named_params = list(zip(network.state_dict().items(), target_network.state_dict().items()))
    num_layers = len(networks_named_params)

    # Initialize OT object
    ot = OptimalTransport(cfg)

    model0_aligned_layers = {}

    if cfg.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(cfg.gpu_id))

    idx = 0
    while idx < num_layers:
        # Uncomment below lines for debugging
        # if idx in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        #     idx += 1
        #     continue
        ((layer0_name, layer0_weight), (target_layer_name, target_layer_weight)) = networks_named_params[idx]
        print(f"\n--------------- At layer index {idx} ------------- \n ")
        print('l0', layer0_name, layer0_weight.shape)
        print('l0', target_layer_name, target_layer_weight)
        # This layer in batch nom is just a single number
        if layer0_name.endswith('num_batches_tracked'):
            idx += 1
            continue

        # assert _check_layer_sizes(cfg, layer0_weight.shape, target_layer_weight.shape)

        layer0_name_reduced = _reduce_layer_name(layer0_name)
        target_layer_name_reduced = _reduce_layer_name(target_layer_name)
        layer_type = layer_operations.get_layer_type(layer0_name_reduced)

        # Some layer types have transposed weights
        layer0_weight = _adjust_weights(layer_type, layer0_weight)
        target_layer_weight = _adjust_weights(layer_type, target_layer_weight)

        a_cardinality = layer0_weight.shape[0]
        b_cardinality = target_layer_weight.shape[0]

        # Align the weights with a matrix from the previous step. The first layer is already aligned.
        if idx == 0 or layer_type == LayerType.bn or _is_bias(layer0_name):
            aligned_wt = layer0_weight

        else:
            # TODO: Think if we should always be able to multiply it this easily (for instance between GNNs and MLPs)
            aligned_wt = torch.matmul(layer0_weight, T_var)

        if cfg.skip_last_layer and _is_last_layer(layer0_weight):
            print(f"Skipping layer {layer0_name}")
            if cfg.skip_last_layer_type == 'average':
                avg_aligned_layers[target_layer_name] = (
                                                                1 - cfg.ensemble_step) * aligned_wt + cfg.ensemble_step * target_layer_weight
                model0_aligned_layers[target_layer_name] = aligned_wt
            elif cfg.skip_last_layer_type == 'second':
                print("Just giving the weights of the second model. No transport map needs to be computed")
                avg_aligned_layers[target_layer_name] = target_layer_weight
                model0_aligned_layers[target_layer_name] = target_layer_weight
            else:
                raise NotImplementedError(f"skip_last_layer_type: {cfg.skip_last_layer_type}. Value not known!")

            idx += 1
            continue
            # return avg_aligned_layers

        # Probability masses (weight vectors) for source (a - expected sum of rows of T_var) and target
        # (b - expected sum of rows of T_var). Chosen uniformly (the first branch of the if) or through
        # weight importance (the second branch of the if).
        if not cfg.importance or (idx == num_layers - 1):
            a = _get_histogram(cfg, a_cardinality)
            b = _get_histogram(cfg, b_cardinality)
        else:
            a = _get_neuron_importance_histogram(cfg, layer0_weight)
            b = _get_neuron_importance_histogram(cfg, target_layer_weight)

        if not layer_type == LayerType.bn and not _is_bias(layer0_name):
            T_var = torch.tensor(
                ot.get_current_transport_map(layer0_weight.detach(), target_layer_weight.detach(), a, b,
                                             layer_type=layer_type, mode='wts'))

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
            print("Shape of fc_layer0_weight_data is ", layer0_weight.shape)

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt)
        else:
            # We probably won't use this. This would only make sense if we don't update the activations.
            t_fc0_model = torch.matmul(T_var.t(), layer0_weight)

        model0_aligned_layers[target_layer_name] = _adjust_weights(layer_type, t_fc0_model)

        # Average the weights of aligned first layers
        geometric_fc = (1 - cfg.ensemble_step) * t_fc0_model + cfg.ensemble_step * target_layer_weight

        # Some layer types have transposed weights
        geometric_fc = _adjust_weights(layer_type, geometric_fc)

        avg_aligned_layers[target_layer_name] = geometric_fc

        print("The averaged parameters are :", geometric_fc)
        print("The model0 and model1 parameters were :", layer0_weight, target_layer_weight)

        idx += 1
    print('model0_aligned_layers', model0_aligned_layers)
    print('model0_aligned_layers', type(model0_aligned_layers))

    # Setting the mode back to the original state
    network.train(mode=original_mode)
    target_network.train()

    return model0_aligned_layers, avg_aligned_layers


def _avg_model_from_aligned_layers(args, aligned_layers_all_models, target):
    # Copy of first model as template
    avg_model = copy.deepcopy(target)
    avg_model_state_dict = avg_model.state_dict()

    if len(aligned_layers_all_models) == 2:
        weights = [args.ensemble_step, 1 - args.ensemble_step]
    else:
        weights = None

    # We take the last model, because it will have all the layers that have been aligned.
    # The first one can have some unaligned ones.
    for layer_name, param in aligned_layers_all_models[-1].items():
        with torch.no_grad():
            # Get parameters of all models
            parameters = [aligned_layers[layer_name] for aligned_layers in aligned_layers_all_models]

            avg_parameters = layer_operations.get_avg_parameters(parameters, weights)
            # Set parameters of new model
            avg_model_state_dict[layer_name] = avg_parameters
    avg_model.load_state_dict(avg_model_state_dict)
    return avg_model


def compose_models(args: argparse.Namespace, models: List, train_loader: DataLoader = None) -> float:
    target = copy.deepcopy(models[0])
    aligned_models = []
    aligned_layers_all_models = [copy.deepcopy(target.state_dict())]
    for m in models[1:]:
        if args.geom_ensemble_type == 'wts':
            aligned_layers, _ = _get_wts_wassersteinized_layers_for_single_model(args, m, target)
        elif args.geom_ensemble_type == 'acts':
            aligned_layers, _ = _get_acts_wassersteinized_layers_for_single_model(args, m, target,
                                                                                  train_loader=train_loader)

        aligned_layers_all_models.append(aligned_layers)
        aligned_models.append(_get_network_from_param_list(args, aligned_layers))

    avg_aligned_model = _avg_model_from_aligned_layers(args, aligned_layers_all_models, target)

    return avg_aligned_model, aligned_models

# TODO: Update the code without the dict error
# TODO: Run the sweep
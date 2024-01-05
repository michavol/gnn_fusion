import copy
from typing import Dict, List

import torch
import dgl

from utils.layer_operations import get_layer_type, LayerType


def get_activation(activation: Dict[str, List], name: str):
    """Creates a hook that computes the activations."""

    def hook(model, input, output):
        # print("num of samples seen before", num_samples_processed)
        # print("output is ", output.detach())
        if name not in activation:
            activation[name] = []

        activation[name].append(output.detach())

    return hook


def model_forward(args, model, batch_graphs):
    """Performs the forward pass, so that we can gather the activations through hooks."""
    batch_graphs = batch_graphs.to(args.device)
    batch_x = batch_graphs.ndata['feat'].to(args.device)
    batch_e = batch_graphs.edata['feat'].to(args.device)
    try:
        # TODO: Not sure if we will make use of models with positional encodings.
        batch_pos_enc = batch_graphs.ndata['pos_enc'].to(args.device)
        model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
    except:
        model.forward(batch_graphs, batch_x, batch_e)


def postprocess_activations(args, graphs, activations):
    """Postprocess activations to the form accepted by our OT implementation."""
    if args.fast_l2:
        matrix_based_layers = [LayerType.mlp, LayerType.embedding, LayerType.gcn, LayerType.bn]
    else:
        matrix_based_layers = [LayerType.mlp, LayerType.embedding]
    preprocessed_activations = {}
    # Iterate over models
    for model_name, model_activations in activations.items():
        model_postprocessed_activations = {}
        # Iterate over layers
        for lnum, (layer_name, layer_activations) in enumerate(model_activations.items()):
            if get_layer_type(layer_name) in matrix_based_layers:
                # Transform to num_neurons x num_samples shape
                model_postprocessed_activations[layer_name] = torch.cat(layer_activations, dim=0).T
            elif get_layer_type(layer_name) in [LayerType.bn, LayerType.gcn]:

                graph_layer_activations = [[] for _ in range(layer_activations[0].shape[1])]
                # Iterate over graphs in a dataset and split the activations by neuron (hidden entry)
                for batch_activations, batch_graph in zip(layer_activations, graphs):
                    # Iterate over neurons (hidden entries)
                    for neuron in range(batch_activations.shape[1]):
                        g = dgl.DGLGraph(batch_graph.edges()).to(args.device)
                        g.ndata['Feature'] = batch_activations[:, neuron]
                        graph_layer_activations[neuron].append(g)

                model_postprocessed_activations[layer_name] = graph_layer_activations
            else:
                # TODO: Think if this is necessary
                model_postprocessed_activations[layer_name] = torch.cat(layer_activations, dim=0)
                # raise NotImplementedError(
                #     f"Layer {layer_name} not recognised while processing activations. activation_operation.py")
        preprocessed_activations[model_name] = model_postprocessed_activations
    return preprocessed_activations


def experiment_with_compute_activations(args, model, train_loader):
    '''
    Helper function to understand what how activations are constructed. Not used while fusing.

    :param model: takes in a pretrained model
    :param train_loader: the particular train loader
    :param num_samples: # of randomly selected training examples to average the activations over

    :return:  list of len: num_layers and each of them is a particular tensor of activations
    '''

    activation = {}
    num_batches_processed = 0

    # Set forward hooks for all the layers
    for name, layer in model.named_modules():
        if name == '':
            print("layer excluded")
            continue
        layer.register_forward_hook(get_activation(activation, name))
        print("set forward hook for layer named: ", name)

    # Run over the samples in training set
    with torch.no_grad():
        for batch_idx, (batch_graphs, _) in enumerate(train_loader):
            model_forward(args, model, batch_graphs)
            num_batches_processed += 1
            if num_batches_processed == args.num_batches:
                break
    return activation, None  # , datapoints


def compute_activations(args, models: List[torch.nn.Module], train_loader, layer_to_break_after=None):
    # Prepare all the models
    activations = {}
    forward_hooks = []

    for idx, model in enumerate(models):

        # Initialize the activation dictionary for each model
        activations[idx] = {}
        layer_hooks = []
        # Set forward hooks for all layers inside a model
        for name, layer in model.named_modules():
            if name == '' or get_layer_type(name) == LayerType.dropout:
                print("layer excluded")
            else:
                layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
                print("set forward hook for layer named: ", name)

            # if layer_to_break_after is not None and name == layer_to_break_after:
            #     break

        forward_hooks.append(layer_hooks)
        # Set the model in train mode

    # Run the same data samples ('num_samples' many) across all the models
    all_graphs = []
    num_batches_processed = 0
    with torch.no_grad():
        for batch_idx, (batch_graphs, _) in enumerate(train_loader):
            all_graphs.append(batch_graphs)
            if num_batches_processed == args.num_batches:
                break
            for idx, model in enumerate(models):
                # Send the model to the device
                model.to(args.device)
                model_forward(args, model, batch_graphs)
            num_batches_processed += 1

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()
    print("ac names", activations[0].keys())
    return postprocess_activations(args, all_graphs, activations)

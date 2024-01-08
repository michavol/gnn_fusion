from typing import Dict, List

import torch
import dgl

from utils.layer_operations import get_layer_type, LayerType


def get_activation(args, activation: Dict[str, List], name: str):
    """Creates a hook that computes the activations."""

    def hook(model, input, output):
        if name not in activation:
            activation[name] = []

        if args.take_single_vertex_acts and 'conv' in name:
            o = output[1]
        else:
            o = output

        activation[name].append(o.detach())

    return hook


def model_forward(args, model, batch_graphs):
    """Performs the forward pass, so that we can gather the activations through hooks."""
    batch_graphs = batch_graphs.to(args.device)
    batch_x = batch_graphs.ndata['feat'].to(args.device)
    batch_e = batch_graphs.edata['feat'].to(args.device)
    try:
        # Not sure if we will make use of models with positional encodings.
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
                pass
                # raise NotImplementedError(
                #     f"Layer {layer_name} not recognised while processing activations. activation_operation.py")
        preprocessed_activations[model_name] = model_postprocessed_activations
    return preprocessed_activations


def experiment_with_compute_activations(args, model, train_loader):
    """Helper function to understand what how activations are constructed. Not used while fusing."""

    activation = {}
    num_batches_processed = 0

    # Set forward hooks for all the layers
    for name, layer in model.named_modules():
        if name == '':
            continue
        layer.register_forward_hook(get_activation(args, activation, name))

    # Run over the samples in training set
    with torch.no_grad():
        for batch_idx, (batch_graphs, _) in enumerate(train_loader):
            model_forward(args, model, batch_graphs)
            num_batches_processed += 1
            if num_batches_processed == args.num_batches:
                break
    return activation, None  # , datapoints


def compute_activations(args, models: List[torch.nn.Module], train_loader, layer_to_break_after=None, seed=0):
    """Computes activations and processed them to the form needed for OT-fusion."""
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
                if args.debug:
                    print(f'Layer {name} excluded for pre-activations computation.')
            else:
                layer_hooks.append(layer.register_forward_hook(get_activation(args, activations[idx], name)))
                if args.debug:
                    print(f'Attached hook to layer {name}.')

            # if layer_to_break_after is not None and name == layer_to_break_after:
            #     break

        forward_hooks.append(layer_hooks)

    # Run the same data samples ('num_samples' many) across all the models
    all_graphs = []
    num_batches_processed = 0
    torch.manual_seed(seed=seed)
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

    # Remove the hooks (as this was interfering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()

    return postprocess_activations(args, all_graphs, activations)

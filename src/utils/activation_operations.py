import copy

import torch
import dgl

from utils.layer_operations import get_layer_type, LayerType


def get_activation(activation, name):
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


def preprocess_activations(graphs, activations):
    """Postprocess graph layer activations to their graph form."""
    print(activations)
    preprocessed_activations = {}
    # Iterate over models
    for key, activation in activations.items():
        model_preprocessed_activations = {}
        # Iterate over layers
        for lnum, (layer_name, layer_activations) in enumerate(activation.items()):
            if get_layer_type(layer_name) in [LayerType.embedding, LayerType.mlp]:
                model_preprocessed_activations[layer_name] = torch.cat(layer_activations, dim=0).T
            elif get_layer_type(layer_name) == LayerType.gcn:

                graph_layer_activations = [[] for _ in range(layer_activations[0].shape[1])]
                # Iterate over batch graphs
                for batch_activations, batch_graph in zip(layer_activations, graphs):

                    # Iterate over nodes
                    for node_idx in range(batch_activations.shape[1]):
                        g = dgl.DGLGraph(batch_graph.edges())
                        g.ndata['Feature'] = batch_activations[:, node_idx]
                        graph_layer_activations[node_idx].append(g)

                model_preprocessed_activations[layer_name] = graph_layer_activations
            else:
                model_preprocessed_activations[layer_name] = layer_activations
        preprocessed_activations[key] = model_preprocessed_activations
    return preprocessed_activations

def compute_activations(args, model, train_loader):
    '''
    Helper function to understand what how activations are constructed. Not used while fusing.

    :param model: takes in a pretrained model
    :param train_loader: the particular train loader
    :param num_samples: # of randomly selected training examples to average the activations over

    :return:  list of len: num_layers and each of them is a particular tensor of activations
    '''

    activation = {}
    num_batches_processed = 0

    model.train()

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

def compute_selective_activation(args, models, train_loader):
    torch.manual_seed(args.activation_seed)

    # Prepare all the models
    activations = {}
    forward_hooks = []

    for idx, model in enumerate(models):

        # Initialize the activation dictionary for each model
        activations[idx] = {}
        layer_hooks = []
        # Set forward hooks for all layers inside a model
        for name, layer in model.named_modules():
            if name == '' or get_layer_type(name) == 'dropout':
                print("layer excluded")
            else:
                layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
                print("set forward hook for layer named: ", name)

        forward_hooks.append(layer_hooks)
        # Set the model in train mode
        model.train()

    # Run the same data samples ('num_samples' many) across all the models
    all_graphs = []
    num_batches_processed = 0
    with torch.no_grad():
        for batch_idx, (batch_graphs, _) in enumerate(train_loader):
            print('graph batch', batch_graphs)
            print(type(batch_graphs))
            all_graphs.append(batch_graphs)
            if num_batches_processed == args.num_batches:
                break
            for idx, model in enumerate(models):
                model_forward(args, model, batch_graphs)
            num_batches_processed += 1

    # Dump the activations for all models onto disk
    if args.dump_activations and args.dump_activations_path is not None:
        pass
        #TODO: Implement saving activations
        # for idx in range(len(models)):
        #     save_activations(idx, activations[idx], dump_path)

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()

    return preprocess_activations(all_graphs, activations)

import torch


def compute_activations(args, model, train_loader):
    '''

    This method can be called from another python module. Example usage demonstrated here.
    Averages the activations across the 'num_samples' many inputs.

    :param model: takes in a pretrained model
    :param train_loader: the particular train loader
    :param num_samples: # of randomly selected training examples to average the activations over

    :return:  list of len: num_layers and each of them is a particular tensor of activations
    '''

    activation = {}
    num_samples_processed = 0

    # Define forward hook that averages the activations
    # over number of samples processed
    def get_activation(name):
        def hook(model, input, output):
            print("num of samples seen before", num_samples_processed)
            # print("output is ", output.detach())
            print(name, output.detach().shape)
            if name not in activation:
                activation[name] = [output.detach()]
            else:
# ]                activation[name] = (num_samples_processed * activation[name] + output.detach()) / (
#                             num_samples_processed + 1)
                activation[name].append(output.detach())
                print("now at layer {}: {}".format(name, activation[name]))

        return hook

    model.train()

    # Set forward hooks for all the layers
    for name, layer in model.named_modules():
        if name == '':
            print("excluded")
            continue
        layer.register_forward_hook(get_activation(name))
        print("set forward hook for layer named: ", name)

    # Run over the samples in training set
    # datapoints= []
    with torch.no_grad():
        for batch_idx, (batch_graphs, batch_targets) in enumerate(train_loader):
            # datapoints.append(data)
            batch_graphs = batch_graphs.to(args.device)
            batch_x = batch_graphs.ndata['feat'].to(args.device)
            batch_e = batch_graphs.edata['feat'].to(args.device)
            print("nodes data", batch_x.shape)
            print("edge data", batch_e)
            batch_targets = batch_targets.to(args.device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(args.device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            num_samples_processed += 1
            if num_samples_processed == args.num_batches:
                break
    return activation, None  # , datapoints


# def save_activations(idx, activation, dump_path):
#     myutils.mkdir(dump_path)
#     myutils.pickle_obj(
#         activation,
#         os.path.join(dump_path, 'model_{}_activations'.format(idx))
#     )
#
#
# def compute_activations_across_models(args, models, train_loader, num_samples, dump_activations=False, dump_path=None):
#     # hook that computes the mean activations across data samples
#     def get_activation(activation, name):
#         def hook(model, input, output):
#             # print("num of samples seen before", num_samples_processed)
#             # print("output is ", output.detach())
#             if name not in activation:
#                 activation[name] = [output.detach()]
#                 # activation[name] = output.detach()
#             else:
#                 # activation[name] = (num_samples_processed * activation[name] + output.detach()) / (
#                 #         num_samples_processed + 1)
#                 activation[name].append(output.detach())
#
#             # print("now at layer {}: {}".format(name, activation[name]))
#
#         return hook
#
#     # Prepare all the models
#     activations = {}
#
#     for idx, model in enumerate(models):
#
#         # Initialize the activation dictionary for each model
#         activations[idx] = {}
#
#         # Set forward hooks for all layers inside a model
#         for name, layer in model.named_modules():
#             if name == '':
#                 print("excluded")
#                 continue
#             layer.register_forward_hook(get_activation(activations[idx], name))
#             print("set forward hook for layer named: ", name)
#
#         # Set the model in train mode
#         model.train()
#
#     # Run the same data samples ('num_samples' many) across all the models
#     num_samples_processed = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if args.gpu_id != -1:
#             data = data.cuda(args.gpu_id)
#         for idx, model in enumerate(models):
#             model(data)
#         num_samples_processed += 1
#         if num_samples_processed == num_samples:
#             break
#
#     # Dump the activations for all models onto disk
#     if dump_activations and dump_path is not None:
#         for idx in range(len(models)):
#             save_activations(idx, activations[idx], dump_path)
#
#     # print("these will be returned", activations)
#     return activations
#
#
# def normalize_tensor(tens):
#     tens_shape = tens.shape
#     assert tens_shape[1] == 1
#     tens = tens.view(tens_shape[0], 1, -1)
#     norms = tens.norm(dim=-1)
#     ntens = tens / norms.view(-1, 1, 1)
#     ntens = ntens.view(tens_shape)
#     return ntens
#
# def compute_selective_activation(args, models, layer_name, train_loader, num_samples, dump_activations=False,
#                                  dump_path=None):
#     torch.manual_seed(args.activation_seed)
#
#     # hook that computes the mean activations across data samples
#     def get_activation(activation, name):
#         def hook(model, input, output):
#             # print("num of samples seen before", num_samples_processed)
#             # print("output is ", output.detach())
#             if name not in activation:
#                 activation[name] = []
#
#             activation[name].append(output.detach())
#
#         return hook
#
#     # Prepare all the models
#     activations = {}
#     forward_hooks = []
#
#     assert args.disable_bias
#     # handle below for bias later on!
#     # print("list of model named params ", list(models[0].named_parameters()))
#     param_names = [tupl[0].replace('.weight', '') for tupl in models[0].named_parameters()]
#
#     for idx, model in enumerate(models):
#
#         # Initialize the activation dictionary for each model
#         activations[idx] = {}
#         layer_hooks = []
#         # Set forward hooks for all layers inside a model
#         for name, layer in model.named_modules():
#             if name == '':
#                 print("excluded")
#             elif args.dataset != 'mnist' and name not in param_names:
#                 print("this was continued, ", name)
#             # elif name!= layer_name:
#             #     print("this layer was not needed, ", name)
#             else:
#                 layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
#                 print("set forward hook for layer named: ", name)
#
#         forward_hooks.append(layer_hooks)
#         # Set the model in train mode
#         model.train()
#
#     # Run the same data samples ('num_samples' many) across all the models
#     num_samples_processed = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if num_samples_processed == num_samples:
#             break
#         if args.gpu_id != -1:
#             data = data.cuda(args.gpu_id)
#         for idx, model in enumerate(models):
#             model(data)
#         num_samples_processed += 1
#
#     relu = torch.nn.ReLU()
#     for idx in range(len(models)):
#         for lnum, layer in enumerate(activations[idx]):
#             print('***********')
#             activations[idx][layer] = torch.stack(activations[idx][layer])
#             print("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
#                                                              torch.max(activations[idx][layer]),
#                                                              torch.mean(activations[idx][layer])))
#             # assert (activations[idx][layer] >= 0).all()
#
#             if not args.prelu_acts and not lnum == (len(activations[idx]) - 1):
#                 # print("activation was ", activations[idx][layer])
#                 print("applying relu ---------------")
#                 activations[idx][layer] = relu(activations[idx][layer])
#                 # print("activation now ", activations[idx][layer])
#                 print("after RELU: min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
#                                                                              torch.max(activations[idx][layer]),
#                                                                              torch.mean(activations[idx][layer])))
#             if args.standardize_acts:
#                 mean_acts = activations[idx][layer].mean(dim=0)
#                 std_acts = activations[idx][layer].std(dim=0)
#                 print("shape of mean, std, and usual acts are: ", mean_acts.shape, std_acts.shape,
#                       activations[idx][layer].shape)
#                 activations[idx][layer] = (activations[idx][layer] - mean_acts) / (std_acts + 1e-9)
#             elif args.center_acts:
#                 mean_acts = activations[idx][layer].mean(dim=0)
#                 print("shape of mean and usual acts are: ", mean_acts.shape, activations[idx][layer].shape)
#                 activations[idx][layer] = (activations[idx][layer] - mean_acts)
#
#             print("activations for idx {} at layer {} have the following shape ".format(idx, layer),
#                   activations[idx][layer].shape)
#             print('-----------')
#     # Dump the activations for all models onto disk
#     if dump_activations and dump_path is not None:
#         for idx in range(len(models)):
#             save_activations(idx, activations[idx], dump_path)
#
#     # Remove the hooks (as this was intefering with prediction ensembling)
#     for idx in range(len(forward_hooks)):
#         for hook in forward_hooks[idx]:
#             hook.remove()
#
#     # print("selective activations returned are", activations)
#     return activations
#
#
# if __name__ == '__main__':
#
#     args = parameters.get_parameters(options_type='mnist_act', deprecated=True)
#
#     config = vgg_hyperparams.config
#
#     model_list = os.listdir(ensemble_dir)
#     num_models = len(model_list)
#
#     train_loader, test_loader = cifar_train.get_dataset(config)
#
#     # Load models
#     models = []
#     for idx in range(num_models):
#         print("Path is ", ensemble_dir)
#         print("loading model with idx {} and checkpoint_type is {}".format(idx, checkpoint_type))
#         models.append(
#             cifar_train.get_pretrained_model(
#                 config, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, checkpoint_type)),
#                 args.gpu_id
#             )
#         )
#
#     # Compute activations and dump them
#     dump_path = os.path.join(activation_root_dir, ensemble_experiment)
#     activations = compute_activations_across_models(models, train_loader, args.num_samples, dump_activations=True,
#                                                     dump_path=dump_path)

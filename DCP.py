import numpy as np
import torch
from scipy.spatial import distance
from torch import nn


def get_conv_layer_weights(model, layer_name):
    try:
        layer = dict(model.named_modules())[layer_name]
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            return layer
        else:
            raise ValueError(f"The layer named '{layer_name}' is not support.")
    except KeyError:
        raise ValueError(f"No layer named '{layer_name}' found in the model.")

def prune(model, layer_name, norm_rate, prune_rate, mode=0):
    layer = get_conv_layer_weights(model, layer_name)
    weights = get_conv_layer_weights(model, layer_name).weight
    weights_vec = weights.view(weights.size()[0], -1)
    norm_num = int(weights_vec.size()[0] * norm_rate)
    pruned_num = int(weights_vec.size()[1] * prune_rate)

    norm = torch.norm(weights_vec, p=1, dim=1)
    norm_small_index = norm.argsort()[:norm_num]
    weight_vec_after_norm = torch.index_select(weights_vec, 0, norm_small_index).cpu().detach().numpy()
    if mode == 0:
        medians = np.median(weight_vec_after_norm, axis=1)
        distances = np.abs(weight_vec_after_norm - medians[:, np.newaxis])
        prune_indices = np.argsort(distances)[:, :pruned_num]
    elif mode == 1:
        prune_indices = weight_vec_after_norm.argsort()[:, :pruned_num]
    elif mode == 2:
        rng = np.random.default_rng()
        prune_indices = np.array([rng.choice(weight_vec_after_norm.shape[1], pruned_num, replace=False)
                                  for _ in range(weight_vec_after_norm.shape[0])])

    mask = torch.ones_like(weights)
    if len(weights.shape) == 3:
        with torch.no_grad():
            for i, idx in enumerate(prune_indices):
                weights[norm_small_index[i], :, idx] = 0
                mask[norm_small_index[i], :, idx] = 0
    else:
        with torch.no_grad():
            for i, idx in enumerate(prune_indices):
                weights[norm_small_index[i], idx] = 0
                mask[norm_small_index[i], idx] = 0

    def mask_hook(grad):
        return grad * mask

    layer.weight.register_hook(mask_hook)
"""
This scripts contain hooks for the dp-backward process
"""
import torch
import utils.global_var as global_var

SENSITIVITY = 2.


def dummy_hook(module, grad_input, grad_output):
    """
    dummy hook
    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    pass


def modify_gradnorm_hook(module, grad_input, grad_output):
    """
    gradient modification hook
    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    CLIP_BOUND = global_var.get_value('CLIP_BOUND')

    # get grad wrt. input (random walker sequence)
    grad_wrt_input = grad_input[0]
    grad_input_shape = grad_wrt_input.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP

    grad_wrt_input = grad_wrt_input.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_input, p=2, dim=1)

    # clip note, the following operation is actually not consistent with the "Clip" operation in DP-SGD, it is indeed
    # a 'normalizing' step, for more details, one can refer to https://github.com/DingfanChen/GS-WGAN/issues/5
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_input = clip_coef * grad_wrt_input

    grad_input_new = [grad_wrt_input.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


def dp_hook(module, grad_input, grad_output):
    """
    gradient modification + noise hook
    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    CLIP_BOUND = global_var.get_value('CLIP_BOUND')

    # get grad wrt. input
    grad_wrt_input = grad_input[0]
    grad_input_shape = grad_wrt_input.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize

    grad_wrt_input = grad_wrt_input.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_input, p=2, dim=1)

    # clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_input = clip_coef * grad_wrt_input

    noise_multiplier = global_var.get_value('noise_multiplier')
    # add noise
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_input)
    grad_wrt_input = grad_wrt_input + noise
    grad_input_new = [grad_wrt_input.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)

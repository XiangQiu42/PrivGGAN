import torch
import numpy as np
import matplotlib.pyplot as plt


def exp_mov_avg(Gs, G, alpha=0.999, global_step=999):
    """
    Exponential moving average
    :param Gs:
    :param G:
    :param alpha:
    :param global_step:
    :return:
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_device():
    num_gpu = 0
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device(torch.cuda.current_device() if use_cuda else "cpu")
    if use_cuda:
        num_gpu = torch.cuda.device_count()
        # for i in range(num_gpu):
        #     print(f"Device {i}: {torch.cuda.get_device_name(0)}")
    return device, num_gpu


def visualization_scores(model, create_graph_every):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    iterations = create_graph_every * np.arange(len(model.eo))
    axes[0].plot(iterations, model.roc_auc)
    axes[0].set_title('roc auc')
    axes[0].grid()
    axes[1].plot(iterations, model.avp, color='g')
    axes[1].set_title('average precision')
    axes[1].grid()
    axes[2].plot(iterations, model.eo, color='r')
    axes[2].set_title('edge overlap')
    axes[2].grid()
    plt.savefig("score.png")

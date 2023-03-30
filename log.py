import numpy as np
import os
import shutil
import sys
from torch.utils.tensorboard import SummaryWriter
import torch


def model_input(data, device):
    datum = data.data[0:1]
    if isinstance(datum, np.ndarray):
        return torch.from_numpy(datum).float().to(device)
    else:
        return datum.float().to(device)


def get_script():
    py_script = os.path.basename(sys.argv[0])
    return os.path.splitext(py_script)[0]


def get_specified_params(hparams):
    keys = [k.split("=")[0][2:] for k in sys.argv[1:]]
    specified = {k: hparams[k] for k in keys}
    return specified


def make_hparam_str(hparams, exclude):
    return ",".join([f"{key}_{value}"
                     for key, value in sorted(hparams.items())
                     if key not in exclude])


class Logger(object):
    def __init__(self, logdir):

        if logdir is None:
            self.writer = None
        else:
            if os.path.exists(logdir) and os.path.isdir(logdir):
                shutil.rmtree(logdir)

            self.writer = SummaryWriter(log_dir=logdir)

    def log_losses(self, iteration, d_loss, g_loss):
        if self.writer is None:
            return

        self.writer.add_scalars("Loss", {'Critic_Loss': d_loss,
                                         'Generator_loss': g_loss}, iteration)

    def log_scores(self, iteration,  roc_auc, avp, eo, epsilon=None):
        if self.writer is None:
            return

        self.writer.add_scalar("Validation Performance/ROC-AUC", roc_auc, iteration)
        self.writer.add_scalar("Validation Performance/Avg-Pre", avp, iteration)
        self.writer.add_scalar("Validation Performance/Edge Overlap", eo, iteration)

        if epsilon is not None:
            self.writer.add_scalar("Epsilon", epsilon, iteration)

    def log_single_layer_grad(self, name, param, iteration):
        # this function is used to recorde the grad information of the generator
        if self.writer is None:
            return

        self.writer.add_histogram(name + '/grad', param, iteration)

    def log_all_grad(self, params, iteration):
        # this function is used to recorde the grad information of the generator
        if self.writer is None:
            return

        for name, param in params:
            self.writer.add_histogram(name + '/grad', param.grad.clone().cpu().numpy(), iteration)

    def log_model(self, model, input_to_model):
        if self.writer is None:
            return
        self.writer.add_graph(model, input_to_model)

    def log_scalar(self, tag, scalar_value, global_step):
        if self.writer is None or scalar_value is None:
            return
        self.writer.add_scalar(tag, scalar_value, global_step)

    def log_input_distribution(self, input_scatters, epoch):
        if self.writer is None:
            return
        self.writer.add_histogram("Input_distribution", input_scatters, epoch)

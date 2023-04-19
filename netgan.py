"""
Based on NetGAN's Pytorch implementation:  https://github.com/mmiller96/netgan_pytorch
"""
from models import Generator, Discriminator
import utils.graph_utils as utils
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.optim as optim
from torch.nn.functional import one_hot
from torch.autograd import grad
import time

from utils.dp_utils import ORDERS, get_epsilon

from matplotlib import pyplot as plt

from utils.dp_hooks import dp_hook, dummy_hook, modify_gradnorm_hook

global dynamic_hook_function


def master_hook_adder(module, grad_input, grad_output):
    """
    global hook
    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


class NetGAN:
    def __init__(self, walker_gen, N, device, max_iterations=20000, rw_len=16, batch_size=128,
                 H_gen=40, H_disc=30, w_down_g=128, w_down_d=128,
                 z_dim=16, latent_type='normal', lr=0.0003, n_critic=3, gp_weight=10.0, betas=(.5, .9),
                 l2_penalty_disc=5e-5, l2_penalty_gen=1e-7, temp_start=5.0, temp_decay=1 - 5e-5, min_temp=0.5,
                 dp_method=None, pretrain_dir=None, weight_norm=True,
                 sample_rate=0.01, noise_mul=1.0):
        """
            Initialize NetGAN.

            Parameters
            ----------
            walker_gen: object
                   Object of Random Walker that generates a single random walk and takes no arguments.
            N: int
               Number of nodes in the graph to generate.
            max_iterations: int, default: 40,000
                        Maximal iterations if the stopping_criterion is not fulfilled.
            rw_len: int
                    Length of random walks to generate.
            batch_size: int, default: 128
                        The batch size.
            H_gen: int, default: 40
                   The hidden_size of the generator.
            H_disc: int, default: 30
                    The hidden_size of the discriminator
            w_down_g: int, 128
                   The down-projection matrix for the generator
            w_down_d: int, 128
                   The down-projection matrix for the discriminator
            latent_type: str, 'normal'
                         Sample from "normal" distribution or "bernoulli" distributions
            z_dim: int, 16
                   The dimension of the random noise that is used as input to the generator.
            lr: float, default: 0.0003
                    The Learning rate will be used for the generator as well as for the discriminator.
            n_critic: int, default: 3
                      The number of discriminator iterations per generator training iteration.
            gp_weight: float, default: 10
                        Gradient penalty weight for the Wasserstein GAN. See the paper 'Improved Training of Wasserstein GANs' for more details.
            betas: tuple, default: (.5, .9)
                    Decay rates of the Adam Optimizers.
            l2_penalty_gen: float, default: 1e-7
                            L2 penalty on the generator weights.
            l2_penalty_disc: float, default: 5e-5
                             L2 penalty on the discriminator weights.
            temp_start: float, default: 5.0
                        The initial temperature for the Gumbel softmax.
            temp_decay: float, default: 1-5e-5
                        After each evaluation, the current temperature is updated as
                        current_temp := max(temperature_decay*current_temp, min_temperature)
            min_temp: float, default: 0.5
                      The minimal temperature for the Gumbel softmax.
            dp_method: str in ['PrivGGAN', 'DPSGD'], default is None
                Specify the  GAN-based Differential privacy method
            pretrain_dir: default is None
                if it is not None, then we load the pretrain discriminators from this dir
        """
        self.device = device
        self.max_iterations = max_iterations
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.N = N
        self.generator = Generator(H_inputs=w_down_g, H=H_gen, N=N, rw_len=rw_len, z_dim=z_dim,
                                   latent_type=latent_type, temp=temp_start).to(self.device)
        self.discriminator = Discriminator(H_inputs=w_down_d, H=H_disc, N=N, rw_len=rw_len).to(self.device)
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.l2_penalty_disc = l2_penalty_disc
        self.l2_penalty_gen = l2_penalty_gen
        self.temp_start = temp_start
        self.temp_decay = temp_decay
        self.min_temp = min_temp

        self.walker = walker_gen
        self.eo = []
        self.critic_loss = []
        self.generator_loss = []
        self.avp = []
        self.roc_auc = []
        self.best_performance = 0.0
        self.running = True

        # define the variable for valuation
        self.stopping_criterion = None
        self.stopping_eo = None  # needed for 'eo' criterion
        self.max_patience = None  # needed for 'val' criterion
        self.patience = None

        self.weight_norm = weight_norm

        self.dp_method = dp_method

        if self.dp_method == 'DPSGD':
            # used opacus for DPSGD,
            # please Note this part in unfinished now !!!
            from opacus.accountants import RDPAccountant
            from opacus import GradSampleModule
            from opacus.optimizers import DPOptimizer

            # initialize privacy accountant
            self.dp_accountant = RDPAccountant()

            # then wrap the discriminator, note in dp_sgd we only entails one discriminators
            self.dp_dis = GradSampleModule(self.discriminator)

            # wrap optimizer
            self.dp_optimizer = DPOptimizer(
                optimizer=self.D_optimizer,
                noise_multiplier=noise_mul,  # same as make_private arguments
                max_grad_norm=1.0,  # same as make_private arguments
                expected_batch_size=self.batch_size
            )

            # attach accountant to track privacy for an optimizer
            assert sample_rate < 1.
            self.dp_optimizer.attach_step_hook(
                self.dp_accountant.get_optimizer_hook_fn(
                    sample_rate=sample_rate
                )
            )

            print(f"==> Note we will train the model with noise_mul {noise_mul} and sample_rate {sample_rate}")

            # # Note: the current version of DP-SGD + NetGAN may not work since opacus 0.14.0 has problems in
            # # handing multiple backpropagation process.
            #
            # print("==> Attach privacy Engine to make a DP Discriminators...")
            # privacy_engine = PrivacyEngine(
            #     self.discriminator,
            #     sample_rate=batch_size / N,
            #     alphas=ORDERS,
            #     target_epsilon=target_epsilon,
            #     target_delta=target_delta,
            #     max_grad_norm=max_grad_norm,
            #     epochs=max_iterations,
            # )
            # privacy_engine.attach(self.D_optimizer)

        elif self.dp_method == 'PrivGGAN':
            # # Register hooks
            global dynamic_hook_function
            # self.discriminator.W_down.register_backward_hook(master_hook_adder)  # used for pytorch < 1.8
            self.discriminator.W_down.register_full_backward_hook(master_hook_adder)

            if pretrain_dir is not None:
                print("\n=> Train the Discriminators from pretrained models.\n")
                # network_path = os.path.join(pretrain_dir, 'netD_%d' % netD_id, 'netD.pth')
                network_path = os.path.join(pretrain_dir, 'netD.pth')
                self.discriminator.load_state_dict(torch.load(network_path))
            else:
                print("\n=> Train the Discriminators from scratch.\n")

    def l2_regularization_G(self, G):
        # regularizaation for the generator. W_down will not be regularized.
        l2_1 = torch.sum(torch.cat([x.view(-1) for x in G.W_down.weight]) ** 2 / 2)
        l2_2 = torch.sum(torch.cat([x.view(-1) for x in G.W_up.weight]) ** 2 / 2)
        l2_3 = torch.sum(torch.cat([x.view(-1) for x in G.W_up.bias]) ** 2 / 2)
        l2_4 = torch.sum(torch.cat([x.view(-1) for x in G.intermediate.weight]) ** 2 / 2)
        l2_5 = torch.sum(torch.cat([x.view(-1) for x in G.intermediate.bias]) ** 2 / 2)
        l2_6 = torch.sum(torch.cat([x.view(-1) for x in G.h_up.weight]) ** 2 / 2)
        l2_7 = torch.sum(torch.cat([x.view(-1) for x in G.h_up.bias]) ** 2 / 2)
        l2_8 = torch.sum(torch.cat([x.view(-1) for x in G.c_up.weight]) ** 2 / 2)
        l2_9 = torch.sum(torch.cat([x.view(-1) for x in G.c_up.bias]) ** 2 / 2)
        l2_10 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.cell.weight]) ** 2 / 2)
        l2_11 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.cell.bias]) ** 2 / 2)
        l2 = self.l2_penalty_gen * (l2_1 + l2_2 + l2_3 + l2_4 + l2_5 + l2_6 + l2_7 + l2_8 + l2_9 + l2_10 + l2_11)
        return l2

    def l2_regularization_D(self, D):
        # regularizaation for the discriminator. W_down will not be regularized.
        l2_1 = torch.sum(torch.cat([x.view(-1) for x in D.W_down.weight]) ** 2 / 2)
        l2_2 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.cell.weight]) ** 2 / 2)
        l2_3 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.cell.bias]) ** 2 / 2)
        l2_4 = torch.sum(torch.cat([x.view(-1) for x in D.lin_out.weight]) ** 2 / 2)
        l2_5 = torch.sum(torch.cat([x.view(-1) for x in D.lin_out.bias]) ** 2 / 2)
        l2 = self.l2_penalty_disc * (l2_1 + l2_2 + l2_3 + l2_4 + l2_5)
        return l2

    def calc_gp(self, fake_inputs, real_inputs):
        # Calculate the gradient penalty.
        # For more details see the paper 'Improved Training of Wasserstein-GANs'.
        alpha = torch.rand((self.batch_size, 1, 1), dtype=torch.float64).to(self.device)
        differences = fake_inputs - real_inputs
        interpolates = real_inputs + alpha * differences

        y_pred_interpolates = self.discriminator(interpolates)
        gradients = grad(outputs=y_pred_interpolates,
                         inputs=interpolates,
                         grad_outputs=torch.ones_like(y_pred_interpolates),
                         create_graph=True,
                         retain_graph=True)[0]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
        gradient_penalty = torch.mean((slopes - 1) ** 2)

        gradient_penalty = gradient_penalty * self.gp_weight
        return gradient_penalty

    def critic_train_iteration(self):
        # Update Discriminator

        for p in self.discriminator.parameters():
            p.requires_grad = True

        self.D_optimizer.zero_grad(set_to_none=True)

        # create fake and real inputs
        fake_inputs = self.generator.sample(self.batch_size, self.device)
        real_inputs = one_hot(torch.tensor(next(self.walker.walk())), num_classes=self.N). \
            type(torch.float64).to(self.device)

        if self.dp_method == 'PrivGGAN':
            global dynamic_hook_function
            dynamic_hook_function = dummy_hook

        y_pred_fake = self.discriminator(fake_inputs)
        y_pred_real = self.discriminator(real_inputs)

        d_real = torch.mean(y_pred_real)
        d_fake = torch.mean(y_pred_fake)

        if self.dp_method == 'DPSGD':
            "For more details why we abandon the gradient penalty in opacus, one can refer the following link:"
            "https://discuss.pytorch.org/t/indexerror-pop-from-empty-list-in-grad-sample-module-py-for-opacus-version-0-9/128843/11"
            disc_cost = d_fake - d_real
        else:
            gp = self.calc_gp(fake_inputs, real_inputs)  # gradient penalty
            disc_cost = d_fake - d_real + gp

        # control the use of weight norm regularization
        if self.weight_norm:
            disc_cost += self.l2_regularization_D(self.discriminator)

        disc_cost.backward()
        self.D_optimizer.step()

        # if self.dp_method == 'DPSGD':
        #     for p in self.discriminator.parameters():
        #         if hasattr(p, "grad_sample"):
        #             del p.grad_sample

        return disc_cost.item()

    def generator_train_iteration(self):
        # Update Generator

        global dynamic_hook_function
        if self.dp_method == "PrivGGAN":
            # Sanitize the gradients passed to the Generator
            dynamic_hook_function = dp_hook
        elif self.dp_method == "DP_SGD":
            # Note, since in the DPSGD mode, we do not add gradient penalty. Hence, to enforce
            # Lipschitz continuity, we need to modify the gradient norm, without adding noise
            dynamic_hook_function = modify_gradnorm_hook
        else:
            dynamic_hook_function = dummy_hook

        for p in self.discriminator.parameters():
            p.requires_grad = False
        self.generator.train()
        self.G_optimizer.zero_grad()

        fake_inputs = self.generator.sample(self.batch_size, self.device)

        y_pred_fake = self.discriminator(fake_inputs)
        gen_cost = -torch.mean(y_pred_fake)

        # control the use of weight norm regularization
        if self.weight_norm:
            gen_cost += self.l2_regularization_G(self.generator)

        gen_cost.backward()
        self.G_optimizer.step()
        return gen_cost.item()

    def create_graph(self, A_orig, val_ones, val_zeros, num_samples, i, num_pre_iter=10000, reset_weights=False):
        if reset_weights:
            self.generator.reset_weights()
        self.generator.eval()

        self.generator.temp = 0.5
        samples = []
        num_iterations = int(num_samples / num_pre_iter) + 1
        for j in range(num_iterations):
            samples.append(self.generator.sample_discrete(num_pre_iter, 'cuda'))
        samples = np.vstack(samples)
        gr = utils.score_matrix_from_random_walks(samples, self.N)
        gr = gr.tocsr()

        # Assemble a graph from the score matrix
        _graph = utils.graph_from_scores(gr, A_orig.sum())
        # Compute edge overlap
        edge_overlap = utils.edge_overlap(A_orig.toarray(), _graph)
        edge_scores = np.append(gr[tuple(val_ones.T)].A1, gr[tuple(val_zeros.T)].A1)
        actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))
        # Compute Validation ROC-AUC and average precision scores.
        self.roc_auc.append(roc_auc_score(actual_labels_val, edge_scores))
        self.avp.append(average_precision_score(actual_labels_val, edge_scores))
        self.eo.append(edge_overlap / A_orig.sum())

        print(
            '**** Iter {}:  ROC: {:.4f}, \t AVP: {:.4f}, \t EO: {:.4f} ****'.format(i, self.roc_auc[-1],
                                                                                    self.avp[-1],
                                                                                    self.eo[-1])
        )

    def check_running(self, i, target_epsilon=None, sample_rate=None, noise_mul=None):
        if self.dp_method and noise_mul:

            # epsilon = self.dp_accountant.get_epsilon(delta=1e-5)

            cur_eps = get_epsilon(sample_rate, mul=noise_mul, num_steps=i)
            if cur_eps > target_epsilon:
                print('Run out of epsilon, finished training after {} iterations'.format(i))
                self.running = False

        if self.stopping_criterion == 'val':
            if self.roc_auc[-1] + self.avp[-1] > self.best_performance:
                self.best_performance = self.roc_auc[-1] + self.avp[-1]
                self.patience = self.max_patience
            else:
                self.patience -= 1
            if self.patience == 0:
                print('finished after {} iterations'.format(i))
                self.running = False

        elif self.stopping_criterion == 'eo':
            if self.stopping_eo < self.eo[-1]:
                print('finished after {} iterations'.format(i))
                self.running = False

    def initialize_validation_settings(self, stopping_criterion, stopping_eo, max_patience):
        self.stopping_criterion = stopping_criterion
        self.stopping_eo = stopping_eo  # needed for 'eo' criterion
        self.max_patience = max_patience  # needed for 'val' criterion
        self.patience = max_patience

        if self.stopping_criterion == 'val':
            print("**** Using VAL criterion for early stopping with max patience of: {} ****".format(self.max_patience))
        elif self.stopping_criterion == 'eo':
            assert self.stopping_eo is not None, "stopping_eo is not a float"
            print("**** Using EO criterion of {} for early stopping ****".format(self.stopping_eo))

    def plot_graph(self, save_dir=None):
        if len(self.critic_loss) > 10:
            plt.plot(self.critic_loss[9::], label="Critic loss")
            plt.plot(self.generator_loss[9::], label="Generator loss")
        else:
            plt.plot(self.critic_loss, label="Critic loss")
            plt.plot(self.generator_loss, label="Generator loss")
        plt.legend()
        f_name = os.path.join(save_dir, 'ret_pretrain.jpg')
        plt.savefig(f_name)

    def train(self, A_orig, val_ones, val_zeros, create_graph_every=2000, plot_graph_every=2000,
              num_samples_graph=100000, stopping_criterion=None, max_patience=5, stopping_eo=0.5,
              logger=None, save_dir=None, update_temp_every=500,
              target_epsilon=None, sample_rate=None, noise_mul=None):
        """
        A_orig: sparse matrix, shape: (N,N)
                Adjacency matrix of the orig graph to be trained on.
        val_ones: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation edges
        val_zeros: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation non-edges
        create_graph_every: int, default: 2000
                            Creates every nth iteration a graph from random walks.
        plot_graph_every: int, default: 2000
                         Plots the lost functions of the generator and discriminator.
        num_samples_graph: int, default 10000
                            Number of random walks that will be created for the graphs.
                            Higher values mean more precise evaluations but also more computational time.
        stopping_criterion: str, default: None
                            The stopping_criterion can be either 'val' or 'eo' or None:
                            None:  Early stopping is not used
                            'val': Stops the optimization if there are no improvements after several iterations. --> defined by max_patience
                            'eo': Stops if the edge overlap exceeds a certain threshold. --> defined by stopping_eo
                            Note if dp_method is not None, we will stop the training if current epsilon > target_epsilon

        max_patience: int, default: 5
                      Maximum evaluation steps without improvement of the validation accuracy to tolerate. Only
                      applies to the VAL criterion.
        stopping_eo: float in (0,1], default: 0.5
                     Stops when the edge overlap exceeds this threshold. Will be used when stopping_criterion is 'eo'.
        logger: object of class 'Logger'
                log the experiment details.
        save_dir: default is None
                Used when in pretrain mode to saving the log
        update_temp_every: int, default is 500
                the number of iterations to update Gumbel temperature
        """
        self.initialize_validation_settings(stopping_criterion, stopping_eo, max_patience)
        starting_time = time.time()
        # Start Training
        for i in range(self.max_iterations):
            if self.running:

                self.critic_loss.append(np.mean([self.critic_train_iteration() for _ in range(self.n_critic)]))
                self.generator_loss.append(self.generator_train_iteration())

                # # update the exponential moving average
                # exp_mov_avg(netGS, self.generator, alpha=0.999, global_step=iter)

                if i % 50 == 1:
                # if i % 50 == 1:
                    print(
                        'Iteration: {}\t D_loss: {:.6f}\t G_loss: {:.6f}'.format(i, self.critic_loss[-1],
                                                                                 self.generator_loss[-1]))
                if logger:
                    logger.log_losses(i, self.critic_loss[-1], self.generator_loss[-1])

                    params = list(self.generator.named_parameters())
                    logger.log_all_grad(params, i)

                # create the graph for evaluation (hold for val_share != 0)
                if val_ones and i % create_graph_every == create_graph_every - 1:
                    ss = time.time()
                    self.create_graph(A_orig, val_ones, val_zeros, num_samples_graph, i, num_pre_iter=1000)
                    print(f"Took {((time.time() - ss) / 60)} for one graph generation")

                    if logger:
                        logger.log_scores(i, self.roc_auc[-1], self.avp[-1], self.eo[-1])

                    self.check_running(i)
                    print('Took {} minutes so far..'.format((time.time() - starting_time) / 60))

                # Update Gumbel temperature
                if i % update_temp_every == 0:
                    self.generator.temp = np.maximum(self.temp_start * np.exp(-(1 - self.temp_decay) * i),
                                                     self.min_temp)

                if logger is None:
                    if plot_graph_every > 0 and (i + 1) % plot_graph_every == 0:
                        self.plot_graph(save_dir=save_dir)

                self.check_running(i, target_epsilon, sample_rate, noise_mul)

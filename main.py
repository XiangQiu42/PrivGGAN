import argparse
import os
import torch
import pickle
import numpy as np
import random
import scipy.sparse as sp
import networkx as nx
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt

from netgan import NetGAN
from log import Logger
from config import load_config
from dataloader import Single_Graph_Dataset
import utils.graph_utils as g_utils
from utils.utils import get_device
import utils.global_var as global_var
from utils.dp_utils import get_noise_mul


def arg_parser():
    # init the common args, expect the model specific args
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default="cora_ml", choices=['cora_ml', 'dblp', 'citeseer'])
    parser.add_argument('--val_share', type=float, default=0.)
    parser.add_argument('--test_share', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--num_dis', type=int, default=1, help="the number of discriminators")
    parser.add_argument('--gpu_id', type=str, default=None)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--latent_type', type=str, default='normal', choices=['bernoulli', 'normal'])

    parser.add_argument('--stopping_criterion', type=str, default=None, choices=['val', 'eo'])

    # Settings for adding differential privacy
    parser.add_argument('--dp_method', default="DPSGD", choices=["GS_WGAN", "DPSGD"])
    parser.add_argument('--weight_norm', type=bool, default=False)
    parser.add_argument('--target_epsilon', type=float, default=3.0, help='privacy parameter epsilon')
    parser.add_argument('--target_delta', type=float, default=1e-5, help='desired delta')

    parser.add_argument('--noise_multiplier', type=float, default=None, help='noise scale')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        metavar="C",
                        help="Clip per-sample gradients to this norm (default 1.0)")

    args = parser.parse_args()
    return args


def save_data(args, data_list, root_path):
    if 'DP' in args.model_name:
        path = root_path + '/data/generated/{}_{}_eps:{}.pkl'.format(
            args.model_name, args.dataset_str, args.model_args.eps_requirement
        )
    else:
        path = root_path + '/data/generated/{}_{}.pkl'.format(args.model_name, args.dataset_str)

    with open(path, 'wb') as a:
        pickle.dump(data_list, a)


def main(args, m_args, sub_graph: Single_Graph_Dataset, sub_graph_index=0, save_model=False):
    """
    Parameters
    ----------
    args:
        the common args, see arg_parser for more details
    m_args:
        the specific args for different datasets, see set_env for more details
    sub_graph: Single_Graph_Dataset
        the input sub-graph of the orig datasets.
        for cora-ml dataset, it actually have only one graph
    sub_graph_index: int
        if the inputs graphs is a multi-graph datasets, specify the index, default is 0
    save_model: bool
        indicating whether to sava the training model, default is False

    Returns
    -------
    sampled_graph:
        The generated graph(i.e. the adjacency matrix)
    statistics_sample: dict
        The graph statistics of the generated graph.
    """

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device, _ = get_device()

    # log the experiments
    # specify the experiment name
    runs_name = '{}_{}_Ndis{}_Zdim{}_BS{}_Lr{}_RwLen{}_WN{}_DP{}_Noise{}'.format(
        args.dataset_name,
        sub_graph_index,
        args.num_dis,
        args.z_dim,
        m_args.batch_size,
        m_args.lr,
        m_args.rw_len,
        args.weight_norm,
        args.dp_method,
        m_args.noise_mul,
    )

    log_dir = os.path.join("runs", args.dataset_name, runs_name)
    logger = Logger(log_dir)

    _A_obs = sub_graph.adj_lcc
    _N = sub_graph.N

    # Define the global variable
    global_var._init()
    global_var.set_value('CLIP_BOUND', args.max_grad_norm)
    global_var.set_value('noise_multiplier', m_args.noise_mul)

    if args.dataset_name in ['cora_ml', 'dblp', 'citeseer']:
        print("\n==> Separate the edges into train, test, validation, "
              f"with val_share={args.val_share}, test_share={args.test_share}")

        val_ones = val_zeros = test_ones = test_zeros = None
        if args.test_share != 0:
            train_ones, val_ones, val_zeros, test_ones, test_zeros = g_utils.train_val_test_split_adjacency(
                _A_obs, args.val_share,
                args.test_share,
                args.random_seed,
                undirected=True,
                connected=True,
                asserts=True
            )
            train_graph = sp.coo_matrix((np.ones(len(train_ones)), (train_ones[:, 0], train_ones[:, 1]))).tocsr()
        else:
            train_graph = _A_obs
    else:
        # note for the 'IMDB_MULTI' or 'DBLP2' datasets,
        # we only wanna train a model for further graph classification
        # Note, this part is Not completely finished since is has problems in adapting NetGAN in small graphs
        train_graph = _A_obs
        val_ones = val_zeros = test_ones = test_zeros = None

    assert (train_graph.toarray() == train_graph.toarray().T).all()

    walker = g_utils.RandomWalker(train_graph, m_args.rw_len, p=1, q=1, batch_size=m_args.batch_size)

    # Actually, we can have a visualization of an example of Random walk
    # print(walker.walk().__next__())

    pretrain_dir = None
    if os.path.exists(os.path.join("./results", args.dataset_name, str(sub_graph_index), 'pretrain')):
        pretrain_dir = os.path.join("./results", args.dataset_name, str(sub_graph_index), 'pretrain')

    # Create our NetGAN model
    netgan = NetGAN(
        walker, _N, device, m_args.max_iterations, m_args.rw_len, m_args.batch_size,
        z_dim=args.z_dim,
        lr=m_args.lr,
        n_critic=5,
        latent_type=args.latent_type,
        dp_method=args.dp_method,
        weight_norm=args.weight_norm,
        pretrain_dir=pretrain_dir,
        sample_rate=m_args.batch_size / m_args.num_samples,
        noise_mul=m_args.noise_mul
    )
    netgan.train(
        A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros,
        create_graph_every=10000,
        num_samples_graph=m_args.num_samples,
        stopping_criterion=args.stopping_criterion, max_patience=5, stopping_eo=0.5,
        logger=logger,
        update_temp_every=500,
        sample_rate=m_args.batch_size / m_args.num_samples,
        target_epsilon=args.target_epsilon,
        noise_mul=m_args.noise_mul
    )

    # Get the score matrix
    scores_matrix_generated = g_utils.create_score_matrix(netgan, m_args.num_samples)

    if test_ones is not None and args.dataset_name in ['cora_ml', 'dblp', 'citeseer']:
        test_labels = np.concatenate((np.ones(len(test_ones)), np.zeros(len(test_zeros))))
        test_scores = np.concatenate((
            scores_matrix_generated[tuple(test_ones.T)].A1,
            scores_matrix_generated[tuple(test_zeros.T)].A1)
        )
        print(f"\n==> Evaluating generalization via link prediction for {sub_graph_index} on {args.dataset_name}")
        print(f"ROC: {roc_auc_score(test_labels, test_scores)}")
        print(f"AVP: {average_precision_score(test_labels, test_scores)}")

    save_dir = os.path.join("./results", args.dataset_name, str(sub_graph_index), 'main')
    os.makedirs(save_dir, exist_ok=True)

    # Compute graph statistics
    A_select = train_graph
    # Assemble a graph from the score matrix
    sampled_graph = g_utils.graph_from_scores(scores_matrix_generated, A_select.sum())

    # one can visualize the graph as follows:
    plt.spy(train_graph, markersize=.2)
    plt.savefig(os.path.join(save_dir, 'train_graph.jpg'))
    plt.spy(sampled_graph, markersize=.2)
    plt.savefig(os.path.join(save_dir, 'sample_graph.jpg'))

    eo = g_utils.edge_overlap(A_select.toarray(), sampled_graph) / A_select.sum()
    print("\n==> The edge overlap: ", eo)

    if save_model:
        os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)
        torch.save(netgan, os.path.join(save_dir, "model", f'NetGAN_DP_{m_args.max_iterations}.pt'))

    return sampled_graph


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    common_args = arg_parser()
    model_args = load_config(common_args.dataset_name)
    model_args.select_noise_mul(common_args.target_epsilon)

    # Random seed
    random.seed(common_args.random_seed)
    np.random.seed(common_args.random_seed)
    torch.manual_seed(common_args.random_seed)
    torch.set_num_threads(4)

    if common_args.dp_method is not None:
        print(
            f"==> We choose to use {common_args.dp_method} to make a differential private GAN on {common_args.dataset_name}.")
    else:
        print(
            f"==> Train the model on {common_args.dataset_name} without DP method.")

    orig_graph_list = []
    generated_graph_list = []
    # property_dict_all = []
    if common_args.dataset_name in ['cora_ml', 'dblp', 'citeseer']:
        g = Single_Graph_Dataset(common_args.dataset_name)

        if common_args.dp_method is not None and common_args.noise_multiplier is not None:
            print(f"We use the specific noise scale({common_args.noise_multiplier}) to override the default setting...")
            model_args.noise_mul = common_args.noise_multiplier

        # # for a given target epsilon and delta, we compute the noise-multiplier based on post-process theory
        # if common_args.dp_method == "GS_WGAN" and model_args.noise_mul is None:
        #     assert common_args.target_epsilon
        #     delta = 1e-5
        #     sigma = get_noise_mul(g.N, model_args.batch_size, common_args.target_epsilon,
        #                           model_args.max_iterations, target_delta=delta)
        #     print(f'Where eps,delta,gamma = ({common_args.target_epsilon},{delta},{model_args.batch_size / g.N}) '
        #           f'==> Noise level sigma=', sigma)
        #     model_args.noise_mul = sigma

        # generated_adj, graph_stats = main(args=common_args, m_args=model_args, sub_graph=g, sub_graph_index=0)
        generated_adj = main(args=common_args, m_args=model_args, sub_graph=g, sub_graph_index=0)
        G = nx.from_numpy_array(generated_adj)
        generated_graph_list.append(G)
        # property_dict_all.append(graph_stats)
    else:
        print("Unknown dataset...")

    # Save the generated graphs in 'data/generated'...
    os.makedirs("./data/generated", exist_ok=True)
    if common_args.dp_method is not None:
        save_path = './data/generated/{}_{}_{}_{}.pkl'.format(common_args.dp_method, common_args.dataset_name,
                                                              common_args.target_epsilon, common_args.weight_norm)
    else:
        save_path = './data/generated/NetGAN_{}_{}.pkl'.format(common_args.dataset_name, common_args.weight_norm)

    with open(save_path, 'wb') as file:
        pickle.dump(generated_graph_list, file)

import argparse
import os
import torch
import numpy as np
import random
import scipy.sparse as sp

from netgan import NetGAN
from config import load_config
from dataloader import Multi_Graph_Dataset, Single_Graph_Dataset
import utils.graph_utils as g_utils
from utils.utils import get_device
import utils.global_var as global_var


def arg_parser():
    # init the common args, expect the model specific args
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default="cora_ml", choices=['cora_ml', 'dblp', 'citeseer',
                        'new_dblp2', 'new_IMDB_MULTI', 'relabeled_dblp2'])
    parser.add_argument('--val_share', type=float, default=0.1)
    parser.add_argument('--test_share', type=float, default=0.05)
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--num_dis', type=int, default=1, help="the number of discriminators")
    parser.add_argument('--gpu_id', type=str, default=None)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--latent_type', type=str, default='normal', choices=['bernoulli', 'normal'])

    parser.add_argument('--lr', type=float, default=0.0005)

    parser.add_argument('--stopping_criterion', type=str, default=None, choices=['val', 'eo'])

    args = parser.parse_args()
    return args


def main(args, m_args, sub_graph: Single_Graph_Dataset, sub_graph_index=0):
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

    Returns
    -------
    sampled_graph:
        The generated graph(i.e. the adjacency matrix)
    statistics_sample: dict
        The graph statistics of the generated graph.
    """

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device, num_gpu = get_device()

    # Define the global variable
    global_var._init()
    global_var.set_value('CLIP_BOUND', 1.)
    global_var.set_value('noise_multiplier', 1.07)

    _A_obs = sub_graph.adj_lcc
    _N = sub_graph.N

    if 'cora' in args.dataset_name:
        # Separate the edges into train, test, validation
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
        # note for the IMDB_MULTI or 'DBLP' datasets, we only wanna train a model for further graph classification
        train_graph = _A_obs
        val_ones = None
        val_zeros = None
    assert (train_graph.toarray() == train_graph.toarray().T).all()

    # Compute the graph statistics of the origin graph
    # print(g_utils.compute_graph_statistics(train_graph.toarray()))

    save_dir = os.path.join("./results", args.dataset_name, str(sub_graph_index), 'pretrain')
    os.makedirs(save_dir, exist_ok=True)

    walker = g_utils.RandomWalker(train_graph, m_args.rw_len, p=1, q=1, batch_size=m_args.batch_size)

    # Create our NetGAN model
    netgan = NetGAN(
        walker, _N, device, m_args.max_iterations, m_args.rw_len, m_args.batch_size,
        z_dim=args.z_dim,
        lr=args.lr,
        latent_type=args.latent_type,
    )
    netgan.train(
        A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros,
        create_graph_every=m_args.max_iterations,
        num_samples_graph=500000,
        stopping_criterion=args.stopping_criterion, max_patience=5, stopping_eo=0.5,
        save_dir=save_dir
    )

    print(f"Finished pretraining on {sub_graph_index} and saving the Discriminators in the {save_dir}")
    torch.save(netgan.discriminator.state_dict(), os.path.join(save_dir, 'netD.pth'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    common_args = arg_parser()
    model_args = load_config(common_args.dataset_name, pretrain=True)

    print(f"Pretrain on {common_args.dataset_name} with {model_args.max_iterations} to get the Discriminators.")

    # Random seed
    random.seed(common_args.random_seed)
    np.random.seed(common_args.random_seed)
    torch.manual_seed(common_args.random_seed)
    torch.set_num_threads(4)

    if common_args.dataset_name in ['cora_ml', 'dblp', 'citeseer']:
        g = Single_Graph_Dataset(common_args.dataset_name)
        main(args=common_args, m_args=model_args, sub_graph=g, sub_graph_index=0)
    else:
        # load multi-graph datasets
        datasets = Multi_Graph_Dataset(common_args.dataset_name)
        for counter, graph in enumerate(datasets.graphs):

            if graph.N < model_args.rw_len:
                print(f"Discard graph {counter} with {graph.N} nodes")
                continue

            main(common_args, model_args, graph, sub_graph_index=counter)




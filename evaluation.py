# This file try to eval the quality of the generated graph, which include the single graph stat and
# the MMD distance between the graph properties

import argparse
import numpy as np
import os
import networkx as nx

from utils.graph_utils import load_graphs, load_ground_truth, compute_graph_statistics
import evaluation.stats as stats

dir_path = os.path.dirname(os.path.realpath(__file__))


def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def compute_basic_stats(real_g_list, target_g_list):
    dist_degree = stats.degree_stats(real_g_list, target_g_list)
    dist_clustering = stats.clustering_stats(real_g_list, target_g_list)
    return dist_degree, dist_clustering


def eval_single_list(graphs, graphs_orig):
    """
    Evaluate a list of graphs by comparing with graphs in directory dir_input.

    Parameters
    ----------
    graphs:
        The generated graph list
    graphs_orig:
        the truth graph list

    Returns
    -------

    """

    # graph_test_len = len(graph_test)
    # graph_test = graph_test[int(0.8 * graph_test_len):]  # test on a hold out test set
    mmd_degree = stats.degree_stats(graphs_orig, graphs)
    print('deg: ', mmd_degree)
    mmd_clustering = stats.clustering_stats(graphs_orig, graphs)
    print('clustering: ', mmd_clustering)

    # try:
    #     mmd_4orbits = stats.orbit_stats_all(graphs_orig, graphs)
    # except:
    #     mmd_4orbits = -1
    # print('orbits: ', mmd_4orbits)

    mmd_spec = stats.spectral_stats(graphs_orig, graphs)
    print('spec: ', mmd_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation arguments.')
    parser.add_argument('--dataset', type=str, default='citeseer', help='[cora_ml, dblp, citeseer]')
    parser.add_argument('--model', default='NetGAN', choices=['NetGAN', 'PrivGGAN', 'DPGGAN', 'GGAN'])
    parser.add_argument('--weight_norm', default=False, type=bool)
    parser.add_argument('--target_epsilon', default='3.0', type=float)
    _args = parser.parse_args()

    input_dir = dir_path + '/data/orig/'
    orig_graph = load_ground_truth(input_dir, _args.dataset)
    orig_graph = [nx.from_scipy_sparse_matrix(orig_graph)]
    generated_graph = load_graphs(dir_path + f'/data/generated/{_args.model}_{_args.dataset}_{_args.weight_norm}.pkl')
    # gen erated_graph = load_graphs(dir_path + f'/data/generated/{_args.model}_{_args.dataset}_'
    #                                          f'{_args.target_epsilon}_{_args.weight_norm}.pkl')

    print(f"==> Start analyse the Single graph stats: {_args.model}_{_args.dataset}\n")
    stat_orig = compute_graph_statistics(nx.to_numpy_array(orig_graph[0]))
    print("Origin Stat:", stat_orig)
    stat_gen = compute_graph_statistics(nx.to_numpy_array(generated_graph[0]))
    print("Generated Stat:", stat_gen)

    print("\n==>MMD:")
    eval_single_list(generated_graph, orig_graph)

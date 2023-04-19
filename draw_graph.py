import os
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from utils.graph_utils import load_graphs, load_ground_truth

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_degree_distributions(graph):
    orig_degrees = graph.degree()
    orig_degrees = dict(orig_degrees)
    _values = sorted(set(orig_degrees.values()))
    _hist = [list(orig_degrees.values()).count(x) for x in _values]
    return _values, _hist


def plot(orig_graph, gen_graph, gen_eps):
    # Plot the subgraphs and degree distributions
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    for index, _name in enumerate(datasets_name):
        # Compute degree distributions
        orig_values, orig_hist = get_degree_distributions(orig_graph[index])
        gen_values, gen_hist = get_degree_distributions(gen_graph[index])
        eps_value, eps_hist = get_degree_distributions(gen_eps[index])

        # Degree distributions
        axs[index].loglog(orig_values, orig_hist, marker='x', color='orangered', linestyle='None',
                          label='original graph')
        axs[index].loglog(gen_values, gen_hist, marker='^', color='steelblue', linestyle='None',
                          label=r'ours $(\varepsilon=3$)')
        axs[index].loglog(eps_value, eps_hist, marker='o', color='goldenrod', linestyle='None',
                          label=r'ours $(\varepsilon=0.2$)')
        axs[index].set_xlabel('Degree', fontsize=16)
        axs[index].set_ylabel('Number of nodes', fontsize=16)
        axs[index].set_title(_name.title(), fontsize=16)
        axs[index].legend(loc='upper right', fontsize=13)

    # Adjust layout and display plot
    fig.tight_layout()
    plt.savefig(f'D:/Github/SEU-master-thesis/figures/degree_distribution.pdf', pad_inches=0.0)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation arguments.')
    parser.add_argument('--dataset', type=str, default='citeseer', help='[cora_ml, dblp, citeseer]')
    parser.add_argument('--model', default='NetGAN', choices=['PrivGGAN', 'PrivGGAN', 'DPGGAN', 'GGAN'])
    parser.add_argument('--weight_norm', default=False, type=bool)
    parser.add_argument('--target_epsilon', default='0.2', type=float)
    _args = parser.parse_args()

    input_dir = dir_path + '/data/orig/'
    original_graph = []
    generated_graph = []
    gen_eps3 = []

    datasets_name = ['cora_ml', 'citeseer', 'dblp']
    for name in datasets_name:
        _graph_orig = load_ground_truth(input_dir, name)
        original_graph.append(nx.from_scipy_sparse_array(_graph_orig))
        # _temp = load_graphs(dir_path + f'/data/generated/{_args.model}_{name}_{_args.weight_norm}.pkl')
        _temp = load_graphs(dir_path + f'/data/generated/PrivGGAN_{name}_3.0_{_args.weight_norm}.pkl')
        generated_graph.append(_temp[0])

        _temp = load_graphs(dir_path + f'/data/generated/PrivGGAN_{name}_0.2_{_args.weight_norm}.pkl')
        gen_eps3.append(_temp[0])

    plot(original_graph, generated_graph, gen_eps3)

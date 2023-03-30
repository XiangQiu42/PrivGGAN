import sys
import networkx as nx
import pickle

from collections import defaultdict

import utils.graph_utils as g_utils


multi_graph_dataset = {'relabeled_dblp2', 'new_dblp2', 'dblp2', 'new_IMDB_MULTI', 'IMDB_MULTI', 'Resampled_IMDB_MULTI'}


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def graph_to_adj_list(adj):
    # Sparse adj matrix to adj lists
    G = nx.from_scipy_sparse_matrix(adj)
    adj_lists = defaultdict(set)

    # Check isolated node before training
    for node, adjacencies in enumerate(G.adjacency()):
        if len(list(adjacencies[1].keys())) == 0:
            print("Node %d is isolated !!!" % node)
            assert False
        adj_lists[node] = set(list(adjacencies[1].keys()))

    return adj_lists


class Single_Graph_Dataset:
    def __init__(self, dataset_str,  graph_adj=None, label=None):
        self.dataset_str = dataset_str
        self.graph_adj = graph_adj
        self.label = label

        self.adj_lcc = None
        self.N = None

        self.load_data()

    def load_data(self):
        if self.dataset_str in ['cora_ml', 'dblp', 'citeseer', 'pubmed']:
            _A_obs, _X_obs, _z_obs = g_utils.load_npz(f'data/orig/{self.dataset_str}.npz')

        elif self.dataset_str in multi_graph_dataset:
            _A_obs = self.graph_adj

        else:
            print("dataset: {} is an unknown Single_Graph_Dataset.".format(self.dataset_str))
            sys.exit(1)

        # For all datasets, We only consider the largest connected components in the origin graph
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1
        lcc = g_utils.largest_connected_components(_A_obs)
        _A_obs = _A_obs[lcc, :][:, lcc]
        _N = _A_obs.shape[0]

        self.adj_lcc = _A_obs
        self.N = _N


# TODO: add graph label
class Multi_Graph_Dataset:
    def __init__(self, dataset_str):
        self.dataset_str = dataset_str
        self.graphs = None

        self.load_data()

    def load_data(self):
        graph_size_list = []
        graph_edge_list = []
        dataset_list = []
        with open('./data/orig/' + self.dataset_str + '.pkl', 'rb') as tf:
            graph_set = pickle.load(tf)
        for graph in graph_set:
            label = graph.graph['label']
            graph_edge_list.append(graph.size())
            graph_size_list.append(graph.number_of_nodes())
            sp_adj_matrix = nx.to_scipy_sparse_matrix(graph)
            dataset_list.append(
                Single_Graph_Dataset(self.dataset_str, graph_adj=sp_adj_matrix, label=label)
            )

        self.graphs = dataset_list

        avg_node = sum(graph_size_list)/len(graph_size_list)
        avg_edge = sum(graph_edge_list)/len(graph_edge_list)
        print(f"There are {len(graph_edge_list)} in the dataset, the average node is {avg_node},"
              f"and the average edge is {avg_edge}")


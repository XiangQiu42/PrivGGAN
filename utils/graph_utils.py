import networkx as nx
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import warnings
import pandas as pd
import igraph
import powerlaw
import pickle
from numba import jit


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        # pdb.set_trace()
        # loader = dict(loader)['arr_0'].item()
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    adj : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graphploy where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep

    ]
    # print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix

    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """
    Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False

    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges

    """

    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]

    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert connected_components(A)[0] == 1  # make sure orig graph is connected
            A_hold = minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = np.array(list(nx.maximal_matching(nx.DiGraph(A))))
                not_in_cover = np.array(list(set(range(N)).difference(hold_edges.flatten())))

                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                min_size = hold_edges.shape[0] + len(not_in_cover)
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))

                d_nic = d[not_in_cover]

                hold_edges_d1 = np.column_stack((not_in_cover[d_nic > 0],
                                                 np.row_stack(map(np.random.choice,
                                                                  A[not_in_cover[d_nic > 0]].tolil().rows))))

                if np.any(d_nic == 0):
                    hold_edges_d0 = np.column_stack(
                        (np.row_stack(map(np.random.choice, A[:, not_in_cover[d_nic == 0]].T.tolil().rows)),
                         not_in_cover[d_nic == 0]))
                    hold_edges = np.row_stack((hold_edges, hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = np.row_stack((hold_edges, hold_edges_d1))

            else:
                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        test_zeros = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        test_zeros = np.row_stack(test_zeros)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back orig A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    if p_val == 0:
        val_ones = None
        val_zeros = None

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def score_matrix_from_random_walks(random_walks, N, symmetric=True):
    """
    Compute the transition scores, i.e. how often a transition occurs, for all node pairs from
    the random walks provided.
    Parameters
    ----------
    random_walks: np.array of shape (n_walks, rw_len, N)
                  The input random walks to count the transitions in.
    N: int
       The number of nodes
    symmetric: bool, default: True
               Whether to symmetrize the resulting scores matrix.

    Returns
    -------
    scores_matrix: sparse matrix, shape (N, N)
                   Matrix whose entries (i,j) correspond to the number of times a transition from node i to j was
                   observed in the input random walks.

    """
    random_walks = np.array(random_walks)
    bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
    bigrams = np.transpose(bigrams, [0, 2, 1])
    bigrams = bigrams.reshape([-1, 2])
    if symmetric:
        bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))

    mat = sp.coo_matrix((np.ones(bigrams.shape[0]), (bigrams[:, 0], bigrams[:, 1])),
                        shape=[N, N])
    return mat


@jit(nopython=True)
def random_walk(edges, node_ixs, rwlen, p=1, q=1, n_walks=1):
    N = len(node_ixs)

    walk = []
    prev_nbs = None
    for w in range(n_walks):
        source_node = np.random.choice(N)
        walk.append(source_node)
        for it in range(rwlen - 1):

            if walk[-1] == N - 1:
                nbs = edges[node_ixs[walk[-1]]::, 1]
            else:
                nbs = edges[node_ixs[walk[-1]]:node_ixs[walk[-1] + 1], 1]

            if it == 0:
                walk.append(np.random.choice(nbs))
                prev_nbs = set(nbs)
                continue

            is_dist_1 = []
            for n in nbs:
                is_dist_1.append(int(n in set(prev_nbs)))

            is_dist_1_np = np.array(is_dist_1)
            is_dist_0 = nbs == walk[-2]
            is_dist_2 = 1 - is_dist_1_np - is_dist_0

            alpha_pq = is_dist_0 / p + is_dist_1_np + is_dist_2 / q
            alpha_pq_norm = alpha_pq / np.sum(alpha_pq)
            rdm_num = np.random.rand()
            cumsum = np.cumsum(alpha_pq_norm)
            nxt = nbs[np.sum(1 - (cumsum > rdm_num))]
            walk.append(nxt)
            prev_nbs = set(nbs)
    return np.array(walk)


class RandomWalker:
    """
    Helper class to generate random walks on the input adjacency matrix.
    """

    def __init__(self, adj, rw_len, p=1, q=1, batch_size=128):
        self.adj = adj
        # if not "lil" in str(type(adj)):
        #    warnings.warn("Input adjacency matrix not in lil format. Converting it to lil.")
        #    self.adj = self.adj.tolil()

        self.rw_len = rw_len
        self.p = p
        self.q = q
        self.edges = np.array(self.adj.nonzero()).T
        self.node_ixs = np.unique(self.edges[:, 0], return_index=True)[1]
        self.batch_size = batch_size

    def walk(self):
        while True:
            yield random_walk(self.edges, self.node_ixs, self.rw_len, self.p, self.q, self.batch_size).reshape(
                [-1, self.rw_len])


def edge_overlap(A, B):
    """
    Compute edge overlap between input graphs A and B, i.e. how many edges in A are also present in graph B. Assumes
    that both graphs contain the same number of edges.

    Parameters
    ----------
    A: sparse matrix or np.array of shape (N,N).
       First input adjacency matrix.
    B: sparse matrix or np.array of shape (N,N).
       Second input adjacency matrix.

    Returns
    -------
    float, the edge overlap.
    """

    return ((A == B) & (A == 1)).sum()


def graph_from_scores(scores, n_edges):
    """
    Assemble a symmetric binary graph from the input score matrix. Ensures that there will be no singleton nodes.
    See the paper for details.

    Parameters
    ----------
    scores: np.array of shape (N,N)
            The input transition scores.
    n_edges: int
             The desired number of edges in the target graph.

    Returns
    -------
    target_g: symmettic binary sparse matrix of shape (N,N)
              The assembled graph.

    """

    if len(scores.nonzero()[0]) < n_edges:
        return symmetric(scores) > 0
    target_g = np.zeros(scores.shape)  # initialize target graph
    scores_int = scores.toarray().copy()  # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero
    degrees_int = scores_int.sum(0)  # The row sum over the scores.

    N = scores.shape[0]

    for n in np.random.choice(N, replace=False, size=N):  # Iterate the nodes in random order

        row = scores_int[n, :].copy()
        if row.sum() == 0:
            continue

        probs = row / row.sum()

        target = np.random.choice(N, p=probs)
        target_g[n, target] = 1
        target_g[target, n] = 1

    diff = np.round((n_edges - target_g.sum()) / 2)
    if diff > 0:
        triu = np.triu(scores_int)
        triu[target_g > 0] = 0
        triu[np.diag_indices_from(scores_int)] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_int)
        extra_edges = np.random.choice(triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff))

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    target_g = symmetric(target_g)
    return target_g


def symmetric(directed_adjacency, clip_to_one=True):
    """
    Symmetrize the input adjacency matrix.
    Parameters
    ----------
    directed_adjacency: sparse matrix or np.array of shape (N,N)
                        Input adjacency matrix.
    clip_to_one: bool, default: True
                 Whether the output should be binarized (i.e. clipped to 1)

    Returns
    -------
    A_symmetric: sparse matrix or np.array of the same shape as the input
                 Symmetrized adjacency matrix.

    """

    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric


def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0)
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
            n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0)
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees + .0001) / (2 * float(m))))
    return H_er


def statistics_cluster_props(A, Z_obs):
    def get_blocks(A_in, Z_obs, normalize=True):
        block = Z_obs.T.dot(A_in.dot(Z_obs))
        counts = np.sum(Z_obs, axis=0)
        blocks_outer = counts[:, None].dot(counts[None, :])
        if normalize:
            blocks_outer = np.multiply(block, 1 / blocks_outer)
        return blocks_outer

    in_blocks = get_blocks(A, Z_obs)
    diag_mean = np.multiply(in_blocks, np.eye(in_blocks.shape[0])).mean()
    offdiag_mean = np.multiply(in_blocks, 1 - np.eye(in_blocks.shape[0])).mean()
    return diag_mean, offdiag_mean


def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def symmetrize_and_without_self_loop(adj_orig):
    def symmetrize(a):
        # print("symmetrize A!")
        a = a + a.T
        sum_a = a
        sum_a = a - np.diag(a.diagonal())
        sum_a[sum_a >= 1] = 1
        # sum_a[sum_a < 1] = 0
        return sum_a

    # input must be np.array not sparse matrix
    if sp.issparse(adj_orig):
        adj_ = adj_orig.todense()
    else:
        adj_ = adj_orig

    # remove self_loop
    np.fill_diagonal(adj_, 0)
    adj_orig = symmetrize(adj_)

    G = nx.from_numpy_array(adj_)
    G.remove_nodes_from(list(nx.isolates(G)))
    adj = nx.to_numpy_array(G)
    return adj


def compute_graph_statistics(A_in, Z_obs=None):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Z_obs: np.matrix [N, K], where K is the number of classes.
          Matrix whose rows are one-hot vectors indicating the class membership of the respective node.

    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """

    A = A_in.copy()

    # important restriction
    A = symmetrize_and_without_self_loop(A)
    # assert ((A == A.T).all())
    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean

    # largest connected component
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]
    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)

    # claw count
    statistics['claw_count'] = statistics_claw_count(A)

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)

    # Square count
    statistics['square_count'] = statistics_square_count(A)

    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)

    # Assortativity
    statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)

    # Clustering coefficient
    if statistics['claw_count'] != 0:
        statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / statistics['claw_count']
    else:
        statistics['clustering_coefficient'] = None

    # Number of connected components
    statistics['n_components'] = connected_components(A)[0]

    if Z_obs is not None:
        # inter- and intra-community density
        intra, inter = statistics_cluster_props(A, Z_obs)
        statistics['intra_community_density'] = intra
        statistics['inter_community_density'] = inter

    statistics['cpl'] = statistics_compute_cpl(A)

    return statistics


def get_graph(path_folder, tag):
    # data = pd.read_csv(path, index_col=0)
    # data = data[data['Network'] == tag]
    # keys = np.unique(data[['fbus', 'tbus']].values)
    # dicto = dict(zip(keys, np.arange(len(keys))))
    # data = data[['fbus', 'tbus']].values.reshape(-1)
    # data = np.array([dicto[key] for key in data])
    # data = data.reshape(-1, 2)
    # edges = [list(edge) for edge in data]
    # nodes = np.sort(np.unique(data))
    # G = nx.Graph()
    # G.add_nodes_from(nodes)
    # G.add_edges_from(edges)
    # G = nx.to_scipy_sparse_matrix(G)

    path_branch = path_folder + r'\branch.csv'
    path_bus = path_folder + r'\bus.csv'
    data_branch = pd.read_csv(path_branch)
    data_bus = pd.read_csv(path_bus)

    data_branch = data_branch[data_branch['Network'] == tag]
    data_bus = data_bus[data_bus['Network'] == tag]

    data_branch = data_branch.drop_duplicates(subset=['fbus', 'tbus']).sort_values(by=['fbus', 'tbus'])

    edges = data_branch[['fbus', 'tbus']].values
    nodes = np.arange(data_bus.shape[0])

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G = nx.to_scipy_sparse_matrix(G)

    return G


def create_score_matrix(trainer, num_samples, num_pre_iter=1000):
    # generate Random walkers from the trained model
    trainer.generator.eval()

    trainer.generator.temp = 0.5
    samples = []
    num_iterations = int(num_samples / num_pre_iter) + 1
    print(f'create graph with {num_pre_iter}/{num_samples}:')
    for j in range(num_iterations):
        # if j % 10 == 1:
            # print('Iter {}/{}...'.format(j, num_iterations))
        rw_samples = trainer.generator.sample_discrete(num_pre_iter, 'cuda')
        samples.append(rw_samples)

    samples = np.vstack(samples)
    gr = score_matrix_from_random_walks(samples, trainer.N)
    gr = gr.tocsr()

    return gr


def load_ground_truth(file_path, dataset_name, select_max_component=True, val_share=0., test_share=0.0, seed=0):
    """
    Read ground truth graphs.
    Parameters
    ----------
    dataset_name:
        the name of the dataset
    file_path:
        the file path of the original graph
    select_max_component:
        weather to select the largest connected component to do the measurement. default is True
    val_share: float
        default is 0.0
    test_share: float
        default is 0.0
    seed: int
       the manual seed, default is 0

    Returns
    -------

    """

    try:
        _A_obs, _, _ = load_npz(file_path + dataset_name)
    except:
        print(f'{file_path} Not found!')
        return None

    if select_max_component:
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1
        lcc = largest_connected_components(_A_obs)
        _A_obs = _A_obs[lcc, :][:, lcc]

    if test_share > 0:
        train_ones, _, _, _, _ = train_val_test_split_adjacency(
            _A_obs, val_share,
            test_share,
            seed,
            undirected=True,
            connected=True,
            asserts=True
        )
        orig_graph = sp.coo_matrix((np.ones(len(train_ones)), (train_ones[:, 0], train_ones[:, 1]))).tocsr()
    else:
        orig_graph = _A_obs

    return orig_graph


def load_graphs(file_path, select_max_component=False):
    with open(file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    if select_max_component:
        print("Select the largest connected components in the original graph")
        results = []
        # We only consider the largest connected components in the graph
        for graph in data:
            _A_obs = nx.to_scipy_sparse_matrix(graph)

            _A_obs = _A_obs + _A_obs.T
            _A_obs[_A_obs > 1] = 1
            lcc = largest_connected_components(_A_obs)
            _A_obs = _A_obs[lcc, :][:, lcc]

            g = nx.from_scipy_sparse_matrix(_A_obs)

            results.append(g)

        return results
    else:
        return data



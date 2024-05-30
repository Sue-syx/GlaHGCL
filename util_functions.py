from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
from copy import deepcopy
import multiprocessing as mp
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import warnings
import logging

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class MyDataset(InMemoryDataset):
    def __init__(self, data_list, root, transform=None, pre_transform=None):
        self.data_list = data_list
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del self.data_list


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, links, labels, h, u_features, v_features, max_node_label, class_values):
        super(MyDynamicDataset, self).__init__(root)
        self.A = A
        self.links = links
        self.labels = labels
        self.h = h
        self.u_features = u_features
        self.v_features = v_features
        self.max_node_label = max_node_label
        self.class_values = class_values

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]

        g, n_labels, n_features = subgraph_extraction_labeling((i, j), self.A, self.h, self.u_features, self.v_features,
                                                               self.class_values)
        g_label = self.labels[idx]
        return nx_to_PyGGraph(g, g_label, n_labels, n_features, self.max_node_label, self.class_values)


def nx_to_PyGGraph(g, graph_label, node_labels, node_features, max_node_label, class_values):
    y = torch.Tensor([graph_label])

    if len(g.edges()) == 0:
        i, j = [], []
    else:
        i, j = zip(*g.edges())

    edge_index = torch.LongTensor([i + j, j + i])

    x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
    x1 = torch.FloatTensor(node_features)
    x = torch.cat([x, x1], 1)

    data = Data(x, edge_index, y=y)
    return data


def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in
                  enumerate(edges)}  # transform r back to rating label
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g


def subgraph_extraction_labeling(ind, A, h=1, u_features=None, v_features=None, class_values=None):
    dist = 0
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    for dist in range(1, h + 1):
        v_fringe, u_fringe = neighbors(u_fringe, A, True), neighbors(v_fringe, A, False)
        if u_fringe == {0}:
            u_fringe = set()
        if v_fringe == {0}:
            v_fringe = set()
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)

        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = A[u_nodes, :][:, v_nodes]

    subgraph[0, 0] = 0

    g = nx.Graph()
    g.add_nodes_from(range(len(u_nodes)), bipartite='u')
    g.add_nodes_from(range(len(u_nodes), len(u_nodes) + len(v_nodes)), bipartite='v')

    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)

    r = r.astype(int)
    v += len(u_nodes)

    g.add_edges_from(zip(u, v))

    # get structural node labels
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]

    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[v_nodes]

    if u_features is not None and v_features is not None:
        node_features = np.concatenate([u_features, v_features], 0)
        g.node_features = np.concatenate([u_features, v_features], 0)
    else:
        node_features = None

    # if True:
    #     # directly use padded node features
    #     if u_features is not None and v_features is not None:
    #         node_features = np.concatenate([u_features, v_features], 0)
    #         g.node_features = np.concatenate([u_features, v_features], 0)

    # if False:
    #     # use identity features (one-hot encodings of node idxes)
    #     u_ids = one_hot(u_nodes, A.shape[0] + A.shape[1])
    #     v_ids = one_hot([x + A.shape[0] for x in v_nodes], A.shape[0] + A.shape[1])
    #     node_ids = np.concatenate([u_ids, v_ids], 0)
    #     node_features = node_ids
    # if False:
    #     # only output node features for the target user and item
    #     if u_features is not None and v_features is not None:
    #         node_features = [u_features[0], v_features[0]]

    return g, node_labels, node_features


def parallel_worker(g_label, ind, A, h=1, u_features=None, v_features=None, class_values=None):
    g, node_labels, node_features = subgraph_extraction_labeling(ind, A, h, u_features, v_features, class_values)
    return g_label, g, node_labels, node_features, ind


def neighbors(fringe, A, row=True):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        if row:
            _, nei, _ = ssp.find(A[node, :])
        else:
            _, nei, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])

    x[np.arange(len(idx)), idx] = 1.0
    return x


def extracting_subgraphs(
        A,
        all_indices,
        all_labels,
        h=1,
        u_features=None,
        v_features=None,
        max_node_label=None):
    # extract enclosing subgraphs
    if max_node_label is None:  # if not provided, infer from graphs
        max_n_label = {'max_node_label': 0}
    class_values = np.array([0, 1], dtype=float)

    g_list = []
    start = time.time()
    pool = mp.Pool(4)
    ind = [[], []]

    results = pool.starmap_async(parallel_worker,
                                 [(g_label, (i, j), A, h, u_features, v_features, class_values) for i, j, g_label in
                                  zip(all_indices[0], all_indices[1], all_labels)])

    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready():
            break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()

    pool.close()

    pbar.close()
    end = time.time()
    for i in range(len(results)):
        ind[0].append(results[i][4][0])
        ind[1].append(results[i][4][1])

    g_list += [nx_to_PyGGraph(g, g_label, n_labels, n_features, max_node_label, class_values) for
               g_label, g, n_labels, n_features, ind in tqdm(results)]

    del results
    end2 = time.time()

    return g_list

    # graphs = helper(A, all_indices, all_labels)
    # return graphs


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

"""Data utils functions for pre-processing and data loading."""
import os
import os.path
import pickle as pkl
import pickle
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset # node classification datasets
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            # check if file exist cause it takes a long time to perform mask_edges function
            if args.dataset in ['pubmed', 'cs', 'amazonphoto', 'amazoncomputer']:
                file_path = f"data/lp_preprocess/{args.dataset}/lp_links.pkl"
                if os.path.isfile(file_path):
                    # LOAD DATA HERE
                    with open(file_path, 'rb') as f:
                        data_dict = pickle.load(f)
                        adj_train = data_dict['adj_train']
                        train_edges = data_dict['train_edges']
                        train_edges_false = data_dict['train_edges_false']
                        val_edges = data_dict['val_edges']
                        val_edges_false = data_dict['val_edges_false']
                        test_edges = data_dict['test_edges']
                        test_edges_false = data_dict['test_edges_false']
                else:
                    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                        adj, args.val_prop, args.test_prop, args.split_seed
                    )
                    data_dict = {
                        'adj_train':adj_train,
                        'train_edges':train_edges,
                        'train_edges_false':train_edges_false,
                        'val_edges':val_edges,
                        'val_edges_false':val_edges_false,
                        'test_edges':test_edges,
                        'test_edges_false':test_edges_false,
                    }
                    with open(file_path, 'wb') as f:
                        pickle.dump(data_dict, f)
            else:
                adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                        adj, args.val_prop, args.test_prop, args.split_seed)

            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
        data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
        test_edges_false)


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def load_data_dgl(dataset):
     # load and preprocess dataset
    if dataset == 'cora':
        data = CoraGraphDataset()
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif dataset == 'amazonphoto':
        data = AmazonCoBuyPhotoDataset()
    elif dataset == 'amazoncomputer':
        data = AmazonCoBuyComputerDataset() 
    elif dataset == 'cs':
        data = CoauthorCSDataset()
    elif dataset =='physics':
        data = CoauthorPhysicsDataset()
    
    g = data[0]
    g = dgl.transform.add_self_loop(dgl.transform.remove_self_loop(g))
    features = g.ndata['feat']
    labels = g.ndata['label']
    adj = g.adjacency_matrix(scipy_fmt="csr")
    labels = labels.numpy()
    return adj, features, labels


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features = load_new_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset in ['amazoncomputer', 'amazonphoto', 'cs', 'physics']:
        adj, features, _ = load_data_dgl(dataset)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        # semi-supervised settings
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
        val_prop, test_prop = 0.20, 0.20 # 60% training, 20% test, 20% valid
        # uncomment for 60/20/20 split
        # idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
    else:
        if dataset == 'disease_nc':
            val_prop, test_prop = 0.10, 0.60
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset in ['amazoncomputer', 'amazonphoto', 'cs', 'physics']:
            val_prop, test_prop = 0.20, 0.20 # 60% training, 20% test, 20% valid
            adj, features, labels = load_data_dgl(dataset)
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val,
            'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    print('graph', type(graph), graph.number_of_nodes(), graph.number_of_edges())
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


def load_new_data(dataset_str, use_feats, data_path):
    adj = sp.load_npz(os.path.join(data_path, "{}.edges.npz".format(dataset_str)))
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    return adj, features, labels


def remove_isolated_nodes(adj):
    G = nx.Graph(adj)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    connected_subgraphs = list(nx.connected_component_subgraphs(G))
    connected_G = connected_subgraphs[0]
    return connected_G.nodes, connected_G.adj

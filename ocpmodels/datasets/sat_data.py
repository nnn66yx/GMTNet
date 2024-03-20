from torch_sparse import SparseTensor
from torch_scatter import scatter
from ocpmodels.datasets.lmdb_dataset import TrajectoryLmdbDataset
import pickle
import time
import torch
import torch_geometric.utils as utils
import torch.nn as nn
import lmdb
from torch_scatter import scatter_add

pos_enc_dim = 768

def compute_pe(data):
    W0 = normalize_adj(data.edge_index, num_nodes=len(data.atomic_numbers)).tocsc()
    W = W0
    vector = torch.zeros((data.num_nodes, pos_enc_dim))
    vector[:, 0] = torch.from_numpy(W0.diagonal())
    for i in range(pos_enc_dim - 1):
        W = W.dot(W0)
        vector[:, i + 1] = torch.from_numpy(W.diagonal())
    return vector.float()

def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)


if __name__ == "__main__":
    # 读取lmdb文件
    dataset = TrajectoryLmdbDataset({"src": "/home/a113/nyx/Graphormer/examples/oc20/data_test_5/train/data.lmdb"})
    l = len(dataset)

    db1 = lmdb.open(
        "/home/a113/nyx/Graphormer/examples/oc20/data_test_5/train/data_sat.lmdb",
        map_size=10000000000,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    m = 0  # 索引

    for i in range(0, l):
        data = dataset[i]
        n = len(data.atomic_numbers)
        s = torch.arange(n)
        data.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        data.degree = None
        data.degree = 1. / torch.sqrt(1. + utils.degree(data.edge_index[0], data.num_nodes))

        data.abs_pe = compute_pe(data)

        # add subgraphs and relevant meta data
        txn = db1.begin(write=True)
        txn.put(f"{m}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
        m = m + 1
    # txn = db1.begin(write=True)
    # txn.put("length".encode("ascii"), pickle.dumps(m, protocol=-1))
    # txn.commit()


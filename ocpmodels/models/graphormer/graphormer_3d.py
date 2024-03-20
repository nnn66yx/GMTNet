# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_cluster import radius_graph
from torch_scatter import scatter
from torch_geometric.nn.models.schnet import GaussianSmearing
import torch_geometric.nn as pygnn

from e3nn import o3

from ocpmodels.models.base import BaseModel
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .gaussian_rbf import GaussianRadialBasisLayer
from .graph_attention_transformer import (
    DepthwiseTensorProduct, EdgeDegreeEmbeddingNetwork, ScaledScatter
)
import numpy as np


def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
    ) -> Tensor:
        query = query.unsqueeze(1)
        n_node, _, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, _ * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.view(n_node, -1)
        attn = self.out_proj(attn)
        return attn


class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor = None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x

def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(0).unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        x = x.view(-1, x.shape[1], self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class RBF(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))
        self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.temps, 0.1, 10)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: Tensor, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs = softmax_dropout(
            attn.view(-1, n_node, n_node) + attn_bias, 0.1, self.training
        ).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force


@registry.register_model("graphormer")
class Graphormer3D(BaseModel):
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = False,
        embed_dim: int = 768,
        ffn_embed_dim: int = 768,
        attention_heads: int = 48,
        blocks: int = 4,
        layers: int = 12,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        num_kernel: int = 128,
    ):
        super().__init__()
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = 50

        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.num_kernel = 5
        self.layers = layers
        self.blocks = blocks
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.num_kernel = num_kernel

        self.atom_types = 64
        self.edge_types = 64 * 64
        self.atom_encoder = nn.Embedding(
            self.atom_types, self.embed_dim, padding_idx=0
        )
        self.tag_encoder = nn.Embedding(3, self.embed_dim)
        self.input_dropout = self.dropout
        self.layer = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    self.embed_dim,
                    self.ffn_embed_dim,
                    num_attention_heads=self.attention_heads,
                    dropout=self.dropout,
                    attention_dropout=self.attention_dropout,
                    activation_dropout=self.activation_dropout,
                )
                for _ in range(self.layers)
            ]
        )
        # # mpnn
        # aggregators = ['mean', 'max', 'sum']
        # scalers = ['identity']
        # deg = torch.from_numpy(np.array([0, 4107, 11719, 7095, 300]))
        # self.local_model = pygnn.PNAConv(self.embed_dim, self.embed_dim,
        #                                  aggregators=aggregators,
        #                                  scalers=scalers,
        #                                  deg=deg,
        #                                  edge_dim=min(self.embed_dim, self.embed_dim),
        #                                  towers=1,
        #                                  pre_layers=1,
        #                                  post_layers=1,
        #                                  divide_input=False)

        self.final_ln: Callable[[Tensor], Tensor] = nn.LayerNorm(self.embed_dim)

        # self.engergy_proj: Callable[[Tensor], Tensor] = NonLinear(
        #     self.embed_dim, 1
        # )
        # self.energe_agg_factor: Callable[[Tensor], Tensor] = nn.Embedding(3, 1)
        # nn.init.normal_(self.energe_agg_factor.weight, 0, 0.01)

        K = self.num_kernel

        self.gbf: Callable[[Tensor, Tensor], Tensor] = GaussianLayer(K, self.edge_types)
        # self.bias_proj: Callable[[Tensor], Tensor] = NonLinear(
        #     K, self.attention_heads
        # )
        self.edge_proj: Callable[[Tensor], Tensor] = nn.Linear(K, self.embed_dim)
        # self.node_proc: Callable[[Tensor, Tensor, Tensor], Tensor] = NodeTaskHead(
        #     self.embed_dim, self.attention_heads
        # )
        # equiformer
        irreps_node_embedding = '256x0e+128x1e'
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        irreps_sh = '1x0e+1x1e'
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        # OC20
        _MAX_ATOM_TYPE = 84
        _NUM_TAGS = 3  # 0: sub-surface, 1: surface, 2: adsorbate
        # self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        # self.tag_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _NUM_TAGS)
        # self.rbf = GaussianRadialBasisLayer(128, cutoff=5.0)
        self.fc_neurons = [128, 64, 64]
        _AVG_DEGREE = 36.60622024536133
        # self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding,
        #                                                  self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        # energy
        self.out_energy = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            ScaledSiLU(),
            nn.Linear(self.embed_dim // 2, 1),
        )
        # self.distance_expansion = GaussianSmearing(0.0, 6.0, 768)

    # equiformer
    def _forward_otf_graph(self, data):
        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.max_radius, self.max_neighbors
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors
            return data
        else:
            return data

    def _forward_use_pbc(self, data):
        pos = data.pos
        batch = data.batch
        if self.use_pbc:
            out = get_pbc_distances(pos,
                                    data.edge_index,
                                    data.cell, data.cell_offsets,
                                    data.neighbors,
                                    return_offsets=True)
            edge_index = out["edge_index"]
            # dist = out["distances"]
            offsets = out["offsets"]
            edge_src, edge_dst = edge_index
            edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst) + offsets
            dist = edge_vec.norm(dim=1)
        else:
            edge_index = radius_graph(pos, r=self.max_radius,
                                      batch=batch, max_num_neighbors=self.max_neighbors)
            edge_src, edge_dst = edge_index
            edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
            dist = edge_vec.norm(dim=1)
            offsets = None
        return edge_index, edge_vec, dist, offsets

    def atom_trans(self, atoms):
        self.atom_list = [
            1,
            5,
            6,
            7,
            8,
            11,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            55,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
        ]
        # fill others as unk
        unk_idx = len(self.atom_list) + 1
        self.atom_mapper = torch.full((128,), unk_idx).to('cuda')
        for idx, atom in enumerate(self.atom_list):
            self.atom_mapper[atom] = idx + 1  # reserve 0 for paddin
        atoms = self.atom_mapper[atoms]
        return atoms

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # torch.cuda.empty_cache()
        # atoms: Tensor, tags: Tensor, pos: Tensor, real_mask: Tensor
        # equiformer embedding
        # data = self._forward_otf_graph(data)
        # edge_index, edge_vec, edge_length, offsets = self._forward_use_pbc(data)
        # batch = data.batch
        #
        # edge_src, edge_dst = edge_index[0], edge_index[1]
        # edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
        #                                  x=edge_vec, normalize=True, normalization='component')
        #
        # # Following Graphoformer, which encodes both atom type and tag
        # atomic_numbers = data.atomic_numbers.long()
        # atom_embedding, atom_attr, atom_onehot = self.atom_embed(atomic_numbers)
        # tags = data.tags.long()
        # tag_embedding, _, _ = self.tag_embed(tags)
        #
        # edge_length_embedding = self.rbf(edge_length, atomic_numbers,
        #                                  edge_src, edge_dst)
        # edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh,
        #                                             edge_length_embedding, edge_src, edge_dst, batch)
        # graph_node_feature = atom_embedding + tag_embedding + edge_degree_embedding
        # atom转64种
        self.device = data.pos.device
        atoms = data.atomic_numbers.long()
        atoms = self.atom_trans(atoms)
        atoms = atoms.to(self.device)
        pos = data.pos
        tags = data.tags
        batch = data.batch
        # data.edge_attr = self.distance_expansion(data.distances)

        padding_mask = atoms.eq(0)

        # n_graph, n_node = atoms.size()
        delta_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist: Tensor = delta_pos.norm(dim=-1)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        edge_type = atoms.view(1, -1, 1) * self.atom_types + atoms.view(
            1, 1, -1
        )

        gbf_feature = self.gbf(dist, edge_type)
        edge_features = gbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        graph_node_feature = (
            self.tag_encoder(tags)
            + self.atom_encoder(atoms)
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        # ===== MAIN MODEL =====
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        )
        # output = output.transpose(0, 1).contiguous()

        # graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
        # graph_attn_bias.masked_fill_(
        #     padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        # )
        #
        # graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.blocks):
            # torch.cuda.empty_cache()
            # local_out = self.local_model(output, data.edge_index, data.edge_attr)
            # output = local_out
            for enc_layer in self.layer:
                output = enc_layer(output)
            # output = local_out + output
            # output = self.final_ln(output)

        output = self.final_ln(output)

        output = self.out_energy(output).squeeze(1)
        energy = scatter(output, batch, dim=0)

        return energy

# equiformer
class NodeEmbeddingNetwork(torch.nn.Module):

    def __init__(self, irreps_node_embedding, max_atom_type=5, bias=True):
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)),
                                      self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)

    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)

        return node_embedding, node_attr, node_atom_onehot


class ScaledSiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor
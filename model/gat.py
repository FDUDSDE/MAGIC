import torch
import torch.nn as nn
from dgl.ops import edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair
from utils.utils import create_activation


class GAT(nn.Module):
    def __init__(self,
                 n_dim,
                 e_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 n_heads,
                 n_heads_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.gats = nn.ModuleList()
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if self.n_layers == 1:
            self.gats.append(GATConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, concat_out=self.concat_out
            ))
        else:
            self.gats.append(GATConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=self.concat_out
            ))
            for _ in range(1, self.n_layers - 1):
                self.gats.append(GATConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=self.concat_out
                ))
            self.gats.append(GATConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
            ))
        self.head = nn.Identity()

    def forward(self, g, input_feature, return_hidden=False):
        h = input_feature
        hidden_list = []
        for layer in range(self.n_layers):
            h = self.gats[layer](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


class GATConv(nn.Module):
    def __init__(self,
                 in_dim,
                 e_dim,
                 out_dim,
                 n_heads,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.src_feat, self.dst_feat = expand_as_pair(in_dim)
        self.edge_feat = e_dim
        self.out_feat = out_dim
        self.allow_zero_in_degree = allow_zero_in_degree
        self.concat_out = concat_out

        if isinstance(in_dim, tuple):
            self.fc_node_embedding = nn.Linear(
                self.src_feat, self.out_feat * self.n_heads, bias=False)
            self.fc_src = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
            self.fc_dst = nn.Linear(self.dst_feat, self.out_feat * self.n_heads, bias=False)
        else:
            self.fc_node_embedding = nn.Linear(
                self.src_feat, self.out_feat * self.n_heads, bias=False)
            self.fc = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
        self.edge_fc = nn.Linear(self.edge_feat, self.out_feat * self.n_heads, bias=False)
        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self.dst_feat != self.n_heads * self.out_feat:
                self.res_fc = nn.Linear(
                    self.dst_feat, self.n_heads * self.out_feat, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.norm = norm
        if norm is not None:
            self.norm = norm(self.n_heads * self.out_feat)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self.allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        edge_feature = graph.edata['attr']
        with graph.local_scope():
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self.n_heads, self.out_feat)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self.n_heads, self.out_feat)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self.n_heads, self.out_feat)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self.n_heads, self.out_feat)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self.n_heads, self.out_feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            edge_prefix_shape = edge_feature.shape[:-1]
            eh = (feat_src * self.attn_h).sum(-1).unsqueeze(-1)
            et = (feat_dst * self.attn_t).sum(-1).unsqueeze(-1)

            graph.srcdata.update({'hs': feat_src, 'eh': eh})
            graph.dstdata.update({'et': et})

            feat_edge = self.edge_fc(edge_feature).view(
                *edge_prefix_shape, self.n_heads, self.out_feat)
            ee = (feat_edge * self.attn_e).sum(-1).unsqueeze(-1)

            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_e('eh', 'ee', 'ee'))
            graph.apply_edges(fn.e_add_v('ee', 'et', 'e'))
            """
            graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
            """
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing

            graph.update_all(fn.u_mul_e('hs', 'a', 'm'),
                             fn.sum('m', 'hs'))

            rst = graph.dstdata['hs'].view(-1, self.n_heads, self.out_feat)

            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self.n_heads, self.out_feat)

            # residual

            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self.out_feat)
                rst = rst + resval

            if self.concat_out:
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)

            if self.norm is not None:
                rst = self.norm(rst)

                # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

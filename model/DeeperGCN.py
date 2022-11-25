import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from model.sparse.torch_vertex import GENConv
from .sparse.torch_nn import norm_layer, MLP
import logging





class DeeperGCN(torch.nn.Module):
    def __init__(self, num_layers, dropout, block, conv_encode_edge, add_virtual_node, hidden_channels, num_tasks,
                 conv, gcn_aggr, t, learn_t,p, learn_p, y,learn_y, msg_norm, learn_msg_scale, norm, mlp_layers, graph_pooling,activations="relu"):
        super(DeeperGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.block = block
        self.conv_encode_edge = conv_encode_edge
        self.add_virtual_node = add_virtual_node

        hidden_channels = hidden_channels
        num_tasks = num_tasks
        conv = conv
        aggr = gcn_aggr
        t = t
        self.learn_t = learn_t
        p = p
        self.learn_p = learn_p
        y = y
        self.learn_y = learn_y

        self.msg_norm = msg_norm
        learn_msg_scale = learn_msg_scale
        self.activation_func = F.relu if activations=='relu' else F.elu

        norm = norm
        mlp_layers = mlp_layers

        graph_pooling = graph_pooling

        print('The number of layers {}'.format(self.num_layers),
              'Aggr aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))
        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels]*3,
                                                     norm=norm))

        for layer in range(self.num_layers):
            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=self.conv_encode_edge, bond_encoder=True,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception('Unknown Pool Type')

        self.graph_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, input_batch, perturb = None):

        x = input_batch.x
        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        h = self.atom_encoder(x) if perturb is None else self.atom_encoder(x) + perturb


        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = self.activation_func(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = global_add_pool(h2, batch) + virtualnode_embedding
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer-1](virtualnode_embedding_temp),
                        self.dropout, training=self.training)

                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = self.activation_func(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = self.activation_func(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = self.activation_func(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = self.activation_func(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')


        return h

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))

class DeeperGCN_TU(torch.nn.Module):
    def __init__(self, num_layers, dropout, block, conv_encode_edge, add_virtual_node, hidden_channels, num_tasks,
                 conv, gcn_aggr, t, learn_t,p, learn_p, y,learn_y, msg_norm, learn_msg_scale, norm, mlp_layers, graph_pooling,activations="relu"):
        super(DeeperGCN_TU, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.block = block
        self.conv_encode_edge = conv_encode_edge
        self.add_virtual_node = add_virtual_node

        hidden_channels = hidden_channels
        num_tasks = num_tasks
        conv = conv
        aggr = gcn_aggr
        t = t
        self.learn_t = learn_t
        p = p
        self.learn_p = learn_p
        y = y
        self.learn_y = learn_y

        self.msg_norm = msg_norm
        learn_msg_scale = learn_msg_scale
        self.activation_func = F.relu if activations=='relu' else F.elu

        norm = norm
        mlp_layers = mlp_layers

        graph_pooling = graph_pooling

        print('The number of layers {}'.format(self.num_layers),
              'Aggr aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))
        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels]*3,
                                                     norm=norm))

        for layer in range(self.num_layers):
            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=self.conv_encode_edge, bond_encoder=True,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        # self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)
        #
        # if not self.conv_encode_edge:
        #     self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception('Unknown Pool Type')

        self.graph_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, input_batch, perturb = None):

        x = input_batch.x
        edge_index = input_batch.edge_index
        #edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        h = x if perturb is None else x + perturb


        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
            h = h + virtualnode_embedding[batch]
        #
        # if self.conv_encode_edge:
        #     edge_emb = edge_attr
        # else:
        #     edge_emb = self.bond_encoder(edge_attr)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = self.activation_func(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = global_add_pool(h2, batch) + virtualnode_embedding
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer-1](virtualnode_embedding_temp),
                        self.dropout, training=self.training)

                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index) + h

            h = self.norms[self.num_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = self.activation_func(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = self.activation_func(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = self.activation_func(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = self.activation_func(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')


        return h
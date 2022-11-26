import torch
import torch.nn.functional as F
from model.graph_node import GNN_node, GNN_node_Virtualnode, GNN_node_Virtualnode_TU,GNN_node_TU
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from model.basic import MLP_PLain,MLP_multi, basic_MLP, GIN, MLP_Sigmoid, MLP_PLain_norm
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch
from torch_scatter import scatter
import torch.nn as nn
from torch_geometric.data import Data
from model.basic import GIN, GCN
from model.DeeperGCN import DeeperGCN, DeeperGCN_TU






class patchGT_TU(torch.nn.Module):
	def __init__(self, args, num_tasks=1, num_layer=5, emb_dim=128,
				 gnn_type='gin', coarse_gnn_type = 'gin', virtual_node=True, residual=True, drop_ratio=0.2,att_drop=0.2, JK="last", graph_pooling="mean", cls_token=True,
				 num_heads=4, attention_layers=2,patch_gcn_num_layers=3,in_edge_channels=4, patch_pooling='sum', device='cuda:0', node_dim=7):
		super(patchGT_TU, self).__init__()
		self.device = device
		self.num_layer = num_layer
		self.drop_ratio = drop_ratio
		self.JK = JK
		self.emb_dim = emb_dim
		self.num_tasks = num_tasks
		self.graph_pooling = graph_pooling
		self.patch_pooling = patch_pooling
		self.attention_layers = attention_layers
		self.use_cls_token = cls_token
		self.gnn_type =gnn_type
		self.node_encoder = basic_MLP(node_dim, args.n_embd, args.n_embd)
		if self.num_layer < 2:
			raise ValueError("Number of GNN layers must be greater than 1.")
			### GNN to generate node embeddings
		if self.gnn_type == 'deepergcn':
			self.gnn_node = DeeperGCN_TU(num_layer, dropout=drop_ratio, block='res+', conv_encode_edge=False,
									  add_virtual_node=virtual_node, hidden_channels=self.emb_dim, num_tasks=self.num_tasks,
									  conv='gen', gcn_aggr='softmax', t=1.0, learn_t=True, p=1.0, learn_p=False, y=0.0,
									  learn_y=False, msg_norm=True, learn_msg_scale=True, norm='batch', mlp_layers=2,
									  graph_pooling=graph_pooling, activations="relu")
		else:
			if virtual_node:
				self.gnn_node = GNN_node_Virtualnode_TU(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
													 residual=residual, gnn_type=gnn_type)
			else:
				self.gnn_node = GNN_node_TU(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
										 gnn_type=gnn_type)



		### Pooling function to generate whole-graph embeddings
		if self.graph_pooling == "sum":
			self.pool = global_add_pool
		elif self.graph_pooling == "mean":
			self.pool = global_mean_pool
		elif self.graph_pooling == "max":
			self.pool = global_max_pool
		elif self.graph_pooling == "attention":
			self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
		elif self.graph_pooling == "set2set":
			self.pool = Set2Set(emb_dim, processing_steps = 2)
		else:
			raise ValueError("Invalid graph pooling type.")
		if graph_pooling == "set2set":
			self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
		else:
			self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

		#Transformer
		self.mhas = nn.ModuleList()
		self.mlps = nn.ModuleList()
		for i in range(attention_layers):
			self.mhas.append(torch.nn.MultiheadAttention(emb_dim, num_heads, dropout=att_drop, batch_first=True))
			self.mlps.append(MLP_PLain_norm(args.n_embd, args.n_embd, args.n_embd))
		# mesh graoh net
		#mesh graoh net
		#self.patch_gnn = MGN(patch_gcn_num_layers,emb_dim, in_edge_channels, emb_dim, hidFeature=emb_dim, aggr=patch_pooling)
		if coarse_gnn_type == 'gin':
			self.patch_gnn =  GIN(emb_dim, patch_gcn_num_layers, emb_dim, emb_dim)
		if coarse_gnn_type == 'gcn':
			self.patch_gnn = GCN(emb_dim, patch_gcn_num_layers, emb_dim, emb_dim)


		#predictor
		#self.predictor = MLP_PLain(args.n_embd, args.n_embd, num_tasks).to(args.device) #num_tasks = 1 for binary classification problem
		#cls_token
		self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
		self.out_norm = nn.LayerNorm(args.n_embd)
		nn.init.normal_(self.cls_token, std=1e-6)
		self.bug = 0
	def forward(self, batch_g, color_list, center_list, color_number,batch_size, sender_list, receiver_list, group_number, perturb = None):
		old_x = batch_g.x
		g = batch_g.to(self.device)
		#edge_index = g.edge_index
		g.x = self.node_encoder(g.x)
		new_x = self.gnn_node(g, perturb)


		g.x=new_x
		g_list = [g[i] for i in range(batch_size)]
		pooling = [self.subgraph_pooling(g_list[i].x, color_list[i], color_number[i], group_number[i]) for i in range(batch_size)]
		ng_list = [self.build_connect_graph(pooling[i], center_list[i], color_number[i], sender_list[i], receiver_list[i]) for i in range(batch_size)]
		batch_ng = Batch.from_data_list(ng_list)

		#px = self.patch_gnn(batch_ng.x, batch_ng.edge_index)   #These two lines control wheter to use patch level gnn
		#batch_ng.x = px


		x, mask = to_dense_batch(batch_ng.x, batch_ng.batch, fill_value=0)

		if self.use_cls_token == True:
			expand_cls_token = self.cls_token.expand(batch_size, -1, -1)
			for layer_index in range(self.attention_layers):
				_expand_cls_token, _ = self.mhas[layer_index](expand_cls_token, x, x, key_padding_mask=~mask)
				expand_cls_token = self.mlps[layer_index](_expand_cls_token) + expand_cls_token

			x = torch.squeeze(expand_cls_token,dim=1)
			x = self.out_norm(x)
			x = F.dropout(x, p=0.5, training=self.training)
			prediction = self.graph_pred_linear (x)
		else:
			x = global_mean_pool(batch_ng.x, batch_ng.batch)
			prediction = self.graph_pred_linear (x)

		batch_g.x = old_x
		return F.log_softmax(prediction, dim=-1)



	def subgraph_pooling(self,x,color, k, gg):


		pooling = scatter(x, color, dim=0, reduce=self.patch_pooling)  # mean, max, sum, min
		dis = k - pooling.shape[0]
		if dis != 0:
			pooling = torch.cat((pooling, torch.zeros((dis, 512)).to(self.device)), dim=0)
			print('make up pooling!')
		weight = gg/gg.sum()
		weight = weight.unsqueeze(dim=-1)
		if weight.size()[0] == pooling.size()[0]:
		    pooling = weight*pooling
		else:
			print("weight pooling failed!")

		return pooling


	def build_connect_graph(self, pooling, eigen_center, k, senders, receivers):

		x = pooling
		senders = torch.tensor(senders, dtype=torch.long).to(self.device)
		receivers = torch.tensor(receivers, dtype=torch.long).to(self.device)

		relative_pos = (torch.index_select(eigen_center, 0, senders) -
							torch.index_select(eigen_center, 0, receivers))

		edge_attr = torch.cat((
				relative_pos,
				torch.norm(relative_pos, dim=-1, keepdim=True)), dim=-1)
		edge_index = torch.cat((senders.unsqueeze(0), receivers.unsqueeze(0)), dim=0)
		graph = Data(x=x, edge_index=edge_index)

		return graph.to(self.device)



class patchGT(torch.nn.Module):
	def __init__(self, args, num_tasks=1, num_layer=5, emb_dim=128,
				 gnn_type='gin', coarse_gnn_type = 'gin', virtual_node=True, residual=True, drop_ratio=0.2,att_drop=0.2, JK="last", graph_pooling="mean", cls_token=True,
				 num_heads=4, attention_layers=2,patch_gcn_num_layers=3,in_edge_channels=4, patch_pooling='sum', device='cuda:0'):
		super(patchGT, self).__init__()
		self.device = device
		self.num_layer = num_layer
		self.drop_ratio = drop_ratio
		self.JK = JK
		self.emb_dim = emb_dim
		self.num_tasks = num_tasks
		self.graph_pooling = graph_pooling
		self.patch_pooling = patch_pooling
		self.attention_layers = attention_layers
		self.use_cls_token = cls_token
		self.gnn_type =gnn_type
		self.args = args
		if self.num_layer < 2:
			raise ValueError("Number of GNN layers must be greater than 1.")
			### GNN to generate node embeddings
		if self.gnn_type == 'deepergcn':
			self.gnn_node = DeeperGCN(num_layer, dropout=drop_ratio, block='res+', conv_encode_edge=False,
									  add_virtual_node=virtual_node, hidden_channels=self.emb_dim, num_tasks=self.num_tasks,
									  conv='gen', gcn_aggr='softmax', t=1.0, learn_t=True, p=1.0, learn_p=False, y=0.0,
									  learn_y=False, msg_norm=True, learn_msg_scale=True, norm='batch', mlp_layers=2,
									  graph_pooling=graph_pooling, activations="relu")
		else:
			if virtual_node:
				self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
													 residual=residual, gnn_type=gnn_type)
			else:
				self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
										 gnn_type=gnn_type)



		### Pooling function to generate whole-graph embeddings
		if self.graph_pooling == "sum":
			self.pool = global_add_pool
		elif self.graph_pooling == "mean":
			self.pool = global_mean_pool
		elif self.graph_pooling == "max":
			self.pool = global_max_pool
		elif self.graph_pooling == "attention":
			self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
		elif self.graph_pooling == "set2set":
			self.pool = Set2Set(emb_dim, processing_steps = 2)
		else:
			raise ValueError("Invalid graph pooling type.")
		if graph_pooling == "set2set":
			self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
		else:
			self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

		#Transformer
		self.mhas = nn.ModuleList()
		self.mlps = nn.ModuleList()
		for i in range(attention_layers):
			self.mhas.append(torch.nn.MultiheadAttention(emb_dim, num_heads, dropout=att_drop, batch_first=True))
			self.mlps.append(MLP_PLain_norm(args.n_embd, args.n_embd, args.n_embd))


		if coarse_gnn_type == 'gin':
			self.patch_gnn =  GIN(emb_dim, patch_gcn_num_layers, emb_dim, emb_dim)
		if coarse_gnn_type == 'gcn':
			self.patch_gnn = GCN(emb_dim, patch_gcn_num_layers, emb_dim, emb_dim)


		self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
		self.out_norm = nn.LayerNorm(args.n_embd)
		nn.init.normal_(self.cls_token, std=1e-6)
		self.bug = 0
	def forward(self, batch_g, color_list, center_list, color_number,batch_size, sender_list, receiver_list, group_number, perturb = None):
		old_x = batch_g.x
		g = batch_g.to(self.device)
		#edge_index = g.edge_index
		new_x = self.gnn_node(g, perturb)


		g.x=new_x
		g_list = [g[i] for i in range(batch_size)]
		pooling = [self.subgraph_pooling(g_list[i].x, color_list[i], color_number[i], group_number[i]) for i in range(batch_size)]
		ng_list = [self.build_connect_graph(pooling[i], center_list[i], color_number[i], sender_list[i], receiver_list[i]) for i in range(batch_size)]
		batch_ng = Batch.from_data_list(ng_list)

		# px = self.patch_gnn(batch_ng.x, batch_ng.edge_index)            #These two lines control wheter to use patch level gnn
		# batch_ng.x = px


		x, mask = to_dense_batch(batch_ng.x, batch_ng.batch, fill_value=0)

		if self.use_cls_token == True:
			expand_cls_token = self.cls_token.expand(batch_size, -1, -1)
			for layer_index in range(self.attention_layers):
				_expand_cls_token, _ = self.mhas[layer_index](expand_cls_token, x, x, key_padding_mask=~mask)
				expand_cls_token = self.mlps[layer_index](_expand_cls_token) + expand_cls_token

			x = torch.squeeze(expand_cls_token,dim=1)
			x = self.out_norm(x)
			x = F.dropout(x, p=0.2, training=self.training)
			prediction = self.graph_pred_linear (x)
		else:
			x = global_mean_pool(batch_ng.x, batch_ng.batch)
			prediction = self.graph_pred_linear (x)

		batch_g.x = old_x
		return prediction


	def subgraph_pooling(self,x,color, k, gg):
		'''
		:param x: node features
		:param color: tensor
		:param k: int
		:return: sub_pooling： tensor(k, x_feature_dim)
		'''


		pooling = scatter(x, color, dim=0, reduce=self.patch_pooling)  # mean, max, sum, min
		dis = k - pooling.shape[0]
		if dis != 0:
			pooling = torch.cat((pooling, torch.zeros((dis, self.args.n_embd)).to(self.device)), dim=0)
			print('make up pooling!')
		weight = gg/gg.sum()
		weight = weight.unsqueeze(dim=-1)
		if weight.size()[0] == pooling.size()[0]:
		    pooling = weight*pooling
		else:
			print("weight pooling failed!")

		return pooling


	def build_connect_graph(self, pooling, eigen_center, k, senders, receivers):

		x = pooling
		senders = torch.tensor(senders, dtype=torch.long).to(self.device)
		receivers = torch.tensor(receivers, dtype=torch.long).to(self.device)

		relative_pos = (torch.index_select(eigen_center, 0, senders) -
							torch.index_select(eigen_center, 0, receivers))

		edge_attr = torch.cat((
				relative_pos,
				torch.norm(relative_pos, dim=-1, keepdim=True)), dim=-1)
		edge_index = torch.cat((senders.unsqueeze(0), receivers.unsqueeze(0)), dim=0)
		graph = Data(x=x, edge_index=edge_index)

		return graph.to(self.device)


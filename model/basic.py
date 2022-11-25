import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv,GINConv, global_mean_pool, global_add_pool




class MLP(torch.nn.Module):
	def __init__(self,nIn,nOut,Hidlayer, withReLU):
		super(MLP, self).__init__()
		numHidlayer=len(Hidlayer)
		net=[]
		net.append(torch.nn.Linear(nIn,Hidlayer[0]))
		if withReLU:
			net.append(torch.nn.ReLU())
		for i in range(0,numHidlayer-1):
			net.append(torch.nn.Linear(Hidlayer[i],Hidlayer[i+1]))
			if withReLU:
				net.append(torch.nn.ReLU())
		net.append(torch.nn.Linear(Hidlayer[-1],nOut))#
		self.mlp=torch.nn.Sequential(*net)
	def forward(self,x):
		return self.mlp(x)



class MLP_PLain(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size=1, dropout=0.1):
        super(MLP_PLain, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
            #nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)

class MLP_PLain_norm(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size=1):
        super(MLP_PLain_norm, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
            #nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)

class MLP_Sigmoid(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size=1, dropout=0.1):
        super(MLP_Sigmoid, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)


class MLP_multi(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0.1):
        super(MLP_multi, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),

        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return F.log_softmax(self.mlp(input), dim=-1)


class MLP_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size=1, dropout=0):
        super(MLP_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
            nn.Softmax(dim=-1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)

class basic_MLP(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size=1, dropout=0):
        super( basic_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
            nn.ReLU()

        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)

class GIN(torch.nn.Module):
    def __init__(self, input_embd, num_layers, hidden, output_embd):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(input_embd, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, output_embd)
        hidNodeNums = [hidden,
                       hidden,
                       hidden]
        self.lin2 =  MLP(hidden, output_embd, hidNodeNums,True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        # x = global_mean_pool(torch.cat(xs, dim=1), batch)
        # x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__

class GCN(torch.nn.Module):
    def __init__(self, input_embd, num_layers, hidden, output_embd):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_embd, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, output_embd)
        hidNodeNums = [hidden,
                       hidden,
                       hidden]
        self.lin2 =  MLP(hidden, output_embd, hidNodeNums,True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        # x = global_mean_pool(torch.cat(xs, dim=1), batch)
        # x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
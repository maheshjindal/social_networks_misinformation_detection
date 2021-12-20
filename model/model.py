import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel, HypergraphConv

class FakeNewsDetection(torch.nn.Module):
    """
    The FakeNewsDetection model Class
    """
    def __init__(self, args, concat=False):
        super(FakeNewsDetection, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model = args.model
        self.concat = concat

        if self.model == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)
        elif self.model == 'hyperconv':
            self.conv1 = HypergraphConv(self.num_features, self.nhid)

        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 8, self.nhid * 4)
            self.lin2 = torch.nn.Linear(self.nhid * 4, self.nhid * 2)
            self.lin3 = torch.nn.Linear(self.nhid * 2, self.nhid)
            self.dropout = torch.nn.Dropout(self.dropout_ratio)
        self.lin4 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch, embedding_tensor = data.x, data.edge_index, data.batch, data.embedding_data
        edge_attr = None
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = gmp(x, batch)

        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news, embedding_tensor], dim=1)
            x = F.tanh(self.lin1(x))
            x = self.dropout(x)
            x = F.tanh(self.lin2(x))
            x = self.dropout(x)
            x = F.tanh(self.lin3(x))
            x = self.dropout(x)
        x = F.log_softmax(self.lin4(x), dim=-1)
        return x

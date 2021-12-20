import argparse
import time
from tqdm import tqdm
import os
import torch
from torch_geometric.data import DataLoader, DataListLoader
from data_loader.data_loaders import UPFD
from torch_geometric.transforms import ToUndirected
from model.model import FakeNewsDetection
from utils.util import train_model, evaluate_model

# The argument parser instance
parser = argparse.ArgumentParser()

# Retrieves the current working directory path
cwd = os.getcwd()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--news_embedding_path', type=str, default=os.getcwd() + '/data/news2embed.pkl',
                    help='News Embeddings Path')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='sage', help='model type, [gcn, gat, sage, hyperconv]')

# Parses the arguments
args = parser.parse_args()

# Sets the custom seed (if any from args)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
# Uses the 'gpu' cores if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creating Datasets and their respective data loaders
train_dataset = UPFD('data', args.dataset, args.feature, args.news_embedding_path, 'train', ToUndirected())
val_dataset = UPFD('data', args.dataset, args.feature, args.news_embedding_path, 'val', ToUndirected())
test_dataset = UPFD('data', args.dataset, args.feature, args.news_embedding_path, 'test', ToUndirected())
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Creates the Fake News Model Instance
model = FakeNewsDetection(args, concat=True)

# Configures the Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if __name__ == '__main__':
    # Starts model training
    start_time = time.time()
    print(f'Training started at {start_time:.4f} seconds.')
    for epoch in tqdm(range(args.epochs)):
        loss = train_model(model, train_loader, device, optimizer)
        train_acc = evaluate_model(model, train_loader, device)
        val_acc = evaluate_model(model, val_loader, device)
        test_acc = evaluate_model(model, test_loader, device)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f},'
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    end_time = time.time()
    print(f'Training ended at {end_time:.4f} seconds.')

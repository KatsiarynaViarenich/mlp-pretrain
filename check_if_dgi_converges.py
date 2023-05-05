import os.path as osp
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import torch
import numpy as np

from tqdm import tqdm

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch.nn import Linear
from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor
import torch.utils.data as data_utils

from torch_geometric.nn.models import DeepGraphInfomax
import wandb
import argparse

import sys
from src.load_dataset import load_data


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-arxiv", choices=["ogbn-products", "ogbn-arxiv", "Reddit", "Flickr", "Yelp", "AmazonProducts", "Reddit2"])
args = parser.parse_args()

if args.dataset == "Flickr":
    args.num_classes = 7
    args.num_feats = 500

elif args.dataset == "Reddit":
    args.num_classes = 41
    args.num_feats = 602

elif args.dataset == "Reddit2":
    args.num_classes = 41
    args.num_feats = 602

elif args.dataset == "ogbn-products":
    args.multi_label = False
    args.num_classes = 47
    args.num_feats = 100

elif args.dataset == "AmazonProducts":
    args.multi_label = True
    args.num_classes = 107
    args.num_feats = 200

elif args.dataset == "Yelp":
    args.multi_label = True
    args.num_classes = 100
    args.num_feats = 300

elif args.dataset == "ogbn-arxiv":
    args.num_feats = 128
    args.num_classes = 40
    args.N_nodes = 169343


num_feats = args.num_feats
num_classes = args.num_classes
N_nodes = args.N_nodes

wandb.init(
    project="pretrain-mlpinit",
    dir="./wandb",
    config={
        "dataset_name": "ogbn-arxiv",
        "batch_size": 4096,
        "num_layers": 4,
        "hidden_channels": 1024,
        "percent_corrupted": 0.25,
        "custom_step": 0
    }
)

dataset_dir = "./data"
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = osp.join(dataset_dir, wandb.config.dataset_name)

(data, x, y, split_masks, evaluator, processed_dir) = load_data("ogbn-arxiv", "./data")

train_idx = split_masks["train"]

x_train = data.x[split_masks["train"]]
y_train = data.y[split_masks["train"]].reshape(-1).type(torch.long)

print("data.x.shape:", data.x.shape)
print("data.y.shape:", data.y.shape)
print("data.x.type:", x.dtype)
print("data.y.type:", y.dtype)
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)

y = data.y.squeeze().type(torch.long)

x_y_train_mlpinit = data_utils.TensorDataset(x_train, y_train)
x_y_all_mlpinit = data_utils.TensorDataset(x, y)

train_mlpinit_loader = data_utils.DataLoader(
    x_y_train_mlpinit,
    batch_size=wandb.config.batch_size,
    shuffle=True,
    num_workers=num_workers,
)
all_mlpinit_loader = data_utils.DataLoader(
    x_y_all_mlpinit,
    batch_size=wandb.config.batch_size,
    shuffle=False,
    num_workers=num_workers,
)


class SAGEConv_PeerMLP(torch.nn.Module):
    """
    A PyTorch module implementing a simplified GraphSAGE convolution-like multilayer perceptron (MLP) layer.

    This layer performs a linear transformation on the input node features, optionally normalizing
    the output and adding a root weight.
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            normalize: bool = False,
            root_weight: bool = True,
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor]) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = x[1]
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv_PeerMLP(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv_PeerMLP(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv_PeerMLP(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        prev_x = x
        for i in range(self.num_layers):
            x_target = x
            x = self.convs[i]((x, x_target))
            if i != self.num_layers - 1:
                x = F.relu(x)
        if x.shape[1] == prev_x.shape[1]:
            x = x + prev_x
        return x.log_softmax(dim=-1)


model_mlpinit = MLP(
    in_channels=args.num_feats,
    hidden_channels=wandb.config.hidden_channels,
    out_channels=args.num_classes,
    num_layers=wandb.config.num_layers,
)

model_mlpinit = model_mlpinit.to(device)
optimizer_model_mlpinit = torch.optim.Adam(model_mlpinit.parameters(), lr=0.001, weight_decay=0.0)


def index_corruption(x):
    num_nodes = x.size()[0]
    mask = torch.ones(num_nodes, num_feats)
    mask[:][torch.randperm(num_nodes)[:int(num_feats * wandb.config.percent_corrupted)]] = 0
    mask = mask.bool().to(device)

    x = torch.where(mask.bool(), x, torch.zeros_like(x))
    return x


def dropout_corruption(x, p=0.9):
    mask = torch.empty_like(x).bernoulli_(p)
    x = torch.where(mask.bool(), x, torch.zeros_like(x))
    return x


def summary(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))


unsupervised_model = DeepGraphInfomax(hidden_channels=num_classes, encoder=model_mlpinit, summary=summary,
                                          corruption=dropout_corruption)
unsupervised_model.to(device)
def train_mlpinit():
    total_loss = 0
    unsupervised_model.train()
    for x, _ in tqdm(train_mlpinit_loader):
        x = x.to(device)

        optimizer_model_mlpinit.zero_grad()
        pos_z, neg_z, summary_value = unsupervised_model(x)
        loss = unsupervised_model.loss(pos_z, neg_z, summary_value)
        loss.backward()
        optimizer_model_mlpinit.step()

        total_loss += float(loss)

    loss_percent = total_loss / len(train_mlpinit_loader)
    wandb.log({"loss_dgi": loss_percent})
    unsupervised_model.eval()
    return loss_percent, 0


model_mlpinit.reset_parameters()

for epoch in range(1, 50):
    loss_without_shadow, acc = train_mlpinit()  # p is the probability of dropping a feature

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


wandb.init(
    project="pretrain-mlpinit",
    dir="./wandb",
    config={
        "dataset_name": args.dataset,
        "batch_size": 4096,
        "num_layers": 4,
        "hidden_channels": 512,
        "percent_corrupted": 0.25,
        "custom_step": 0
    }
)

dataset_dir = "./data"
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = osp.join(dataset_dir, wandb.config.dataset_name)

dataset = PygNodePropPredDataset(wandb.config.dataset_name, root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name=wandb.config.dataset_name)
data = dataset[0]
train_idx = split_idx["train"]

x_train = data.x[split_idx["train"]]
y_train = data.y[split_idx["train"]].reshape(-1).type(torch.long)

x = data.x
y = data.y.squeeze()

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
        for i in range(self.num_layers):
            x_target = x
            x = self.convs[i]((x, x_target))
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description("Evaluating")

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


model_mlpinit = MLP(
    in_channels=dataset.num_features,
    hidden_channels=wandb.config.hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=wandb.config.num_layers,
)

model_mlpinit = model_mlpinit.to(device)
optimizer_model_mlpinit = torch.optim.Adam(model_mlpinit.parameters(), lr=0.001, weight_decay=0.0)


def train_mlpinit():
    def index_corruption(x):
        num_nodes = x.size()[0]
        mask = torch.ones(num_nodes, args.num_feats)
        mask[:][torch.randperm(num_nodes)[:args.num_feats*wandb.config.percent_corrupted]] = 0
        mask = mask.bool().to(device)

        x = torch.where(mask.bool(), x, torch.zeros_like(x))
        return x

    def dropout_corruption(x):
        pass

    def summary(z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    total_loss = 0

    unsupervised_model = DeepGraphInfomax(hidden_channels=args.num_classes, encoder=model_mlpinit, summary=summary,
                                          corruption=index_corruption)
    unsupervised_model.to(device)
    unsupervised_model.train()
    for x, _ in tqdm(train_mlpinit_loader):
        x = x.to(device)

        optimizer_model_mlpinit.zero_grad()
        pos_z, neg_z, summary = unsupervised_model(x)
        loss = unsupervised_model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer_model_mlpinit.step()

        total_loss += float(loss)

    loss = total_loss / len(train_mlpinit_loader)
    wandb.log({"loss_dgi": loss})
    unsupervised_model.eval()
    return loss, 0


@torch.no_grad()
def test_mlpinit():
    model_mlpinit.eval()

    out_list = []
    y_list = []

    for x, y in tqdm(all_mlpinit_loader):
        x = x.to(device)
        y = y.to(device)
        out = model_mlpinit(x)
        out_list.append(out)
        y_list.append(y)

    out = torch.cat(out_list, dim=0)
    y_true = torch.cat(y_list, dim=0).cpu().unsqueeze(-1)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    val_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return train_acc, val_acc, test_acc


train_loader = NeighborSampler(
    data.edge_index,
    node_idx=train_idx,
    sizes=[25, 10, 5, 5],
    batch_size=wandb.config.batch_size,
    shuffle=True,
    num_workers=num_workers,
)
subgraph_loader = NeighborSampler(
    data.edge_index,
    node_idx=None,
    sizes=[-1],
    batch_size=wandb.config.batch_size,
    shuffle=False,
    num_workers=num_workers,
)

model = SAGE(
    in_channels=dataset.num_features,
    hidden_channels=wandb.config.hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=wandb.config.num_layers,
)
model = model.to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0

    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()

        out = model(x[n_id].to(device), adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]].to(device))
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]].to(device)).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    val_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return train_acc, val_acc, test_acc


random_losses = []
random_test_accs = []

model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

best_val_acc = final_test_acc = 0
for epoch in range(1, 20):
    loss, acc = train(epoch)
    train_acc, val_acc, test_acc = test()
    print(f'Epoch {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
    wandb.log({"loss_random": loss, "acc_random": test_acc})
    random_losses.append(loss)
    random_test_accs.append(test_acc)

model_mlpinit.reset_parameters()

for epoch in range(1, 20):
    loss, acc = train_mlpinit()

torch.save(model_mlpinit.state_dict(), f'./model_mlpinit.pt')
train_acc_init, val_acc_init, test_acc_init = test_mlpinit()
print("train_acc_init, val_acc_init, test_acc_init:", train_acc_init, val_acc_init, test_acc_init)

mlpinit_losses = []
mlpinit_test_accs = []

model.load_state_dict(torch.load(f'./model_mlpinit.pt'))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

best_val_acc = final_test_acc = 0
for epoch in range(0, 20):
    loss, acc = train(epoch)
    train_acc, val_acc, test_acc = test()
    print(f'Epoch {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
    mlpinit_losses.append(loss)
    mlpinit_test_accs.append(test_acc)

    wandb.define_metric("loss_mlpinit", step_metric='custom_step')
    wandb.define_metric("acc_mlpinit", step_metric='custom_step')
    log_dict = {
        "custom_step": epoch,
        "loss_mlpinit": loss,
        "acc_mlpinit": test_acc
    }
    wandb.log(log_dict)


def find_best_speedup(random_accs, mlpinit_accs):
    best_speedup = 0
    best_line = 0
    for value in np.linspace(start=0, stop=1, num=10):
        first_achieved = 0
        second_achieved = 1000
        for j in range(len(random_accs)):
            if random_accs[j] > value:
                first_achieved = j
                break
        first_achieved += 1
        for j in range(len(mlpinit_accs)):
            if mlpinit_accs[j] > value:
                second_achieved = j
                break
        second_achieved += 1
        if second_achieved == 0:
            best_speedup = 0
        else:
            if best_speedup < first_achieved / second_achieved:
                best_line = value
                best_speedup = first_achieved / second_achieved
    return best_line, best_speedup


best_line, best_speedup = find_best_speedup(random_test_accs, mlpinit_test_accs)
my_table = wandb.Table(columns=["best_line", "best_speedup"], data=[[best_line, best_speedup]])
wandb.log({"table": my_table})

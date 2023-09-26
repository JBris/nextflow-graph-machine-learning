#!/usr/bin/env python

######################################
# Imports
######################################

from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient
import hydra
import matplotlib.pyplot as plt
import mlflow
import networkx as nx
from omegaconf import DictConfig
from os.path import join as join_path
import pandas as pd
from pathlib import Path
import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, VGAE

######################################
# Classes
######################################


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 2,
        normalize: bool = False,
        bias: bool = True,
        aggr: str = "mean",
    ) -> None:
        """
        SAGENet constructor.

        Args:
            in_channels (int):
                The number of input channels.
            hidden_channels (int):
                The number of hidden channels.
            out_channels (int):
                The number of output channels.
            n_layers (int, optional):
                The number of SAGE convolutional layers. Defaults to 5.
            normalize (bool, optional):
                Whether to apply normalisation. Defaults to False.
            bias (bool, optional):
                Whether to include the bias term. Defaults to True.
            aggr (str, optional):
                The tensor aggregation type. Defaults to "mean".
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.conv1 = SAGEConv(
            in_channels, hidden_channels, normalize=normalize, aggr=aggr, bias=bias
        )

        self.conv2 = SAGEConv(
            hidden_channels, 2 * out_channels, normalize=normalize, aggr=aggr, bias=bias
        )

        self.conv_mu = SAGEConv(
            2 * out_channels, out_channels, normalize=normalize, aggr=aggr, bias=bias
        )

        self.conv_logstd = SAGEConv(
            2 * out_channels, out_channels, normalize=normalize, aggr=aggr, bias=bias
        )

        self.layers.append(self.conv1)
        for _ in range(n_layers):
            self.layers.append(
                SAGEConv(
                    hidden_channels,
                    hidden_channels,
                    normalize=normalize,
                    aggr=aggr,
                    bias=bias,
                )
            )

        self.activation = F.leaky_relu

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass.

        Args:
            x (torch.Tensor):
                Input data.
            edge_index (torch.Tensor):
                The graph edge index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                The convolutional mean and log-standard deviation.
        """
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.activation(x)

        x = self.conv2(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


######################################
# Functions
######################################


def log_results(
    tracking_uri: str,
    experiment_prefix: str,
    grn_name: str,
    in_channels: int,
    config: DictConfig,
) -> None:
    """
    Log experiment results to the experiment tracker.

    Args:
        tracking_uri (str):
            The tracking URI.
        experiment_prefix (str):
            The experiment name prefix.
        grn_name (str):
            The name of the GRN.
        in_channels (int):
            The number of input channels.
        config (DictConfig):
            The pipeline configuration.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"{experiment_prefix}_train_vae_gnn"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    mlflow.set_tag("grn", grn_name)
    mlflow.set_tag("gnn", "VAE")

    mlflow.log_param("grn", grn_name)
    mlflow.log_param("in_channels", in_channels)

    for k in config["gnn"]:
        mlflow.log_param(k, config["gnn"][k])


def get_graph(
    db_host: str,
    db_name: str,
    db_username: str,
    db_password: str,
    collection: str,
    feature_k: str = "expression",
) -> nx.Graph:
    """
    Retrieve the graph from the database.

    Args:
        db_host (str):
            The database host.
        db_name (str):
            The database name.
        db_username (str):
            The database username.
        db_password (str):
            The database password.
        collection (str):
            The database collection.
        feature_k (str):
            The dictionary key for node features.

    Returns:
        nx.Graph:
            The retrieved graph.
    """
    db = ArangoClient(hosts=db_host).db(
        db_name, username=db_username, password=db_password
    )
    adapter = ADBNX_Adapter(db)
    db_G = adapter.arangodb_graph_to_networkx(collection)
    db_G = nx.Graph(db_G)
    db_G = nx.convert_node_labels_to_integers(db_G)

    G = nx.Graph()
    G.add_edges_from(db_G.edges)
    for node_id, node_features in list(db_G.nodes(data=True)):
        features = list(node_features[feature_k].values())
        G.nodes[node_id][feature_k] = features

    return G


def get_split(
    G: nx.Graph, num_val: float, num_test: float, device: torch.device
) -> tuple[nx.Graph, nx.Graph, nx.Graph]:
    """
    Get train-validation-test split.

    Args:
        G (nx.Graph):
            The graph.
        num_val (float):
            The proportion of validation data.
        num_test (float):
            The proportion of testing data.
        device (torch.device):
            The training device.

    Returns:
        tuple[nx.Graph, nx.Graph, nx.Graph]:
            The train-validation-test split.
    """
    transform = T.Compose(
        [
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(
                num_val=num_val,
                num_test=num_test,
                is_undirected=True,
                add_negative_train_samples=False,
                split_labels=True,
            ),
        ]
    )

    train_data, val_data, test_data = transform(G)
    return train_data, val_data, test_data


def get_model_components(
    lr: float,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    device: torch.device,
    n_layers: int,
    normalize: bool,
    bias: bool,
    aggr: str,
) -> tuple:
    """
    Get the components for training the model.

    Args:
        lr (float):
            The learning rate.
        in_channels (int):
            The number of input channels.
        hidden_channels (int):
            The number of hidden channels.
        out_channels (int):
            The number of output channels.
        device (torch.device):
            The training device.
        n_layers (int):
            The number of SAGE convolutional layers.
        normalize (bool):
            Whether to normalize the input tensors.
        bias (bool):
            Whether to include the bias term.
        aggr (str):
            The data aggregation method.

    Returns:
        tuple:
            The components for training the model.
    """
    model = VGAE(
        VariationalGCNEncoder(
            in_channels,
            hidden_channels,
            out_channels,
            n_layers,
            normalize,
            bias,
            aggr,
        )
    ).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.05
    )

    return model, optimizer, scheduler


def train_model(
    model: torch.nn.Module,
    train_data: nx.Graph,
    val_data: nx.Graph,
    test_data: nx.Graph,
    n_epochs: int,
    optimizer: torch.nn.Module,
    device: torch.device,
    enable_tracking: bool,
) -> float:
    """
    Train the graph neural network.

    Args:
        model (torch.nn.Module):
            The graph neural network.
        train_data (nx.Graph):
            The training data.
        val_data (nx.Graph):
            The validation data.
        test_data (nx.Graph):
            The testing data.
        n_epochs (int):
            The number of epochs.
        optimizer (torch.nn.Module):
            The model optimiser.
        device (torch.device):
            The training device.
        enable_tracking (bool):
            Whether to enable experiment tracking.

    Returns:
        float:
            The final area-under-curve score.
    """

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.expression, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.expression, data.edge_index)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    for epoch in range(n_epochs):
        loss = train()
        val_auc, val_ap = test(val_data)
        test_auc, test_ap = test(test_data)

        if epoch % int(n_epochs * 0.05) == 0:
            if enable_tracking:
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("val_auc", val_auc, step=epoch)
                mlflow.log_metric("val_ap", val_ap, step=epoch)
                mlflow.log_metric("test_auc", test_auc, step=epoch)
                mlflow.log_metric("test_ap", test_ap, step=epoch)

            print(
                f"Epoch: {epoch:03d}, loss {loss:.4f}",
                f"Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}",
                f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}",
            )

    final_test_auc, final_test_ap = test(test_data)
    print(f"Final Test AUC: {final_test_auc:.4f}, Final Test AP: {final_test_ap:.4f}")

    if enable_tracking:
        mlflow.log_metric("final_test_auc", final_test_auc)
        mlflow.log_metric("final_test_ap", final_test_ap)

    return model


def view_embeddings(
    model: torch.nn.Module, data: nx.Graph, output_dir: str, enable_tracking: bool
) -> str:
    """
    View the latent embeddings in 2D.

    Args:
        model (torch.nn.Module):
            The variational autoencoder.
        data (nx.Graph):
            The graph data.
        output_dir (str):
            The output directory for saving plots.
        enable_tracking (bool):
            Whether experiment tracking is enabled.

    Returns:
        str:
            The saved visualisation.
    """
    embeddings = model.encode(data.expression, data.edge_index).detach().cpu().numpy()
    transformer = PCA(n_components=2)
    emb_transformed = pd.DataFrame(
        transformer.fit_transform(embeddings), columns=["x", "y"]
    )
    emb_transformed.plot.scatter("x", "y")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outfile = join_path(output_dir, "graph.png")
    plt.savefig(outfile)

    if enable_tracking:
        mlflow.log_artifact(outfile)

    return outfile


######################################
# Main
######################################


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    The main entry point for the plotting pipeline.

    Args:
        config (DictConfig):
            The pipeline configuration.
    """
    EXPERIMENT_PREFIX = config["experiment"]["name"]

    DATA_DIR = config["dir"]["data_dir"]
    OUT_DIR = config["dir"]["out_dir"]

    GRN_NAME = config["grn"]["input_dir"]

    DB_HOST = config["db"]["host"]
    DB_NAME = config["db"]["name"]
    DB_USERNAME = config["db"]["username"]
    DB_PASSWORD = config["db"]["password"]

    NUM_VAL = config["gnn"]["num_val"]
    NUM_TEST = config["gnn"]["num_test"]
    HIDDEN_CHANNELS = config["gnn"]["hidden_channels"]
    OUT_CHANNELS = config["gnn"]["out_channels"]
    LR = config["gnn"]["lr"]
    N_EPOCHS = config["gnn"]["n_epochs"]
    N_LAYERS = config["gnn"]["n_layers"]
    NORMALIZE = config["gnn"]["normalize"]
    BIAS = config["gnn"]["bias"]
    AGGR = config["gnn"]["aggr"]

    TRACKING_URI = config["experiment_tracking"]["tracking_uri"]
    ENABLE_TRACKING = config["experiment_tracking"]["enabled"]

    G = get_graph(DB_HOST, DB_NAME, DB_USERNAME, DB_PASSWORD, GRN_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = from_networkx(G)

    train_data, val_data, test_data = get_split(G, NUM_VAL, NUM_TEST, device)

    in_channels = G.expression.shape[1]
    model, optimizer, scheduler = get_model_components(
        LR,
        in_channels,
        HIDDEN_CHANNELS,
        OUT_CHANNELS,
        device,
        N_LAYERS,
        NORMALIZE,
        BIAS,
        AGGR,
    )

    if ENABLE_TRACKING:
        log_results(TRACKING_URI, EXPERIMENT_PREFIX, GRN_NAME, in_channels, config)

    model = train_model(
        model,
        train_data,
        val_data,
        test_data,
        N_EPOCHS,
        optimizer,
        device,
        ENABLE_TRACKING,
    )

    output_dir = join_path(DATA_DIR, OUT_DIR, GRN_NAME, "vae_gnn")
    view_embeddings(model, train_data, output_dir, ENABLE_TRACKING)

    if ENABLE_TRACKING:
        mlflow.end_run()


if __name__ == "__main__":
    main()

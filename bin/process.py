#!/usr/bin/env python

######################################
# Imports
######################################

import hydra
import networkx as nx
from omegaconf import DictConfig
from os.path import join as join_path
import pandas as pd
from pathlib import Path

######################################
# Functions
######################################


def process_network(
    feature_matrix: pd.DataFrame,
    edge_list: pd.DataFrame,
    from_col: str,
    to_col: str,
    len_component: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct a graph from edge list data.

    Args:
        feature_matrix (pd.DataFrame):
            The feature matrix.
        edge_list (pd.DataFrame):
            The edge list.
        from_col (str):
            The "from" column name.
        to_col (str):
            The "to" column name.
        len_component (int, optional):
            The minimum size of a subgraph to filter out. Defaults to 5.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            The processed graph as a feature matrix and edge list.
    """
    edges = edge_list.sort_values(from_col)

    G = nx.from_pandas_edgelist(edges, from_col, to_col, create_using=nx.Graph())

    for component in list(nx.connected_components(G)):
        if len(component) <= len_component:
            for node in component:
                G.remove_node(node)

    nodes = list(G.nodes)
    filtered_feature_matrix = feature_matrix[nodes]
    filtered_edge_list = nx.to_pandas_edgelist(G, source=from_col, target=to_col)
    return filtered_feature_matrix, filtered_edge_list


def log_results(
    tracking_uri: str,
    experiment_prefix: str,
    grn_name: str,
    feature_matrix: pd.DataFrame,
    edge_list: pd.DataFrame,
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
        feature_matrix (pd.DataFrame):
            The feature matrix.
        edge_list (pd.DataFrame):
            The edge list.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = f"{experiment_prefix}_process"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    mlflow.set_tag("grn", grn_name)

    mlflow.log_param("grn", grn_name)

    mlflow.log_metric("num_features", len(feature_matrix.index))
    mlflow.log_metric("num_nodes", len(feature_matrix.columns))
    mlflow.log_metric("num_edges", len(edge_list.index))

    mlflow.end_run()


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
    # Constants
    EXPERIMENT_PREFIX = config["experiment"]["name"]

    DATA_DIR = config["dir"]["data_dir"]
    PREPROCESS_DIR = config["dir"]["preprocessed_dir"]
    PROCESS_DIR = config["dir"]["processed_dir"]
    OUT_DIR = config["dir"]["out_dir"]

    GRN_NAME = config["grn"]["input_dir"]
    FEATURE_MATRIX_FILE = config["grn"]["feature_matrix"]
    EDGE_LIST_FILE = config["grn"]["edge_list"]
    FROM_COL = config["grn"]["from_col"]
    TO_COL = config["grn"]["to_col"]

    TRACKING_URI = config["experiment_tracking"]["tracking_uri"]
    ENABLE_TRACKING = config["experiment_tracking"]["enabled"]

    input_dir = join_path(DATA_DIR, PREPROCESS_DIR, GRN_NAME)
    feature_matrix = pd.read_csv(join_path(input_dir, FEATURE_MATRIX_FILE))
    edge_list = pd.read_csv(join_path(input_dir, EDGE_LIST_FILE))

    filtered_feature_matrix, filtered_edge_list = process_network(
        feature_matrix, edge_list, FROM_COL, TO_COL
    )

    output_dir = join_path(DATA_DIR, OUT_DIR, GRN_NAME, PROCESS_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filtered_feature_matrix.to_csv(join_path(output_dir, FEATURE_MATRIX_FILE))
    filtered_edge_list.to_csv(join_path(output_dir, EDGE_LIST_FILE), index=False)

    if ENABLE_TRACKING:
        log_results(
            TRACKING_URI,
            EXPERIMENT_PREFIX,
            GRN_NAME,
            filtered_feature_matrix,
            filtered_edge_list,
        )


if __name__ == "__main__":
    main()

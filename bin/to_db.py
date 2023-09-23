#!/usr/bin/env python

######################################
# Imports
######################################

from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient
import hydra
import networkx as nx
from omegaconf import DictConfig
from os.path import join as join_path
import pandas as pd


######################################
# Functions
######################################


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


def get_graph(
    feature_matrix: pd.DataFrame, edge_list: pd.DataFrame, from_col: str, to_col: str
) -> nx.Graph:
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

    Returns:
        nx.Graph:
            The graph to write to the database.
    """
    edges = edge_list.sort_values(from_col)

    G = nx.from_pandas_edgelist(edges, from_col, to_col, create_using=nx.Graph())
    node_features = feature_matrix.to_dict()
    nx.set_node_attributes(G, node_features, "expression")

    return G


def to_db(
    db_host: str,
    db_name: str,
    db_username: str,
    db_password: str,
    collection: str,
    G: nx.Graph,
) -> None:
    """
    Write the graph to the database.

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
        G (nx.Graph):
            The graph.
    """
    sys_db = ArangoClient(hosts=db_host).db(
        "_system", username=db_username, password=db_password
    )
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
    db = ArangoClient(hosts=db_host).db(
        db_name, username=db_username, password=db_password
    )

    edges_collection = f"{collection}_edges"
    for db_collection in [collection, edges_collection]:
        if db.has_collection(db_collection):
            db.delete_collection(db_collection)

    if db.has_graph(collection):
        db.delete_graph(collection)

    graph_definitions = [
        {
            "edge_collection": edges_collection,
            "from_vertex_collections": [collection],
            "to_vertex_collections": [collection],
        }
    ]

    adapter = ADBNX_Adapter(db)
    adapter.networkx_to_arangodb(collection, G, graph_definitions)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    The main entry point for the plotting pipeline.

    Args:
        config (DictConfig):
            The pipeline configuration.
    """
    # Constants
    DATA_DIR = config["dir"]["data_dir"]
    PROCESS_DIR = config["dir"]["processed_dir"]
    OUT_DIR = config["dir"]["out_dir"]

    GRN_NAME = config["grn"]["input_dir"]
    FEATURE_MATRIX_FILE = config["grn"]["feature_matrix"]
    EDGE_LIST_FILE = config["grn"]["edge_list"]
    FROM_COL = config["grn"]["from_col"]
    TO_COL = config["grn"]["to_col"]

    DB_HOST = config["db"]["host"]
    DB_NAME = config["db"]["name"]
    DB_USERNAME = config["db"]["username"]
    DB_PASSWORD = config["db"]["password"]

    input_dir = join_path(DATA_DIR, OUT_DIR, GRN_NAME, PROCESS_DIR)
    feature_matrix = pd.read_csv(join_path(input_dir, FEATURE_MATRIX_FILE))
    edge_list = pd.read_csv(join_path(input_dir, EDGE_LIST_FILE))

    G = get_graph(feature_matrix, edge_list, FROM_COL, TO_COL)
    to_db(DB_HOST, DB_NAME, DB_USERNAME, DB_PASSWORD, GRN_NAME, G)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

######################################
# Imports
######################################

# External
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from os.path import join as join_path
import pandas as pd

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
    INPUT_DIR = config["preprocess"]["input_dir"]
    FEATURE_MATRIX_FILE = config["preprocess"]["feature_matrix"]
    EDGE_LIST_FILE = config["preprocess"]["edge_list"]

    TRACKING_URI = config["experiment_tracking"]["tracking_uri"]
    ENABLE_TRACKING = config["experiment_tracking"]["enabled"]

    print(FEATURE_MATRIX_FILE)

if __name__ == "__main__":
    main()
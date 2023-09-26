# Nextflow Graph Machine Learning

[![Validate Pipeline](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/validation.yml/badge.svg)](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/validation.yml) [![Generate Documentation](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/docs.yml/badge.svg)](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/docs.yml) [![pages-build-deployment](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/pages/pages-build-deployment)

Website: [Nextflow Graph Machine Learning](https://jbris.github.io/nextflow-graph-machine-learning/)

*A Nextflow pipeline demonstrating how to train graph neural networks for gene regulatory network reconstruction using DREAM5 data.*

# Table of contents

- [Nextflow Graph Machine Learning](#nextflow-graph-machine-learning)
- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [The Nextflow pipeline](#the-nextflow-pipeline)
- [Python Environment](#python-environment)
  - [MLOps](#mlops)
- [ArangoDB](#arangodb)

# Introduction

The purpose of this project is to provide a simple demonstration of how to construct a Nextflow pipeline, with MLOps integration, for performing gene regulatory network (GRN) reconstruction using graph neural networks (GNNs). In practice, GRN reconstruction is an unsupervised link prediction problem.

[For developing GNNs, we use PyTorch Geometric.](https://pytorch-geometric.readthedocs.io/en/latest/)

# The Nextflow pipeline

[Nextflow has been included to orchestrate the GRN reconstruction pipeline.](https://www.nextflow.io/)

The pipeline is composed of the following steps:

1. Exploratory data analysis: View the GRN and calculate some summary statistics.
2. Processing: Process the graph feature matrix and edge list. Remove the disconnected subgraph.
3. ArangoDB Importing: Import the graph into ArangoDB.
4. GNN training: Train a GNN using SAGE convolutional layers.
5. GNN training: Train a variational autoencoder GNN, and save the neural embeddings.

# Python Environment

[Python dependencies are specified in this requirements.txt file.](services/python/requirements.txt). 

These dependencies are installed during the build process for the following Docker image: ghcr.io/jbris/nextflow-graph-machine-learning:1.0.0

Execute the following command to pull the image: *docker pull ghcr.io/jbris/nextflow-graph-machine-learning:1.0.0*

## MLOps

* [A Docker compose file has been provided to launch an MLOps stack.](docker-compose.yml)
* [See the .env file for Docker environment variables.](.env)
* [The docker_up.sh script can be executed to launch the Docker services.](scripts/docker_up.sh)
* [DVC is included for data version control.](https://dvc.org/)
* [MLFlow is available for experiment tracking.](https://mlflow.org/)
* [MinIO is available for storing experiment artifacts.](https://min.io/)

# ArangoDB

[This pipeline provides a simple demonstration for saving and retrieving graph data to ArangoDB, combined with NetworkX usage and integration.](https://docs.arangodb.com/3.11/data-science/adapters/arangodb-networkx-adapter/) 

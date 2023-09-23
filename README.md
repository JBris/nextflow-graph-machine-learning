# Nextflow Graph Machine Learning

[![Validate Pipeline](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/validation.yml/badge.svg)](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/validation.yml) [![Generate Documentation](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/docs.yml/badge.svg)](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/docs.yml) [![pages-build-deployment](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/JBris/nextflow-graph-machine-learning/actions/workflows/pages/pages-build-deployment)

Website: [Nextflow Graph Machine Learning](https://jbris.github.io/nextflow-graph-machine-learning/)

*A Nextflow pipeline demonstrating how to train graph neural networks for gene regulatory network reconstruction using DREAM5 data.*

# Table of contents

- [Nextflow Graph Machine Learning](#nextflow-graph-machine-learning)
- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [The pipeline](#the-pipeline)

# Introduction

The purpose of this project is to provide a simple demonstration of how to construct a Nextflow pipeline, with MLOps integration, for performing gene regulatory network (GRN) reconstruction via graph neural networks (GNNs). We then demonstrate how to extend the implementation of the GNN to perform uncertainty quantification using deep kernel learning (DKL).

# The pipeline

The pipeline is composed of the following steps:

1. Exploratory data analysis: View the GRN and calculate some summary statistics.
2. Processing: Process the graph feature matrix and edge list. Remove the disconnected subgraph.
3. ArangoDB Importing: Import the graph into ArangoDB.

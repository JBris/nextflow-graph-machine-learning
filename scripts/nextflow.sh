#!/usr/bin/env bash

. .env

docker compose run --rm nextflow \
    nextflow run gnn_pipeline.nf -profile standard,docker

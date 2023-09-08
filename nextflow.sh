#!/usr/bin/env bash

. .env

docker compose run nextflow cd nf && \
    nextflow "$@"

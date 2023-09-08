#!/usr/bin/env bash

. .env

docker compose run --rm nextflow cd nf && \
    nextflow "$@"

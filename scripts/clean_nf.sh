#!/usr/bin/env bash

rm -f pipeline_dag.html
rm -f pipeline_trace.txt
rm -f report-*.html
rm -f trace-*.txt
rm -f .nextflow.log*

rm -rf work && \
    mkdir work

rm -rf out && \
    mkdir out

rm -rf outputs && \
    mkdir outputs
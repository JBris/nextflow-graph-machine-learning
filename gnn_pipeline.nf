#!/usr/bin/env nextflow

// ##################################################################
// Imports
// ##################################################################

include { dvc_repro as dvc } from './modules/mlops.nf'

// ##################################################################
// Parameters
// ##################################################################

params.grn = "in_silico"
params.edge_list = "$projectDir/data/preprocessed/in_silico/gold_standard.csv"
params.outdir = "$projectDir/data/out"

// ##################################################################
// Workflow
// ##################################################################

workflow {
    dvc_res = dvc(params.grn, projectDir)
}
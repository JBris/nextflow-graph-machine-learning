#!/usr/bin/env nextflow

// ##################################################################
// Imports
// ##################################################################

include { dvcRepro as dvc } from './modules/mlops.nf'
include { toDb } from './modules/db.nf'

// ##################################################################
// Parameters
// ##################################################################

params.grn = "in_silico"
params.edgeList = "gold_standard.csv"
params.featureMatrix = "expression_data.csv"

// ##################################################################
// Workflow
// ##################################################################

workflow {
    processedDir = dvc(params.grn, projectDir)
    db_res = toDb(params.grn, projectDir, processedDir, params.featureMatrix, params.edgeList)
}
#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// ##################################################################
// Imports
// ##################################################################

include { toDb } from './modules/db.nf'
include { trainSAGE } from './modules/gnn.nf'
include { dvcRepro as dvc } from './modules/mlops.nf'

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
    (db_log, grn_db) = toDb(params.grn, projectDir, processedDir, params.featureMatrix, params.edgeList)
    gnn_res = trainSAGE(grn_db)
}

workflow.onComplete {
    log.info ( workflow.success ? "Training completed successfully." : "Something went wrong" )
}
#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// ##################################################################
// Imports
// ##################################################################

include { toDb } from './modules/db.nf'
include { trainSAGE as SAGE } from './modules/gnn.nf'
include { trainVAE as VAE } from './modules/gnn.nf'
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

/**
* In practice, we should parallelise the training of the GraphSAGE and VAE 
* nueral networks - which is what we'd do in an HPC environment. 
* This would also introduce async channels and observables. 
* But I'm running this on my laptop. 
*/
workflow {
    processedDir = dvc(params.grn, projectDir)
    (db_log, grn_db) = toDb(params.grn, projectDir, processedDir, params.featureMatrix, params.edgeList)
    (gnn_log, gnn_db) = SAGE(grn_db)
    VAE(gnn_db)
}

workflow.onComplete {
    log.info ( workflow.success ? "Training completed successfully." : "Something went wrong" )
}
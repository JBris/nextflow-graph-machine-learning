// ##################################################################
// Module
// ##################################################################

/*
* I feel like DVC and Nextflow don't seem to mix too well within a single pipeline. 
* This is a fairly over-engineered example.
* I guess DVC goes against Nextflow's stateless philosophy...
* Maybe it makes more sense to call a Nextflow pipeline from a DVC step. 
* Or use Airflow/Prefect + DVC instead.
*/
process dvcRepro {
    tag "Data for GRN: $grn"
    
    input:
    val grn
    path baseDir

    output: 
    val 'processed'

    script:
    """
    cd $baseDir && \
    dvc repro
    """
}
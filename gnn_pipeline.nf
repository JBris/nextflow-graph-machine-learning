params.id = "1"
params.reads = "$projectDir/rna_seq/data/ggal/gut_1.fq"
params.outdir = "out"

process INDEX {
    tag "Data for sample: $sample_id"
    publishDir params.outdir, mode:'copy'

    input:
    val sample_id
    path reads

    output:
    path 'results.txt'

    script:
    """
    pip list --format=freeze  > results.txt
    """
}

workflow {
    index_ch = INDEX(params.id, params.reads)
}
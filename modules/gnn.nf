// ##################################################################
// Module
// ##################################################################

process trainSAGE {
    tag "Gene regulatory network: $grn"

    input:
    val grn

    output: 
    path 'train_gnn.log'
    val grn_db
    
    script:
    grn_db = grn
    """
    train_gnn.py grn.input_dir="$grn" db.password=$ARANGO_ROOT_PASSWORD > train_gnn.log
    """
}
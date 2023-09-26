// ##################################################################
// Module
// ##################################################################

process toDb {
    tag "Graph DB for $grn"

    input:
    val grn
    path baseDir
    val processedDir
    val featureMatrix
    val edgeList

    output: 
    path 'to_db.log'
    val grn_db
    
    script:
    grn_db = grn
    """
    to_db.py dir.data_dir="$baseDir/data" dir.processed_dir="$processedDir" grn.input_dir="$grn" \
    grn.feature_matrix="$featureMatrix" grn.edge_list="$edgeList" db.password=$ARANGO_ROOT_PASSWORD > to_db.log
    """
}
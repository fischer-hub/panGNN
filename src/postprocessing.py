from src.setup import log
import os


def write_groups_file(dataset, binary_prediction):

    edge_index = dataset.edge_index
    gene_ids_lst = dataset.gene_ids_lst
    sets = []

    for label, origin_id, target_id in zip(binary_prediction.tolist(), edge_index[0].tolist(), edge_index[1].tolist()):

        # there was an edge predicted for these nodes
        if label:
            for node_set in sets:
                # sets list is still empty, we initialize it with the first set
                if not sets:
                    sets.append({origin_id, target_id})
                # the current nodes are part of a set so we found a connected component and add the new IDs to its set
                if node_set & {origin_id, target_id}:
                    node_set.update({origin_id, target_id})
                    break
            # we have checked all sets in the list but we haven't found an intersection, we have to create a new set for
            # these nodes since they are not connected to any previous group of nodes 
            sets.append({origin_id, target_id})
        # there was no edge predicted between these nodes
        else:
            continue


    log.info("Writing output file 'holiest_of_all_tables.csv' ..")

    with open(os.path.join('data', 'holiest_of_all_tables.csv'), 'w') as group_output_file:
        
        for idx, set in enumerate(sets):
            group_output_file.write(f"group_{idx}, {', '.join([gene_ids_lst[gene_id] for gene_id in list(set)])}")


def write_stats_csv(stats, path = os.path.join('stats.csv')):

    line = ''

    for value in stats.values():

        if value is None or not value:
            value = 'NA'

        line += f'{value}, '
    
    line = line[:-1]

    with open(path, 'a') as file:
        file.write(f'\n{line}')

    header = ''

    for key in stats.keys():

        if key is None or not key:
            key = 'NA'

        header += f'{key}, '
    
    header = header[:-1]
    print(header)



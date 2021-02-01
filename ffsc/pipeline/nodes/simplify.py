import logging, sys, json, time
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import pandas as pd
from shapely import geometry, wkt, ops
from tqdm import tqdm
tqdm.pandas()

import networkx as nx


def simplify_edges(node_df, edge_df, keep_nodes):
    print ('node df')
    print (node_df)
    print ('keep nodes')
    print (keep_nodes)
    print ('edge df')
    print (edge_df)
    asset_name = node_df.iloc[0]['NODES'].split('_')[0]
    logger = logging.getLogger(f'Simplify_edges_{asset_name}')
    
    orig_node_len = len(node_df)
    orig_edge_len = len(edge_df)
    
    logger.info('Making Graph object')
    G = nx.Graph()
    G.add_edges_from([(r[1],r[2],{'DISTANCE':r[3]})for r in edge_df.to_records()])

    # logger.info(f'Dumping graph object.. {time.time()-tic:.2f}') ... lol no way, OoM error
    # pickle.dump(G, open(os.path.join(os.getcwd(),'results_backup','edge_graph.pkl'),'wb'))

    # get edges where degree==2
    logger.info(f'Getting edges where degree')
    degrees = {node:val for (node, val) in G.degree() if val==2}
    
    

    # remove nodes to keep
    simplify_nodes = set(degrees.keys()) - set(keep_nodes['KEEP_NODES'].values.tolist())

    # get subgraph node lists
    logger.info(f'getting sugraph nodes')
    subgraph_nodes = [g for g in nx.connected_components(G.subgraph(list(simplify_nodes)))]

    
    logger.info(f'new edges and drop nodes')
    ## for each subgraph, simplify -> maybe parallelise
    drop_nodes = []
    new_edges = []
    logger.info(f'n sugraph nodes {len(subgraph_nodes)}')

    for g in tqdm(subgraph_nodes):

        subgraph = G.subgraph(g)

        if len(subgraph.nodes)>2:

            # get endpoints of component
            end_pts = {node:val for (node, val) in subgraph.degree() if val==1}
            
            if len(list(end_pts.keys()))==2:

                # get nodes to drop
                drop_nodes_subgraph = list(g-set(list(end_pts.keys())))

                # create new edge
                new_dist = sum([e[2]['DISTANCE'] for e in subgraph.edges(data=True)])
                #print ('new edge',list(end_pts.keys())[0],list(end_pts.keys())[1])
                new_edge = {
                    'START':list(end_pts.keys())[0],
                    'END':list(end_pts.keys())[1],
                    'DISTANCE':new_dist,
                }

                # add to list collection
                drop_nodes += drop_nodes_subgraph
                new_edges += [new_edge]
            else:
                print ('ERROR!', len(end_pts))
                print ('end_pts',end_pts)
                print (subgraph.edges())
                print (subgraph.degree())
                
                
                
                
    logger.info(f'New edges: {len(new_edges)}')
    new_edges = pd.DataFrame(new_edges)

    
    drop_edge_index = edge_df['START'].isin(drop_nodes).values + edge_df['END'].isin(drop_nodes).values
      
    edge_df.drop(index=edge_df[drop_edge_index].index, inplace=True)
    edge_df = edge_df.append(new_edges, ignore_index=True)#

    node_df = node_df[~node_df['NODES'].isin(drop_nodes)]
    
    logger.info(f'Simplified! Reduced nodes from {orig_node_len} to {len(node_df)}; reduced edges from {orig_edge_len} to {len(edge_df)}')

    return node_df, edge_df


import networkx as nx

G = nx.Graph()
G.add_nodes_from(['A','B','C','D','E','F','G','H','I','J','K'])

G.add_edges_from([
    ('A','B',{'z':5}),
    ('B','C',{'z':10}),
    ('C','D',{'z':4}),
    ('D','E',{'z':5}),
    ('E','H',{'z':3}),
    ('D','G',{'z':6}),
    ('G','I',{'z':12}),
    ('J','I',{'z':11}),
    ('C','F',{'z':2}),
    ('F','J',{'z':3}),
    ('B','K',{'z':7}),
    ('J','K',{'z':9}),
    ('H','I',{'z':7}),
    
])

G_orig = G.copy()

def simplify(G, start_node, end_node):
    
    # get degree-2 nodes
    nodes = [n for n,d in dict(G.degree).items() if ((d==2) and  (n not in [start_node,end_node]))]
    print ('d2 nodes',nodes)

    # get component subgraphs based on 
    subgraph_components = [c for c in nx.connected_components(G.subgraph(nodes))]
    
    
    for cc in subgraph_components:
        print ('cc:',cc)
        
        #if (start_node not in cc) and (end_node not in cc):

        
        # for each subgraph, get the edges
        edges = list(G.edges(cc, data=True))

        # get the start and end nodes
        startend_nodes = list(set([el for tup in [(e[0],e[1]) for e in edges] for el in tup])-cc)

        # get series impedence
        z_eq = sum([e[2]['z'] for e in edges])
        
        # add the equivalent edge to the graph or get parallel equivalent
        if G.has_edge(startend_nodes[0], startend_nodes[1]):
            # parallel addition
            z_eq = 1/(1/z_eq + 1/G.get_edge_data(startend_nodes[0],startend_nodes[1])['z'])
            G.get_edge_data(startend_nodes[0],startend_nodes[1])['z']=z_eq  
            print ('update edge',startend_nodes[0], startend_nodes[1],f'z={z_eq}')              
        else:
            # new edge
            G.add_edge(startend_nodes[0], startend_nodes[1],z=z_eq)
            print ('new edge',startend_nodes[0], startend_nodes[1],f'z={z_eq}')

        # remove edges and nodes
        G.remove_nodes_from(cc) # also removes edges
        
    return G     

# run this loop uptil only the two main nodes are left
for ii in range(4):
	print (f'run {ii}')
	G = simplify(G,'A','K')
	for e in G.edges(data=True):
		print (e)


# performance vs time
# 
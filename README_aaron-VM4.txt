I decided to push the entirety of the changes I made to the repository up to a new branch. Quite a large portion of it is not super useful and I wanted to highlight the files that were of use (mentioned below). The less useful files will likely either be related to NetworkX or using impedance as edge weighting, rather than flow.


Current Workflow [./ indicates global-fossil-fuel-chain as root repository]

1) ./notebook/Aaron_edge_conversion_coal_uw.py
- this script converts all the edges from coordinates to node pairs (with no weighting)
- node pairs are saved in graph_coal_uw.txt
- this was originally in notebook form (./notebook/Aaron_edge_conversion_coal_uw.ipynb), but due to the fact it takes ~24 hrs to run I converted it into a script

2) ./notebook/Aaron_edge_conversion_coal_flow.ipynb
- this notebook takes in the graph_coal_uw.txt and adds the flow weights to each edge
- node pairs with weights are saved into graph_coal_flow_weighted.txt
- this step could have been done in conjuction with step 1, but step 1 allows different weights to be appended to the unweighted nodes

graph_coal_flow_weighted.txt is moved to ./DirectedLouvain/graphs/graph_coal_flow_weighted.txt

3) move to repository ./DirectedLouvain/output & run commands for Directed Louvain community detection
- ../bin/convert -i ../graph/graph_coal_flow_weighted.txt -o graph_coal_flow_weighted.bin -w graph_coal_flow_weighted.weights
    - creates graph_coal_flow_weighted.bin & graph_coal_flow_weighted.weights
- ../bin/community graph_coal_flow_weighted.bin -l -1 -w graph_coal_flow_weighted.weights > graph_coal_flow_weighted.tree
    - runs community detection analysis, creating graph_coal_flow_weighted.tree
- ../bin/hierarchy graph_coal_flow_weighted.tree -l 0 > graph_coal_flow_node2comm_level0
    - creates file of the last level of the tree, contains pairs of node and associated community

4a) ./notebook/Aaron_edge_conversion_coal_flow_info.ipynb
- reads in graph_coal_flow_node2comm_level0 and analyses communities formed

4b) ./notebook/Aaron_edge_conversion_coal_flow_large.ipynb
- reads in graph_coal_flow_node2comm_level0 and plots large communities (images saved)

4c) ./notebook/Aaron_edge_conversion_coal_flow_small.ipynb
- reads in graph_coal_flow_node2comm_level0 and plots small communities (images saved)

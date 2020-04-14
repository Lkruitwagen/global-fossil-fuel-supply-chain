import networkx as nx
import numpy as np
from pulp import *

class PipeEquilibrium:

    def __init__(self, G):
        self.G = G

    def equilibrium_min(self):
        """
        Solve for flow in the network assuming all sources have uniform pressure
        - super source/sink
        - solve for P
        - then solve Q
        """
        self.G_original = self.G.copy()
        
        self.G = self.equilibrium_min_W()


        


    def equilibrium_min_W(self):

        self.source_nodes = [n[0] for n in self.G.nodes(data=True) if self._bool_ss(n,True)]
        self.sink_nodes = [n[0] for n in self.G.nodes(data=True) if self._bool_ss(n,False)]
        self.other_nodes = [n for n in self.G.nodes if not ((n in self.sink_nodes) or (n in self.source_nodes))]

        ### declare the model
        self.model = LpProblem('Network fill problem',LpMinimize)


        ### Declare the variables
        # edges
        edge_Q = LpVariable.dicts('Edge_Q',G.edges,0,None,cat='Continous')

        # nodes
        node_P = LpVariable.dicts('Node_P',G.nodes,0,None,cat='Continous')

        ### Declare Constraints

        # at non-sink, non-source nodes, sum of edge flows = 0
        for n in self.other_nodes:
            self.model += lpSum([edge_Q[e] for e in self.G.in_edges(n)]) - lpSum([edge_Q[e] for e in self.G.out_edges(n)]) == 0

        # at each edge, flow = pressure drop
        for e in self.G.edges:
            self.model += node_P[e[0]] - node_P[e[1]] - self.G.edges[e]['z']*edge_Q[e] == 0

        # at sinks, flow = boundary condition
        for n in self.sink_nodes:
            self.model += lpSum([edge_Q[e] for e in self.G.in_edges(n)]) - lpSum([edge_Q[e] for e in self.G.out_edges(n)]) >= -self.G.nodes[n]['ss']

        ### Affine equations
        # excess sink pressure
        excess_P = sum([node_P[n] for n in self.sink_nodes])

        # excess sink flow
        excess_Q = sum([(sum([edge_Q[e] for e in self.G.in_edges(n)]) - sum([edge_Q[e] for e in self.G.out_edges(n)])+self.G.nodes[n]['ss']) for n in self.sink_nodes] )

        # source flows
        #source_flows = {n:(sum([edge_Q[e] for e in self.G.in_edges(n)]) - sum([edge_Q[e] for e in self.G.out_edges(n)])) for n in self.source_nodes}

        # source work
        #source_work = sum([source_flows[n]*node_P[n] for n in self.source_nodes])

        #### Objective function
        #self.model += source_work

        # alternative - minimise the pressure at the sinks
        self.model += excess_P + excess_Q

        self.model.solve()

        print (pulp.LpStatus[self.model.status])
        print (node_P)
        print (edge_Q)

        for n in node_P.keys():
            print (node_P[n].name, node_P[n].value())

        for e in edge_Q.keys():
            print (edge_Q[e].name, edge_Q[e].value())
            

    def _bool_ss(self,n,source=True):
        if not n[1]['ss']:
            return False
        if n[1]['ss']>0 and source:
            return True
        if n[1]['ss']<0 and not source:
            return True



def net1():
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
    return G

def net2():
    G = nx.DiGraph()
    G.add_nodes_from([
        ('A',{'ss':np.inf}),
        ('B',{'ss':np.inf}),
        ('C',{'ss':None}),
        ('D',{'ss':None}),
        ('E',{'ss':-10}),
        ('F',{'ss':-20})
        ])

    G.add_edges_from([
        ('A','C',{'z':2}),
        ('B','D',{'z':5}),
        ('C','D',{'z':4}),
        ('C','E',{'z':3}),
        ('D','F',{'z':6})
    ])
    return G

def net3():
    G = nx.DiGraph()
    G.add_nodes_from([
        ('A',{'ss':np.inf}),
        ('B',{'ss':np.inf}),
        ('C',{'ss':None}),
        ('E',{'ss':-10}),
        ('F',{'ss':-20})
        ])

    G.add_edges_from([
        ('A','C',{'z':2}),
        ('B','C',{'z':5}),
        ('C','E',{'z':3}),
        ('C','F',{'z':6})
    ])
    return G

def net4():
    G = nx.DiGraph()
    G.add_nodes_from([
        ('A',{'ss':np.inf}),
        ('B',{'ss':np.inf}),
        ('C',{'ss':np.inf}),
        ('D',{'ss':None}),
        ('E',{'ss':None}),
        ('F',{'ss':None}),
        ('G',{'ss':None}),
        ('H',{'ss':-10}),
        ('I',{'ss':-25}),
        ('J',{'ss':-15})])

    G.add_edges_from([
        ('A','D',{'z':3}),
        ('D','H',{'z':7}),
        ('B','E',{'z':6}),
        ('E','D',{'z':2}),
        ('E','G',{'z':5}),
        ('G','I',{'z':11}),
        ('F','G',{'z':6}),
        ('C','F',{'z':3}),
        ('F','J',{'z':9}),
        ])
    return G


if __name__=="__main__":
    
    G = net2()


    print ('nodes',G.nodes(data=True))
    
    pe=PipeEquilibrium(G)
    pe.equilibrium_min_W()
    #pe._Z_equivalent('A','K')
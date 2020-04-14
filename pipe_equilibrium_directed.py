import networkx as nx
import numpy as np
from scipy.optimize import root

class PipeEquilibrium:

    def __init__(self, G):
        self.G = G

    def equilibrium_P_uniform(self):
        """
        Solve for flow in the network assuming all sources have uniform pressure
        - super source/sink
        - solve for P
        - then solve Q
        """
        self.G_original = self.G.copy()
        self.G = self._add_superss()

        self.z_eq = self._Z_equivalent('supersource','supersink')
        

        self.Q_tot = self.G.nodes['supersink']['ss']
        self.P_tot = self.z_eq * -self.Q_tot
        self.G = self._add_property_unknowns()

        print ('P_tot',self.P_tot)

        ini_guess = []
        for kk,vv in self.unknowns.items():
            for kk2, vv2 in vv.items():
                ini_guess.append(vv2)

        sol = root(self._residue,ini_guess)
        print (sol.x)
        print (self.unknowns)
        


    def equilibrium_min_W(self):
        pass

    def _bool_ss(self,n,source=True):
        if not n[1]['ss']:
            return False
        if n[1]['ss']>0 and source:
            return True
        if n[1]['ss']<0 and not source:
            return True

    def _add_property_unknowns(self):

        self.unknowns = {'node_P':{},'edge_Q':{}}

        ### add known properties
        # set P for supersources/sinks
        self.G.nodes['supersink']['P'] = 0
        self.G.nodes['supersource']['P'] = self.P_tot

        # set Q for sinks

        for n in self.sink_nodes:
            self.G.edges[(n[0],'supersink')]['Q'] = -self.G.nodes[n[0]]['ss']

        ### add initial properties and guesses
        # add P -> sum Q === 0
        for n in self.G.nodes:
            if n not in ['supersink','supersource']:
                self.G.nodes[n]['P'] = self.unknowns['node_P'][n]= 0

        # add Q for edges -> dP - z*Q === 0
        for e in self.G.edges:
            if 'supersink' not in [e[0],e[1]]:
                self.G.edges[e]['Q'] = self.unknowns['edge_Q'][e]=0

        print ('unknowns',self.unknowns)

        return self.G

    def _add_superss(self):

        self.source_nodes = [n for n in self.G.nodes(data=True) if self._bool_ss(n,True)]
        self.sink_nodes = [n for n in self.G.nodes(data=True) if self._bool_ss(n,False)]

        self.G.add_node('supersource',ss=sum([n[1]['ss'] for n in self.source_nodes]))
        self.G.add_node('supersink',ss=sum([n[1]['ss'] for n in self.sink_nodes]))

        for n in self.source_nodes:
            self.G.add_edge('supersource',n[0],z=0)
        for n in self.sink_nodes:
            self.G.add_edge(n[0],'supersink',z=0)

        return self.G


    def _residue(self,try_soln=None):

        print (type(try_soln))
        if try_soln is not None:
            print (try_soln.shape)
            # reassign the solution vector to the graph components
            for ii_n, n in enumerate(self.unknowns['node_P'].keys()):
                self.G.nodes[n]['P']=try_soln[ii_n]

            len_nodes = len(self.unknowns['node_P'].keys())
            for ii_e,e in enumerate(self.unknowns['edge_Q'].keys()):
                self.G.edges[e]['Q'] = try_soln[ii_e+len_nodes]

        cost = []

        for n in self.unknowns['node_P'].keys():
            # sum(edges[Q]) === 0
            print ('residue',n,self.G.edges(n,data=True))

            flow_diff = sum([e[2]['Q'] for e in self.G.in_edges(n,data=True)]) - sum([e[2]['Q'] for e in self.G.out_edges(n,data=True)])
            cost.append(flow_diff)

        for e in self.unknowns['edge_Q'].keys():
            # dP - z*Q
            print (e)
            pressure_diff = self.G.nodes[e[0]]['P'] - self.G.nodes[e[1]]['P'] - self.G.edges[e]['z']*self.G.edges[e]['Q'] 
            cost.append(pressure_diff)

        return cost




    def _Z_equivalent(self, start_node, end_node):
        G = self.G.to_undirected()
        while len(G.edges())>1:
            G = self._Z_simplify(G,start_node,end_node)

        return G.get_edge_data(start_node,end_node)['z']


    def _Z_simplify(self,G,start_node,end_node):


        # get degree-2 nodes
        nodes = [n for n,d in dict(G.degree).items() if ((d==2) and  (n not in [start_node,end_node]))]

        # get component subgraphs based on 
        subgraph_components = [c for c in nx.connected_components(G.subgraph(nodes))]
        
        
        for cc in subgraph_components:
            print ('cc:',cc)
            
            #if (start_node not in cc) and (end_node not in cc):

            
            # for each subgraph, get the edges
            edges = list(G.edges(cc, data=True))
            print (edges)

            # get the start and end nodes
            startend_nodes = list(set([el for tup in [(e[0],e[1]) for e in edges] for el in tup])-cc)

            start_node=startend_nodes[0]
            end_node=startend_nodes[1]

            # get series impedence
            z_eq = sum([e[2]['z'] for e in edges])
            
            # add the equivalent edge to the graph or get parallel equivalent
            if G.has_edge(start_node, end_node):
                # parallel addition
                z_eq = 1/(1/z_eq + 1/G.get_edge_data(start_node,end_node)['z'])
                G.get_edge_data(start_node,end_node)['z']=z_eq  
                print ('update edge',start_node, end_node,f'z={z_eq}')              
            else:
                # new edge
                G.add_edge(start_node, end_node,z=z_eq)
                print ('new edge',start_node, end_node,f'z={z_eq}')

            # remove edges and nodes
            G.remove_nodes_from(cc) # also removes edges
            
        return G

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
        ('D',{'ss':-20}),
        ('E',{'ss':-10})
        ])

    G.add_edges_from([
        ('A','C',{'z':2}),
        ('B','C',{'z':5}),
        ('C','D',{'z':4}),
        ('C','E',{'z':3}),
    ])
    return G

def net3():
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
    pe.equilibrium_P_uniform()
    #pe._Z_equivalent('A','K')
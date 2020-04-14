import networkx as nx
import numpy as np
import time

from neomodel import StructuredNode, StructuredRel, StringProperty, RelationshipTo, RelationshipFrom, config, db, FloatProperty

config.DATABASE_URL = 'bolt://neo4j:neo4jpass@localhost:7687'
#config.ENCRYPTED_CONNECTION=False



class Edge(StructuredRel):
    #name = StringProperty(unique_index=True)
    impedance = FloatProperty(required=True) 
    capacity=FloatProperty(default=np.inf)

class Node(StructuredNode):
    name = StringProperty(unique_index=True)
    demand = FloatProperty(default=0)
    outedge = RelationshipTo('Node', 'OUT', model=Edge)
    #outedge = RelationshipFrom('Edge', 'IN')



#lefthand = Book(title='Left Hand of Darkness').save()
#ursula =  Author(name='ursula  K le guin').save()
#lefthand.author.connect(ursula)

# https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/flow/networksimplex.html#network_simplex

class PipeEquilibriumNeo4j:

    def __init__(self, G, clear_db=False):
        self.G = G

        if clear_db:
            # clear the database
            results, meta = db.cypher_query("MATCH (n) DETACH DELETE n;", {})

    def equilibrium_simplex(self):
        """
        Solve for flow in the network assuming all sources have uniform pressure
        - super source/sink
        - solve for P
        - then solve Q
        """
        self.G_original = self.G.copy()
        self.G = self._add_superss(source=True,sink=False)

        for n in self.G.nodes:

            self.G.nodes[n]['demand'] = 0 

        for n in self.sink_nodes:
            self.G.nodes[n[0]]['demand'] = -self.G.nodes[n[0]]['ss']

        self.G.nodes['supersource']['demand'] = np.sum([self.G.nodes[n[0]]['ss'] for n in self.sink_nodes])

        tic = time.time()
        null1, null2 = nx.algorithms.flow.network_simplex(self.G, demand='demand',capacity='capacity',weight='z')
        toc = time.time()-tic
        print (f'tictoc: simplex: {toc}')

        self.write_neo4j()



    def write_neo4j(self):
        for n in self.G.nodes(data=True):
            Node(name=n[0], demand=n[1]['demand']).save()

        for e in self.G.edges(data=True):
            s = Node.nodes.filter(name=e[0])[0] #source
            t = Node.nodes.filter(name=e[1])[0] #target
            s.outedge.connect(t,{'impedance':e[2]['z']}).save()




    def _bool_ss(self,n,source=True):
        if not n[1]['ss']:
            return False
        if n[1]['ss']>0 and source:
            return True
        if n[1]['ss']<0 and not source:
            return True



    def _add_superss(self,source=True,sink=True):

        self.source_nodes = [n for n in self.G.nodes(data=True) if self._bool_ss(n,True)]
        self.sink_nodes = [n for n in self.G.nodes(data=True) if self._bool_ss(n,False)]

        if source:
            self.G.add_node('supersource',ss=sum([n[1]['ss'] for n in self.source_nodes]))
        if sink:
            self.G.add_node('supersink',ss=sum([n[1]['ss'] for n in self.sink_nodes]))

        if source:
            for n in self.source_nodes:
                self.G.add_edge('supersource',n[0],z=0)
        if sink:
            for n in self.sink_nodes:
                self.G.add_edge(n[0],'supersink',z=0)

        return self.G



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

def random_graph(n_nodes, n_edges, n_sources, n_sinks):

    G = nx.DiGraph()
    G.add_nodes_from([(ii_n,{'ss':None}) for ii_n in range(n_nodes)])

    for ii_e in range(n_edges):
        s,t = np.random.choice(n_nodes, 2, replace=False)
        if not G.has_edge(s,t):
            z = int(round(20*np.random.rand()))
            G.add_edge(s,t,z=z)
            G.add_edge(t,s,z=z)

    sourcessinks = np.random.choice(n_nodes, n_sources+n_sinks, replace=False)
    sources = sourcessinks[:n_sources]
    sinks = sourcessinks[n_sources:]

    for ii in sources:#
        G.nodes[ii]['ss']=np.inf

    for ii in sinks:
        G.nodes[ii]['ss']=int(round(-50* np.random.rand()))

    #print (G.nodes)
    Gc = max(nx.connected_components(G.to_undirected()), key=len)
    G = G.subgraph(Gc).copy()

    #for n in G.nodes(data=True):
    #    print (n)
    #for e in G.edges(data=True):
    #    print (e)


    return G


if __name__=="__main__":
    
    #G = net4()

    G = random_graph(1000,1500,10,5)


    #print ('nodes',G.nodes(data=True))
    
    pe=PipeEquilibriumNeo4j(G, clear_db=True)
    pe.equilibrium_simplex()
    #pe._Z_equivalent('A','K')
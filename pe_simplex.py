import networkx as nx
import numpy as np
import time

from ortools.graph import pywrapgraph

from simplex_backup import *

import matplotlib.pyplot as plt

# https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/flow/networksimplex.html#network_simplex

# sudo apt-get install openjdk-8-jre






class PipeEquilibrium:

    def __init__(self, G):
        self.G = G

    def equilibrium_simplex(self):
        """
        Solve for flow in the network assuming all sources have uniform pressure
        - super source/sink
        - solve for P
        - then solve Q
        """
        self.G_original = self.G.copy()
        #fig, ax = plt.subplots(1,1,figsize=(12,12))


        #nx.draw_networkx_nodes(G, pos=pos, ax=ax, with_labels=True)
        #nx.draw_networkx_edges(G, pos=pos, ax=ax, width=[e[2]['z'] for e in G.edges(data=True)])
        #plt.show()

        self.G = self._add_superss(source=True,sink=False)

        for n in self.G.nodes:

            self.G.nodes[n]['demand'] = 0 

        for n in self.sink_nodes:
            self.G.nodes[n[0]]['demand'] = -self.G.nodes[n[0]]['ss']


        self.G.nodes['supersource']['demand'] = np.sum([self.G.nodes[n[0]]['ss'] for n in self.sink_nodes])

        #pos = nx.spring_layout(G, weight='z')
        #print ('doign ortoools..')
        #tic0 = time.time()
        #self.ormincostflow(self.G)

        tic1=time.time()
        print ('doing simplex... its just os fast.')
        null1, null2 = nx.network_simplex(self.G, demand='demand',capacity='capacity',weight='z')
        tic2=time.time()
        #print ('doin scaling caapacity')
        #null3, null4 = nx.capacity_scaling(self.G, demand='demand',capacity='capacity',weight='z')
        #tic3=time.time()
        print (f'tictoc: simplex: {tic2-tic1}')
        return tic2-tic1


    def ormincostflow(self,G):
        """MinCostFlow simple interface example."""

        # Define four parallel arrays: start_nodes, end_nodes, capacities, and unit costs
        # between each pair. For instance, the arc from node 0 to node 1 has a
        # capacity of 15 and a unit cost of 4.

        supplies = []
        start_nodes = []
        end_nodes = []
        capacities = []
        unit_costs = []

        node_relookup = {n:ii for ii,n in enumerate(G.nodes)}

        for n in G.nodes(data=True):
            supplies.append(int(-n[-1]['demand']))

        for e in G.edges(data=True):
            start_nodes.append(node_relookup[e[0]])
            end_nodes.append(node_relookup[e[1]])
            capacities.append(int(abs(max(supplies))))
            unit_costs.append(e[-1]['z'])



        #start_nodes = [ 0, 0,  1, 1,  1,  2, 2,  3, 4]
        #end_nodes   = [ 1, 2,  2, 3,  4,  3, 4,  4, 2]
        #capacities  = [15, 8, 20, 4, 10, 15, 4, 20, 5]
        #unit_costs  = [ 4, 4,  2, 2,  6,  1, 3,  2, 3]

        # Define an array of supplies at each node.

        #supplies = [20, 0, 0, -5, -15]


        # Instantiate a SimpleMinCostFlow solver.
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()

        # Add each arc.
        for i in range(0, len(start_nodes)):
          #print (start_nodes[i], end_nodes[i],capacities[i], unit_costs[i])
          #print (type(start_nodes[i]), type(end_nodes[i]),type(capacities[i]), type(unit_costs[i]))
          min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                      capacities[i], unit_costs[i])

        # Add node supplies.

        for i in range(0, len(supplies)):
          min_cost_flow.SetNodeSupply(i, supplies[i])


        # Find the minimum cost flow between node 0 and node 4.
        if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
          print ('soln found')
          #print('Minimum cost:', min_cost_flow.OptimalCost())
          #print('')
          #print('  Arc    Flow / Capacity  Cost')
          for i in range(min_cost_flow.NumArcs()):

            cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
            """
            print('%1s -> %1s   %3s  / %3s       %3s' % (
                min_cost_flow.Tail(i),
                min_cost_flow.Head(i),
                min_cost_flow.Flow(i),
                min_cost_flow.Capacity(i),
                cost))
            """
        else:
          print('There was an issue with the min cost flow input.')


    def equilibrium_min_W(self):
        pass

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

    #for net in [net2,net3,net4]:
    
    #    G = net()

    #G = net4()
    #print ('nodes',G.nodes(data=True))

    time_del = {}

    nodes_lists = [1000,5000,10000,50000,100000]

    for n_nodes in nodes_lists:
        time_del[n_nodes] = {}
        for ii_e, n_edges in enumerate([n_nodes*1.25, n_nodes*1.5, n_nodes*2]):
            n_sources = int(n_nodes/100)
            n_sinks = int(n_edges/100)
            G = random_graph(int(n_nodes),int(n_edges),n_sources,n_sinks)
            pe=PipeEquilibrium(G)
            time_del[n_nodes][ii_e] = pe.equilibrium_simplex()
            #pe._Z_equivalent('A','K')


    fig, axs = plt.subplots(1,3,figsize=(18,6))
    axs[0].scatter(nodes_lists, [time_del[n_nodes][0] for n_nodes in nodes_lists])
    axs[1].scatter(nodes_lists, [time_del[n_nodes][1] for n_nodes in nodes_lists])
    axs[2].scatter(nodes_lists, [time_del[n_nodes][2] for n_nodes in nodes_lists])

    plt.show()
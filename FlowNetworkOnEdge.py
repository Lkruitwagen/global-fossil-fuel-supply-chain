import networkx as nx
import collections
import numpy as np
from collections import defaultdict


class FlowNetwork:
    """Base class for flow networks"""

    def __init__(self, rho=1.0, nu=8.9 * 10 ** -4, nodes=None):
        """Initialize a flowNetwork by creating a graph from the nodes and setting rho and nu.

        Args:
            rho (float): Density
            nu (float): Dynamic viscosity
            nodes (Nodes datatype): Nodes to add directly
        """
        self.rho = rho                          # Density of the fluid
        self.nu = nu                            # Dynamic viscosity of the fluid
        self.graph = nx.DiGraph(nodes)
        self.junctionmodel = self.standard_junctionmodel
        # Specify standard variables for nodes and components
        self.nodeVariables = {'s': 0}
        self.componentVariables = {'A': 1.0, 'q': 1.0, 'pin': 1.0, 'pout': 2.0}
        self.translation = {}

    def addnodes(self, nodes):
        """Add nodes to the flowNetwork.
        :param nodes: the nodes to add
        """
        if not isinstance(nodes, collections.Iterable):
            # Add single node
            self.graph.add_node(nodes)
        elif isinstance(nodes, tuple):
            # Add nodes with data
            if not isinstance(nodes[0], collections.Iterable):
                # Single node with data
                self.graph.add_node(*nodes)
            else:
                # List of nodes with same date
                self.graph.add_nodes_from(nodes[0], **nodes[1])
        else:
            # Add any list of nodes
            self.graph.add_nodes_from(nodes)

    def addcomponents(self, edges):
        """Add components to the flowNetwork.

        Args:
            edges (multiple): the components to add
        """
        if isinstance(edges, tuple):
            if isinstance(edges[0], list):
                # Add multiple edges with same data
                self.graph.add_edges_from(edges[0], **edges[1])
            else:
                # Add single edge
                self.graph.add_edge(*edges)
        else:
            # Add multiple edges with or without own data
            self.graph.add_edges_from(edges)

    def setJunction(self, nodes):
        """Set a specific node to be a junction

        Args:
            nodes (String):
        """
        for node in list(nodes):
            self.addnodes((node, {'junction': True, 's': 0}))

    def getJunction(self, node, vec):
        """ Get the junction type as used by the junction model for the given node.

        Args:
            node (node type): The node to get the junction for
            vec (object): The variable vector
        """
        junction = {'in': [], 'out': []}
        for component in self.graph.in_edges_iter(node[0]):
            direction = 'in'
            q = self.getval((component[0], component[1]), vec, 'q')
            if q < 0:
                direction = 'out'
                q *= -1
            junction[direction].append({'theta': self.getval((component[0], component[1]), vec, 'thetaout'),
                                        'p': self.getval((component[0], component[1]), vec, 'pout'),
                                        'q': q,
                                        'A': self.getval((component[0], component[1]), vec, 'A')})

        for component in self.graph.out_edges_iter(node[0]):
            direction = 'out'
            q = self.getval((component[0], component[1]), vec, 'q')
            if q < 0:
                direction = 'in'
                q *= -1
            junction[direction].append({'theta': self.getval((component[0], component[1]), vec, 'thetain'),
                                        'p': self.getval((component[0], component[1]), vec, 'pin'),
                                        'q': q,
                                        'A': self.getval((component[0], component[1]), vec, 'A')})
        return junction

    def getunknowns(self, vector=False):
        """Find the unknowns of the network and create a vector with initial values
        :return: unknown dict, (solution vector)

        Args:
            vector (bool): should the solution factory be created?
        """
        unknown = {'nodes': defaultdict(dict), 'components': defaultdict(dict)}
        vec = []
        i = 0
        for node in self.getnodes(True):
            # For every node check if all keys have a value, if not create a
            # vector entry
            for key, value in self.nodeVariables.iteritems():
                if key not in node[1]:
                    unknown['nodes'][node[0]][key] = i
                    vec.append(value)
                    i += 1

        for component in self.getcomponents(True):
            # For every component check if all keys have a value, if not create a
            # vector entry
            for key, value in self.componentVariables.iteritems():
                if key not in component[2]:
                    unknown['components'][(component[0], component[1])][key] = i
                    vec.append(value)
                    i += 1
        self.translation = unknown
        if vector:
            return unknown, vec
        return unknown

    def setrho(self, rho):
        """ Set the fluid density

        Args:
            rho (float): fluid density
        """
        self.rho = rho

    def setnu(self, nu):
        """ Set the fluid dynamic viscosity

        Args:
            nu (float): Fluid dynamic viscosity
        """
        self.nu = nu

    """ Cost Functions"""

    def residue(self, vec=None):
        """
        The root function of the flowNetwork.

        Args:
            vec (list): the solution vector
        """
        cost = []
        for node in self.getnodes(True):
            # Conservation of mass for every node.
            # add source
            masseq = self.getval(node[0], vec, 's')
            if 'junction' in node[1]:
                junceq = self.callJunctionModel(node, vec)
            else:
                junceq = self.noJunctionResidueModel(node, vec)
            for component in self.graph.in_edges_iter(node[0]):
                # add incoming mass flows
                masseq += self.getval((component[0], component[1]), vec, 'q')
            for component in self.graph.out_edges_iter(node[0]):
                # subtract outgoing mass flows
                masseq -= self.getval((component[0], component[1]), vec, 'q')
            cost.append(masseq)
            cost.extend(junceq)

        for component in self.getcomponents(True, False):
            # Pressuredrop over every component
            # calculate pressure drop (k*(q/A)^2)

            f = 1.0 if 'f' not in component[2] else component[2]['f']
            l = 1.0 if 'l' not in component[2] else component[2]['l']
            cost.append(FlowNetwork.standard_pipemodel(
                pin=self.getval((component[0], component[1]), vec, 'pin'),
                pout=self.getval((component[0], component[1]), vec, 'pout'),
                m=self.getval((component[0], component[1]), vec, 'q'),
                A=self.getval((component[0], component[1]), vec, 'A'),
                l=l,
                rho=self.rho,
                nu=self.nu))
        return cost

    def noJunctionResidueModel(self, node, vec):
        """ Calculate the residue in case of no junction, all pressures should be the same.

        Args:
            node (node type): The junction node
            vec (list): The solution vector
        """
        ps = []
        for component in self.graph.in_edges_iter(node[0]):
            # add incoming mass flows
            ps.append(self.getval((component[0], component[1]), vec, 'pout'))
        for component in self.graph.out_edges_iter(node[0]):
            # subtract outgoing mass flows
            ps.append(self.getval((component[0], component[1]), vec, 'pin'))
        eq = []
        for i in range(len(ps) - 1):
            eq.append(ps[i] - ps[i + 1])
        return eq

    def callJunctionModel(self, node, vec):
        """ Call the set junction model and return residue

        Args:
            node (node type): The junction node
            vec (list): The solution vector
        """
        return self.junctionmodel(self.getJunction(node, vec), self.rho)

    @staticmethod
    def standard_junctionmodel(junction, rho):
        """ Calculate the residue of the function according to the new model.

        Args:
            junction (junction type): The junction
            rho (float): Fluid density
        """
        eqs = []
        M = 0
        for j in junction['out']:
            M += j['q']

        for i in junction['in']:
            sol = i['p']
            for j in junction['out']:
                uj = j['q'] / (j['A'] * rho)
                q = j['q'] / i['q']
                psi = i['A'] / j['A']
                theta = (np.abs(i['theta'] - j['theta']) % 180) * np.pi / 180.0
                sol -= j['q'] / M * (j['p'] + FlowNetwork.C(uj, q, psi, rho, theta) * rho * uj ** 2)
            eqs.append(sol)

        M = 0
        for i in junction['in']:
            M += i['q']

        for j in junction['out']:
            sol = -1 * j['p']
            for i in junction['in']:
                uj = j['q'] / (j['A'] * rho)
                q = j['q'] / i['q']
                psi = i['A'] / j['A']
                theta = (np.abs(i['theta'] - j['theta']) % 180) * np.pi / 180.0
                sol += i['q'] / M * (i['p'] - FlowNetwork.C(uj, q, psi, rho, theta) * rho * uj ** 2)
            eqs.append(sol)

        # return the equations
        return eqs

    @staticmethod
    def C(uj, q, psi, rho, theta):
        """ Calculate the pressure difference in a junction

        Args:
            uj (float): flow speed of fluid in outgoing pipe
            q (float): massflow ratio
            psi (float): area ratio between pipes
            rho (float): density of the fluid
            theta (float): the angle between the pipes
        """
        K = q ** 2 * psi ** 2 + 1.0 - 2.0 * q * psi * np.cos(3 / 4.0 * (np.pi - theta))
        return ((K - 1) / (q * psi) ** 2 + 1) * rho * uj ** 2

    @staticmethod
    def standard_pipemodel(pin, pout, m, A, l, rho, nu):
        """ Calculate the pressure difference in a pipe

        Args:
            pin (float): pressure at ingoing pipe
            pout (float): pressure at outgoing pipe
            m (float): massflow through pipe
            A (float): crossectional area of pipe
            l (float): length of pipe
            rho (float): density of the fluid
            nu (float): dynamic viscosity of fluid
        """
        dp = pin - pout
        if m < 0:
            dp *= -1.0
        return dp - m *nu* l / (rho * A ** 2 )

    """ HELPER FUNCTIONS """

    def getnodes(self, data=False):
        return self.graph.nodes(data=data)

    def getcomponents(self, data=False, node_data=False):
        """
        get all components of the network
        :param data: Need data from components
        :param node_data: Need data from nodes
        :return: Components (with data)
        """
        components = self.graph.edges(data=data)
        if node_data:
            for n, component in enumerate(components):
                components[n] = ((component[0], self.graph.node[component[0]]),
                                 (component[1], self.graph.node[component[1]]), component[2])
        return components

    def getval(self, obj, vec, key):
        """
        Get value if in graph otherwise get from solution
        :return: the data asked for

        Args:
            obj (object type): either the node or the component
            vec (list): Solution vector
            key (string): variable key

        """
        if not isinstance(obj, tuple):
            if key in self.graph.node[obj]:
                return self.graph.node[obj][key]
            return vec[self.translation['nodes'][obj][key]]

        (a, b) = obj
        if key in self.graph[a][b]:
            return self.graph[a][b][key]
        return vec[self.translation['components'][obj][key]]

    def info(self, nodes=False, edges=False):
        """ Print some general info about the network.

        Args:
            edges (bool): print edges
            nodes (bool): print nodes
        """
        print(nx.info(self.graph))
        if nodes:
            print(self.graph.edges(data=True))
        if edges:
            print(self.graph.nodes(data=True))

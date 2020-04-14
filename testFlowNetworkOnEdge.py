from FlowNetworkOnEdge import *
from scipy.optimize import root
from itertools import combinations


def sameElements(l1, l2):
    return all([i in l1 for i in l2]) and all([i in l2 for i in l1])


def test_flowNetwork_new_empty():
    N = FlowNetwork()
    assert isinstance(N.graph, nx.DiGraph)


def test_flowNetwork_add_nodes():
    N = FlowNetwork()
    nodes = 1
    N.addnodes(nodes)
    assert N.graph.nodes() == [1]
    nodes = range(10)
    N.addnodes(nodes)
    assert N.graph.nodes() == range(10)


def test_flowNetwork_add_nodes_data():
    'Single node with data'
    N = FlowNetwork()
    nodes = (1, {'p': 1})
    N.addnodes(nodes)
    assert N.graph.nodes(data=True) == [nodes]

    'Multiple nodes with same data'
    N = FlowNetwork()
    nodes = (range(10), {'p': 1, 'q': 2})
    N.addnodes(nodes)
    assert N.graph.nodes(data=True) == [(i, {'p': 1, 'q': 2}) for i in range(10)]

    'Multiple nodes with different data'
    N = FlowNetwork()
    nodes = [(i, {'p': i ** 2, 'q': i + 1}) for i in range(10)]
    N.addnodes(nodes)
    assert N.graph.nodes(data=True) == nodes


def test_flowNetwork_add_components():
    'Add one component'
    N = FlowNetwork()
    edges = (0, 1)
    N.addcomponents(edges)
    assert sameElements(N.graph.edges(), [edges])

    'Add multiple components'
    nodes = range(10)
    edges = list(combinations(nodes, 2))
    N = FlowNetwork()
    N.addcomponents(edges)
    assert sameElements(N.graph.edges(), edges)


def test_flowNetwork_add_components_data():
    'Add one component with data'
    N = FlowNetwork()
    edges = (0, 1, {'q': 1})
    N.addcomponents(edges)
    assert sameElements(N.graph.edges(data=True), [edges])

    'Multiple nodes with same data'
    nodes = range(10)
    edges = list(combinations(nodes, 2))
    N = FlowNetwork()
    N.addcomponents((edges, {'q': 1}))
    assert sameElements(N.graph.edges(data=True), [(e[0], e[1], {'q': 1}) for e in edges])

    'Multiple nodes with different data'
    nodes = range(10)
    edges = list(combinations(nodes, 2))
    edges = [(e[0], e[1], {'q': e[1]}) for e in edges]
    N = FlowNetwork()
    N.addcomponents(edges)
    assert sameElements(N.graph.edges(data=True), edges)


def test_flowNetwork_residue():
    'Simple branch'
    N = FlowNetwork();
    N.addnodes((0, {'s': 3}));
    N.addnodes((1, {'s': -3}));
    N.addcomponents((0, 1, {'q': 3, 'A': 3, 'pin': 3, 'pout': 1}));
    N.info(True, True);
    (_, x) = N.getunknowns(True);
    assert N.residue() == [0, 0, 1.9997033333333334];
    'Create a simple T with s=-3 and q=1.5 for all branches'
    N = FlowNetwork();
    N.addnodes((range(3), {'s': 3}));
    N.addnodes((3, {'s': -2}));
    N.addcomponents(([(i, 3) for i in range(3)], {'q': 3, 'k': 1, 'A': 1, 'pin': 1, 'pout': -1}));
    N.info(True, True);
    (_, x) = N.getunknowns(True);
    assert N.residue() == [0, 0, 0, 7, 0, 0, 1.99733, 1.99733, 1.99733];


def test_flowNetwork_getunknowns():
    'Random nodes'
    N = FlowNetwork();
    N.addnodes((0));
    N.addnodes((1, {'s': -3}));
    N.addnodes((2, {'s': 1}));
    assert N.getunknowns() == {'nodes':
                                   {0: {'s': 0}},
                               'components': {}};

    'Branch'
    N = FlowNetwork();
    N.addnodes((0, {'p': 3}));
    N.addnodes((1, {'s': -3}));
    N.addcomponents((0, 1, {'q': 2}));
    assert N.getunknowns() == {'nodes':
                                   {0: {'s': 0}}, 'components': {(0, 1): {'A': 1, 'pin': 3, 'pout': 2}}};


def test_flowNetwork_costFunction_with_unknowns():
    'Simple branch'
    N = FlowNetwork();
    N.addnodes((0));
    N.addnodes((1, {'s': -3}));
    N.addcomponents((0, 1, {'A': 3, 'pin': 3}));

    (dic, x) = N.getunknowns(True);
    print dic
    print x;
    print N.residue(x)
    assert N.residue(x) == [-1.0, -2.0, 0.9999011111111111];
    'residual'

    'solve it'
    sol = root(N.residue, x)
    assert list(sol.x) == [3.0, 3.0, 2.9997033333333332];


def test_flowNetwork_residue_setjunction():
    N = FlowNetwork()
    N.setJunction('a')
    N.addnodes((0, {'p': 3}))
    N.addnodes((1, {'s': -3}))
    N.addnodes((2, {'s': 1}))
    N.addcomponents([(1, 'a', {'thetaout': 120}), (2, 'a', {'pin':1,'thetaout': 80}), ( 'a', 0, {'pout':3,'thetain': 220})])
    N.info(True, True)
    (dict, vec) = N.getunknowns(True)
    print dict
    print N.residue(vec)
    assert N.residue(vec) == [1.0, 2.2204460492503131e-16, 0.73205080756887719, 0.36602540378443871, 1.0, 0.0, -4.0, -2.00089, -1.00089, -1.00089]


def test_flowNetwork_getJunction():
    N = FlowNetwork()
    N.setJunction('a')
    N.addnodes((0, {'p': 3}))
    N.addnodes((1, {'s': -3}))
    N.addnodes((2, {'s': 1, 'p': 1}))
    N.addcomponents([(1, 'a', {'thetaout': 30}), (2, 'a', {'thetaout': 180}), ('a', 0, {'thetain': 100})])
    N.info(True, True)
    (dict, vec) = N.getunknowns(True)
    print dict
    assert N.getJunction('a', vec) == {
        'in': [{'A': 1, 'p': 2, 'q': 1, 'theta': 30}, {'A': 1, 'p': 2, 'q': 1, 'theta': 180}],
        'out': [{'A': 1, 'p': 1, 'q': 1, 'theta': 100}]}

def test_flowNetwork_fsolve_junction():
    N = FlowNetwork()
    N.addnodes([('a',{'s' : 1}), ('b',{}), ('c',{})])
    N.setJunction('j')
    N.addcomponents([('a', 'j', {'A':1,'k':1, 'pin':0,'thetaout': 0}), ('c', 'j', {'A': 1,'k':1,'thetaout': 180}), ('b','j', {'A':1, 'k':1,'thetaout': 90})])


    (dict, vec) = N.getunknowns(True)
    print dict
    print N.getJunction('j', vec)
    sol = root(N.residue, vec)
    print list(sol.x)
    print N.residue(sol.x)
    assert list(sol.x) == [-0.48820973173912569, -0.51179026826087426, 1.0, -0.00089000000000000006, -0.48820973173912569, -0.27907404091075233, -0.27863953424971338, -0.51179026826087426, -0.035504280111429337, -0.035048786772534458]

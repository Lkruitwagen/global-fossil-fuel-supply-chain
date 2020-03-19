from py2neo.data import Node, Relationship, Subgraph


def create_sample_graph():
    a = Node("Person", name="Alice")
    b = Node("Person", name="Bob")
    ab = Relationship(a, "KNOWS", b)
    return ab

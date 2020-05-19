from kedro.io.core import AbstractDataSet
from py2neo import Graph, Subgraph
import multiprocessing

from typing import Dict, Any, List


class Neo4jDataset(AbstractDataSet):
    def __init__(self, credentials, chunk_size=100_00):
        super().__init__()

        self._graph = Graph(user=credentials["user"], password=credentials["password"])
        self.chunk_size = chunk_size

    def _commit_subgraph(self, subgraph: Subgraph):
        """Commits any py2neo subgraph to a neo4j database"""
        tx = self._graph.begin()
        tx.create(subgraph)
        tx.commit()

    def _save_node_chunk(self, node_chunk: List):
        """Creates a subgraph from a list of nodes and commits it"""
        sg = Subgraph(node_chunk)
        self._commit_subgraph(sg)

    def _save_edge_chunk(self, edge_chunk):
        """Creates a subgraph from a list of edges and commits it"""
        sg = Subgraph(None, edge_chunk)
        self._commit_subgraph(sg)

    def _save(self, subgraph: Subgraph):
        """
        Saves a py2neo subgraph to a Neo4j database
        :param subgraph: Any valid subgraph
        :return: None
        """

        nodes = list(subgraph.nodes)
        edges = list(subgraph.relationships)

        print("Saving Nodes")
        if len(nodes) > self.chunk_size:
            node_chunk_size = self.chunk_size
            node_chunks = [
                nodes[i : i + node_chunk_size]
                for i in range(0, len(nodes), node_chunk_size)
            ]
            for chunk in node_chunks:
                self._save_node_chunk(chunk)
        elif len(nodes) > 0:
            self._save_node_chunk(nodes)

        print("Saving Edges")
        if len(edges) > self.chunk_size:
            edge_chunk_size = self.chunk_size
            edge_chunks = [
                edges[i : i + edge_chunk_size]
                for i in range(0, len(edges), edge_chunk_size)
            ]
            for chunk in edge_chunks:
                self._save_edge_chunk(chunk)
        elif len(edges) > 0:
            self._save_edge_chunk(edges)

    def _load(self) -> Any:
        raise NotImplementedError

    def _describe(self) -> str:
        return "Neo4J Graph Dataset Early Demo"

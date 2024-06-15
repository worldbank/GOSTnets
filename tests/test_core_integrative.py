"""Integrative tests for core.py"""

import pytest  # noqa: F401
import os
from GOSTnets import core
import networkx as nx


class TestGraphFunctions:
    fpath_edges = os.path.join("tests", "iceland_unclean_edges.csv")
    fpath_nodes = os.path.join("tests", "iceland_unclean_nodes.csv")
    G = core.edges_and_nodes_csv_to_graph(fpath_nodes, fpath_edges)

    def test_edges_and_nodes_csv_to_graph_01(self):
        """Test the edges_and_nodes_csv_to_graph function."""
        G = core.edges_and_nodes_csv_to_graph(self.fpath_nodes, self.fpath_edges)
        assert isinstance(G, nx.Graph)

    def test_convert_to_time(self):
        """Test time conversion."""
        G_time = core.convert_network_to_time(
            self.G, distance_tag="length", road_col="infra_type", factor=1000
        )
        assert isinstance(G_time, nx.Graph)
        # avg speed
        avg_speed = core.find_graph_avg_speed(G_time, "length", "time")
        assert isinstance(avg_speed, float)
        assert avg_speed > 0

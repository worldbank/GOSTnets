"""Integrative tests for core.py"""

import pytest  # noqa: F401
import os
import pandas as pd
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

    def test_combo_csv_to_graph_largest(self):
        """Testing the largest graph for the combo_csv_to_graph function."""
        G = core.combo_csv_to_graph(
            self.fpath_edges,
            u_tag="stnode",
            v_tag="endnode",
            geometry_tag="Wkt",
            largest_G=True,
        )
        G_all = core.combo_csv_to_graph(
            self.fpath_edges,
            u_tag="stnode",
            v_tag="endnode",
            geometry_tag="Wkt",
            largest_G=False,
        )
        # assertions
        assert isinstance(G, nx.Graph)
        assert isinstance(G_all, nx.Graph)
        assert len(G_all.nodes) > len(G.nodes)

    def test_edges_nodes_df_to_graph_01(self):
        """edges_and_nodes_gdf_to_graph with checks==True"""
        # read csv files as dfs
        nodes_df = pd.read_csv(self.fpath_nodes)
        edges_df = pd.read_csv(self.fpath_edges)
        # call function with checks
        G = core.edges_and_nodes_gdf_to_graph(nodes_df, edges_df, checks=True)
        # assert that the function returned a series this time
        assert isinstance(G, pd.Series)

    def test_edges_nodes_df_to_graph_02(self):
        """edges_and_nodes_gdf_to_graph, add_missing_reflected_edges==True"""
        # read csv files as dfs
        nodes_df = pd.read_csv(self.fpath_nodes)
        edges_df = pd.read_csv(self.fpath_edges)
        # call function with adding missing reflected edges
        G = core.edges_and_nodes_gdf_to_graph(
            nodes_df, edges_df, add_missing_reflected_edges=True
        )
        # assert that the function returned a nx graph this time
        assert isinstance(G, nx.Graph)

    def test_edges_nodes_df_to_graph_03(self):
        """edges_and_nodes_gdf_to_graph, largest_G==True"""
        # read csv files as dfs
        nodes_df = pd.read_csv(self.fpath_nodes)
        edges_df = pd.read_csv(self.fpath_edges)
        # call function with adding missing reflected edges
        G = core.edges_and_nodes_gdf_to_graph(nodes_df, edges_df, largest_G=True)
        # assert that the function returned a nx graph this time
        assert isinstance(G, nx.Graph)

    def test_edges_nodes_df_to_graph_04(self):
        # read csv files as dfs
        nodes_df = pd.read_csv(self.fpath_nodes)
        edges_df = pd.read_csv(self.fpath_edges)
        # switch stnode and endnode to floats
        edges_df["stnode"] = edges_df["stnode"].astype(float)
        edges_df["endnode"] = edges_df["endnode"].astype(float)
        # call function with adding missing reflected edges and the oneway tag
        G = core.edges_and_nodes_gdf_to_graph(
            nodes_df, edges_df, add_missing_reflected_edges=True, oneway_tag="one_way"
        )
        # assert that the function returned a nx graph this time
        assert isinstance(G, nx.Graph)

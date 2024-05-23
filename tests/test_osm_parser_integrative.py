"""Integrative tests for osm_parser.py"""
import pytest  # noqa: F401
from GOSTnets import osm_parser
import networkx as nx
import urllib
import os


def test_read_osm():
    """Test the read_osm() function using some test data."""
    fp = urllib.request.urlopen("file://" + os.getcwd() + "/tests/osm_map.map")
    network = osm_parser.read_osm(fp, only_roads=True)
    assert isinstance(network, nx.DiGraph)


def test_OSM():
    """Test the OSM class."""
    fp = urllib.request.urlopen("file://" + os.getcwd() + "/tests/osm_map.map")
    osm_obj = osm_parser.OSM(fp)
    assert isinstance(osm_obj, osm_parser.OSM)
    assert isinstance(osm_obj.nodes, dict)
    assert isinstance(osm_obj.ways, dict)

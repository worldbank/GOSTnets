"""Integrative tests for load_osm.py"""
import pytest  # noqa: F401
from GOSTnets import load_osm
import os
import geopandas as gpd


def test_OSM_to_network():
    """Test the OSM_to_network class."""
    fpath = os.path.join("Tutorials", "tutorial_data", "iceland-latest.osm.pbf")
    # load the class
    otn = load_osm.OSM_to_network(fpath)
    assert isinstance(otn, load_osm.OSM_to_network)
    assert isinstance(otn.osmFile, str)
    assert isinstance(otn.roads_raw, gpd.GeoDataFrame)

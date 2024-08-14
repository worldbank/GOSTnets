"""Integrative tests for load_osm.py"""

import pytest  # noqa: F401
from GOSTnets import load_osm
import geopandas as gpd
import pooch


class TestOSMtoNetwork:
    # pooch for the osm file
    osm_file = pooch.retrieve(
        url="http://download.geofabrik.de/europe/iceland-latest.osm.pbf",
        known_hash=None,
        # known_hash="md5:d4cab811ec9a3ecad5e76d9838be3dec",
    )

    def test_OSM_to_network(self):
        """Test the OSM_to_network class."""
        # load the class
        otn = load_osm.OSM_to_network(self.osm_file)
        assert isinstance(otn, load_osm.OSM_to_network)
        assert isinstance(otn.osmFile, str)
        assert isinstance(otn.roads_raw, gpd.GeoDataFrame)

    def test_OSM_to_network_ferries(self):
        """Test the OSM_to_network class with ferries."""
        # load the class
        otn = load_osm.OSM_to_network(self.osm_file, includeFerries=True)
        assert isinstance(otn, load_osm.OSM_to_network)
        assert isinstance(otn.osmFile, str)
        assert isinstance(otn.roads_raw, gpd.GeoDataFrame)

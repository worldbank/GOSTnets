"""Integrative test for the GOSTnets workflow based on tutorial notebooks."""

import pytest
import geopandas as gpd
from pathlib import Path
import GOSTnets as gn
from GOSTnets.load_osm import OSM_to_network
import shapely
import networkx as nx


def test_workflow():
    pth = (
        Path(__file__).resolve().parent.parent / "docs" / "tutorials"
    )  # change this path to your working folder
    fil = "iceland-latest.osm.pbf"  # download this file from geofabrik: http://download.geofabrik.de/europe/iceland.html.

    # be sure to place the .osm.pbf file in the 'tutorial data' folder.

    f = pth / "tutorial_data" / fil

    # convert the .osm.pbf file to a GOSTnets object
    iceland = OSM_to_network(f)
    assert isinstance(iceland, OSM_to_network)

    # show the different road types and counts
    og_counts = iceland.roads_raw.infra_type.value_counts()

    accepted_road_types = [
        "residential",
        "unclassified",
        "track",
        "service",
        "tertiary",
        "road",
        "secondary",
        "primary",
        "trunk",
        "primary_link",
        "trunk_link",
        "tertiary_link",
        "secondary_link",
    ]

    iceland.filterRoads(acceptedRoads=accepted_road_types)

    new_counts = iceland.roads_raw.infra_type.value_counts()
    assert og_counts.shape[0] > new_counts.shape[0]

    # read the shapefile for the clip area
    clip_shp = gpd.read_file(pth / "tutorial_data" / "rek2.shp")
    clip_shp = clip_shp.to_crs("epsg:4326")
    clip_shp_obj = clip_shp.geometry.iloc[0]
    assert isinstance(clip_shp, gpd.GeoDataFrame)
    assert isinstance(clip_shp_obj, shapely.geometry.multipolygon.MultiPolygon)

    # intersects is a Shapely function that returns True if the boundary or interior of the object intersect in any way with those of the other
    int_counts = iceland.roads_raw.geometry.intersects(clip_shp_obj).value_counts()
    assert int_counts[True] > 0

    iceland.roads_raw = iceland.roads_raw.loc[
        iceland.roads_raw.geometry.intersects(clip_shp_obj) == True  # noqa: E712
    ]
    # reprint the intersects value counts (should only be True now)
    new_ints = iceland.roads_raw.geometry.intersects(clip_shp_obj).value_counts()
    assert new_ints[True] == int_counts[True]

    # generate the roads GeoDataFrame, may take a few minutes
    iceland.generateRoadsGDF(verbose=False)
    assert isinstance(iceland.roadsGPD, gpd.GeoDataFrame)

    with pytest.raises(AttributeError):
        iceland.network

    iceland.initialReadIn()
    assert isinstance(iceland.network, nx.classes.multidigraph.MultiDiGraph)

    G = iceland.network

    # inspect the graph
    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))
    assert len(nodes) > 0
    assert len(edges) > 0

    Iceland_UTMZ = "epsg:32627"

    G_clean = gn.clean_network(
        G, UTM=Iceland_UTMZ, WGS="epsg:4326", junctdist=10, verbose=False
    )
    assert isinstance(G_clean, nx.classes.multidigraph.MultiDiGraph)

    G = G_clean

    G_time = gn.convert_network_to_time(
        G, distance_tag="length", road_col="infra_type", factor=1000
    )
    assert isinstance(G_time, nx.classes.multidigraph.MultiDiGraph)

    rek = gpd.read_file(pth / "tutorial_data" / "rek2.shp")
    assert isinstance(rek, gpd.GeoDataFrame)
    rek = rek.to_crs("epsg:4326")
    poly = rek.geometry.iloc[0]

    churches = gpd.read_file(pth / "tutorial_data" / "churches.shp")
    assert isinstance(churches, gpd.GeoDataFrame)
    churches = churches.loc[churches.within(poly)]

    churches = gn.pandana_snap_c(
        G_time,
        churches,
        source_crs="epsg:4326",
        target_crs="epsg:32627",
        add_dist_to_node_col=True,
    )
    assert isinstance(churches, gpd.GeoDataFrame)
    assert "NN" in churches.columns

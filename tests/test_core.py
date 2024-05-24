import pytest  # noqa: F401
from GOSTnets import core
import networkx as nx
from unittest import mock
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely import LineString
import shutil


def test_convert():
    """Test convert() function."""
    # make pandas dataframe
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            "data": [1, 2, 3],
        }
    )
    # convert geometry to wkt-loadable string
    df["geometry"] = df["geometry"].apply(lambda x: x.wkt)
    # call the function
    G = core.convert(df.iloc[0, :], "a", "b", "geometry", ["data"])
    # check the output
    assert isinstance(G, tuple)
    assert G[0] == 1
    assert G[1] == 4


def test_check():
    """Test the check() function."""
    # call the function
    G = core.check(1, set([1, 2, 3]))
    # check the output
    assert G == 1
    # call again with different input
    G = core.check(4, set([1, 2, 3]))
    # check the output
    assert G == 0


def test_selector():
    """Test the selector() function."""
    # call the function
    G = core.selector(1)
    # check the output
    assert G == 1


def test_selector_non_int():
    """Test selector() if not an int."""
    x = core.selector("y")
    assert x == "y"


def test_flatten():
    """Test the flatten() function."""
    # call the function
    G = core.flatten([[1, 2], [3, 4], [5, 6]])
    # check the output
    assert G == [1, 2, 3, 4, 5, 6]


def test_chck():
    """Test the chck() function."""
    # define the polygon and point
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    pt = Point(0.5, 0.5)
    # call the function
    G = core.chck(pt, poly)
    # check the output
    assert G == 1
    # case where point is outside the polygon
    pt = Point(2, 2)
    # call the function
    G = core.chck(pt, poly)
    # check the output
    assert G == 0


def test_combo_csv_to_graph():
    """Test the combo_csv_to_graph function."""
    pass


def test_edges_and_nodes_gdf_to_graph():
    """Test the edges_and_nodes_gdf_to_graph function."""
    pass


def test_edges_and_nodes_csv_to_graph():
    """Test the edges_and_nodes_csv_to_graph function."""
    pass


def test_node_gdf_from_graph():
    """Test the node_gdf_from_graph function."""
    pass


def test_edge_gdf_from_graph():
    """Test the edge_gdf_from_graph function."""
    pass


def test_graph_nodes_intersecting_polgyon():
    """Test the graph_nodes_intersecting_polgyon function."""
    pass


def test_graph_edges_intersecting_polgyon():
    """Test the graph_edges_intersecting_polgyon function."""
    pass


def test_sample_raster():
    """Test the sample_raster function."""
    pass


def test_generate_isochrones():
    """Test the generate_isochrones function."""
    pass


def test_make_iso_polys():
    """Test the make_iso_polys function."""
    pass


def test_make_iso_polys_original():
    """Test the make_iso_polys_original function."""
    pass


def test_find_hwy_distances_by_class():
    """Test the find_hwy_distances_by_class function."""
    pass


def test_find_graph_avg_speed():
    """Test the find_graph_avg_speed function."""
    pass


def test_example_edge(capsys):
    """Test the ExampleEdge class."""
    # make a networkx graph
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    G.add_edge(1, 2, length=1)
    # call example_edge function
    core.example_edge(G, n=1)
    # capture the output
    captured = capsys.readouterr()
    # check the output
    assert captured.out == "(1, 2, {'length': 1})\n"


def test_example_node(capsys):
    """Test the ExampleNode class."""
    # make a networkx graph
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    G.add_edge(1, 2, length=1)
    # call example_edge function
    core.example_edge(G, n=1)
    # capture the output
    captured = capsys.readouterr()
    # check the output
    assert captured.out == "(1, 2, {'length': 1})\n"


def test_convert_network_to_time():
    """Test the convert_network_to_time function."""
    pass


def test_first_val():
    """Test the first_val function."""
    # call function
    G = core.first_val([1, 2, 3])
    # check the output
    assert G == 1
    # call function with single value
    G = core.first_val("test")
    # check the output
    assert G == "test"


def test_assign_traffic_times():
    """Test the assign_traffic_times function."""
    pass


def test_calculate_OD():
    """Test the calculate_OD function."""
    pass


def test_disrupt_network():
    """Test the disrupt_network function."""
    # make a graph object
    G = nx.Graph()
    # add some nodes and edges
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=5, y=1)
    G.add_node(3, x=5, y=0)
    G.add_edge(1, 2, length=1)
    G.add_edge(2, 3, length=1)
    # call the function
    disrupted_G = core.disrupt_network(G, "x", 1, 2)
    # assert type of function
    assert isinstance(disrupted_G, nx.classes.graph.Graph)
    # need to check function and make sure graph actually gets modified


def test_randomly_disrupt_network():
    """Test the randomly_disrupt_network function."""
    pass


def test_gravity_demand():
    """Test the gravity_demand function."""
    pass


def test_unbundle_geometry():
    """Test the unbundle_geometry function."""
    pass


@mock.patch("GOSTnets.core.node_gdf_from_graph", return_value=pd.DataFrame())
def test_save_01(tmp_path):
    """Test the save function."""
    G = nx.Graph()
    # make output directory
    os.makedirs(tmp_path, exist_ok=True)
    # call the function
    core.save(G, "test", tmp_path, pickle=False, edges=False, nodes=True)
    # check the output
    assert (tmp_path / "test_nodes.csv").exists()
    # remove the output directory
    shutil.rmtree(tmp_path)
    if os.path.isdir("MagicMock"):
        shutil.rmtree("MagicMock")


@mock.patch("GOSTnets.core.edge_gdf_from_graph", return_value=pd.DataFrame())
def test_save_02(tmp_path):
    """Test the save function."""
    G = nx.Graph()
    # make output directory
    os.makedirs(tmp_path, exist_ok=True)
    # call the function
    core.save(G, "test", tmp_path, pickle=False, edges=True, nodes=False)
    # check the output
    assert (tmp_path / "test_edges.csv").exists()
    # remove the output directory
    shutil.rmtree(tmp_path)
    if os.path.isdir("MagicMock"):
        shutil.rmtree("MagicMock")


def test_save_03(tmp_path):
    """Test the save function."""
    G = nx.Graph()
    # make output directory
    os.makedirs(tmp_path, exist_ok=True)
    # call the function
    core.save(G, "test", tmp_path, pickle=True, edges=False, nodes=False)
    # check the output
    assert (tmp_path / "test.pickle").exists()
    # remove the output directory
    shutil.rmtree(tmp_path)
    if os.path.isdir("MagicMock"):
        shutil.rmtree("MagicMock")


def test_add_missing_reflected_edges():
    """Test the add_missing_reflected_edges function."""
    pass


def test_add_missing_reflected_edges_old():
    """Test the add_missing_reflected_edges_old function."""
    pass


def test_remove_duplicate_edges():
    """Test the remove_duplicate_edges function."""
    # define graph
    G = nx.MultiGraph()
    # add some nodes and edges
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=0, y=1)
    G.add_node(3, x=0, y=2)
    G.add_edge(1, 2, length=1)
    G.add_edge(2, 3, length=1)
    G.add_edge(1, 2, length=1)
    # count number of edges
    assert len(G.edges) == 3
    # call the function
    G = core.remove_duplicate_edges(G)
    # count number of edges
    assert len(G.edges) < 3


def test_convert_to_MultiDiGraph():
    """Test the convert_to_MultiDiGraph function."""
    # input graph
    G = nx.Graph()
    # add some nodes and edges
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=0, y=1)
    G.add_edge(1, 2, length=1)
    # assert
    assert isinstance(G, nx.Graph)
    # call the function
    G = core.convert_to_MultiDiGraph(G)
    # assert
    assert isinstance(G, nx.MultiDiGraph)


def test_simplify_junctions():
    """Test the simplify_junctions function."""
    pass


def test_custom_simplify():
    """Test the custom_simplify function."""
    pass


def test_cut():
    """Test the cut function."""
    # define parameters
    line = LineString([[0, 0], [1, 0], [1, 1]])
    distance = 0.5
    # call function
    two_lines = core.cut(line, distance)
    assert line.length == 2.0
    assert two_lines[0].length == 0.5
    assert two_lines[1].length == 1.5
    assert line.length == (two_lines[0].length + two_lines[1].length)


def test_salt_long_lines():
    """Test the salt_long_lines function."""
    pass


def test_pandana_snap():
    """Test the pandana_snap function."""
    pass


def test_pandana_snap_c():
    """Test the pandana_snap_c function."""
    pass


def test_pandana_snap_to_many():
    """Test the pandana_snap_to_many function."""
    pass


def test_pandana_snap_single_point():
    """Test the pandana_snap_single_point function."""
    pass


def test_pandana_snap_points():
    """Test the pandana_snap_points function."""
    pass


def test_join_networks():
    """Test the join_networks function."""
    pass


def test_clip():
    """Test the clip function."""
    pass


def test_new_edge_generator():
    """Test the new_edge_generator function."""
    pass


def test_project_gdf():
    """Test the project_gdf function."""
    # make input gdf
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]}, crs="epsg:4326")
    # call the function
    projected_gdf = core.project_gdf(gdf, to_crs="epsg:32633", to_latlong=False)
    # check the output
    assert projected_gdf.crs == "epsg:32633"


def test_gn_project_graph():
    """Test the gn_project_graph function."""
    pass


def test_reproject_graph():
    """Test the reproject_graph function."""
    pass


def test_euclidean_distance():
    """Test the euclidean_distance function."""
    # define the lon/lat points
    lon1 = 0.00
    lat1 = 0.00
    lon2 = 0.00
    lat2 = 0.00
    # calculate the haversine distance
    distance = core.euclidean_distance(lon1, lat1, lon2, lat2)
    # check the result
    assert distance == 0.0


def mocked_node_gdf_from_graph(G):
    """Mock the node_gdf_from_graph function."""
    return gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)], "osmid": [1, 2]})


@mock.patch("GOSTnets.core.node_gdf_from_graph", mocked_node_gdf_from_graph)
def test_utm_of_graph():
    """Test the utm_of_graph function."""
    # make the input graph
    G = nx.Graph()
    # call the function
    utm = core.utm_of_graph(G)
    # check the output
    assert isinstance(utm, str)
    assert utm == "+proj=utm +zone=31 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"


def test_advanced_snap():
    """Test the advanced_snap function."""
    pass


def test_add_intersection_delay():
    """Test the add_intersection_delay function."""
    pass

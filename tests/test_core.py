import pytest  # noqa: F401
from GOSTnets import core
import networkx as nx
from unittest import mock
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint
from shapely import LineString
import shutil
import io
import copy


def test_convert():
    """Test convert() function."""
    # make pandas dataframe
    df = pd.DataFrame(
        {
            "a": [1, "2", "s"],
            "b": [4, "5", "x"],
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
    # call the function - next row
    G = core.convert(df.iloc[1, :], "a", "b", "geometry", ["data"])
    # check the output
    assert isinstance(G, tuple)
    assert G[0] == 2
    assert G[1] == 5
    # call the function - last row
    G = core.convert(df.iloc[2, :], "a", "b", "geometry", ["data"])
    # check the output
    assert isinstance(G, tuple)
    assert G[0] == "s"
    assert G[1] == "x"


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


def mocked_convert(x, y, z, f, g):
    return ["one", "two", "three"]


class TestGraphCreationFunctions:
    @mock.patch("GOSTnets.core.convert", mocked_convert)
    def test_combo_csv_to_graph(self):
        """Test the combo_csv_to_graph function."""
        # create the test csv object
        f_df = pd.DataFrame(
            data={
                "u_col": ["a", "b", "c"],
                "v_col": ["x", "2", 3],
                "geo_col": [1, 2, 3],
            }
        )
        # write to buffer
        s_buf = io.StringIO()
        f_df.to_csv(s_buf)
        s_buf.seek(0)
        # call function
        G = core.combo_csv_to_graph(
            s_buf, u_tag="u_col", v_tag="v_col", geometry_tag="geo_col"
        )
        # assertions
        assert isinstance(G, nx.MultiDiGraph)
        assert "a" in G.nodes
        assert "x" in G.nodes


class TestGDFfromGraph:
    # create graph to use for tests
    G = nx.Graph()
    # add some nodes w/ (x, y) attributes
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=-1, y=0.3)
    G.add_node(3, x=2, y=0.17)
    G.add_node(4, x=4, y=0.255)
    G.add_node(5, x=5, y=0.03)
    # create some edges
    G.add_edge(1, 2)
    G.add_edge(4, 5)
    # define polygon
    poly = Polygon([(-1, -1), (0.0, 1.0), (1.0, 0.0)])
    poly_gdf = gpd.GeoDataFrame({"x": 1, "geometry": poly}, index=[1])

    def test_node_gdf_from_graph(self):
        """Test the node_gdf_from_graph function."""
        node_gdf = core.node_gdf_from_graph(self.G)
        assert isinstance(node_gdf, gpd.GeoDataFrame)
        assert "x" in node_gdf.columns
        assert "y" in node_gdf.columns
        assert "geometry" in node_gdf.columns
        assert node_gdf.shape[0] == 5

    def test_edge_gdf_from_graph(self):
        """Test the edge_gdf_from_graph function."""
        edge_gdf = core.edge_gdf_from_graph(self.G)
        assert edge_gdf.shape[0] == 2
        assert isinstance(edge_gdf, gpd.GeoDataFrame)
        assert "stnode" in edge_gdf.columns
        assert "endnode" in edge_gdf.columns
        assert "geometry" in edge_gdf.columns

    def test_graph_nodes_intersecting_polygon_errors(self):
        """Test raising errors for graph_nodes_intersection_polygon."""
        # invalid graph type
        with pytest.raises(TypeError):
            core.graph_nodes_intersecting_polygon("str", self.poly_gdf)
        # invalid polygon parameter
        with pytest.raises(TypeError):
            core.graph_nodes_intersecting_polygon(self.G, "str")

    # def test_graph_nodes_intersecting_polgyon(self):
    #     """Test the graph_nodes_intersecting_polygon function."""
    #     # call function
    #     int_list = core.graph_nodes_intersecting_polygon(self.G, self.poly_gdf)
    #     import pdb; pdb.set_trace()

    # def test_graph_edges_intersecting_polgyon(self):
    #     """Test the graph_edges_intersecting_polygon function."""
    #     int_list = core.graph_edges_intersecting_polygon(
    #         self.G, self.poly_gdf, mode="contains"
    #     )


def test_find_hwy_distances_by_class_error():
    """Test the find_hwy_distances_by_class function type error."""
    with pytest.raises(TypeError):
        core.find_hwy_distances_by_class("str")


def test_find_graph_avg_speed_error():
    """Test the find_graph_avg_speed function type error."""
    with pytest.raises(TypeError):
        core.find_graph_avg_speed("str", "km", "distance")


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
    # call example_node function
    core.example_node(G, n=1)
    # capture the output
    captured = capsys.readouterr()
    # check the output
    assert captured.out == "(1, {'x': 0, 'y': 0})\n"


def test_convert_network_to_time_error():
    """Test the convert_network_to_time function type error."""
    with pytest.raises(TypeError):
        core.convert_network_to_time("str", "dist")


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


def test_split_line():
    """Test the split_line function."""
    # define parameters
    line = LineString([[0, 0], [1, 0], [1, 1]])
    pps = MultiPoint([[0.1, 0.0], [1.0, 0.5]])
    # call function
    line_list = core.split_line(line, pps)
    assert isinstance(line_list, list)


def test_project_gdf():
    """Test the project_gdf function."""
    # make input gdf
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]}, crs="epsg:4326")
    # call the function
    projected_gdf = core.project_gdf(gdf, to_crs="epsg:32633", to_latlong=False)
    # check the output
    assert projected_gdf.crs == "epsg:32633"


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


def test_isochrones():
    """Test the generate_isochrones() function."""
    # create a networkx graph object
    G = nx.Graph()

    # Add four nodes to the graph
    G.add_nodes_from(["A", "B", "C", "D"])

    # Add edges with time values as dictionaries
    edge_data = [
        {"source": "A", "target": "B", "time": 10},
        {"source": "A", "target": "C", "time": 15},
        {"source": "A", "target": "D", "time": 20},
        {"source": "B", "target": "B", "time": 11},
        {"source": "B", "target": "C", "time": 16},
        {"source": "B", "target": "D", "time": 21},
    ]

    G.add_edges_from([(edge["source"], edge["target"], edge) for edge in edge_data])

    G_nodes = copy.deepcopy(list(G.nodes(data=True)))

    # call the function
    new_G = core.generate_isochrones(G, ["A", "B"], 0)
    new_G_nodes = copy.deepcopy(list(new_G.nodes(data=True)))

    # assert new G nodes are not the same as the old ones
    assert G_nodes != new_G_nodes

    # call again with higher threshold
    newer_G = core.generate_isochrones(G, ["A", "B"], 50)
    newer_G_nodes = copy.deepcopy(list(newer_G.nodes(data=True)))

    # assert new G nodes are not the same as the old ones
    assert G_nodes != newer_G_nodes
    assert new_G_nodes != newer_G_nodes

    # call again as stacking
    stacking_G = core.generate_isochrones(G, ["A", "B"], 50, stacking=True)
    stacking_G_nodes = copy.deepcopy(list(stacking_G.nodes(data=True)))

    # assert new G nodes are not the same as the old ones
    assert stacking_G_nodes != newer_G_nodes
    assert new_G_nodes != stacking_G_nodes

    # raise error
    with pytest.raises(ValueError):
        core.generate_isochrones(G, "invalid", 1)
    with pytest.raises(ValueError):
        core.generate_isochrones(G, ["A"], 1, stacking="invalid")


def test_hwy_distances():
    """Test the find_hwy_distances_by_class() function."""
    # create a networkx graph object
    G = nx.DiGraph()

    # Add four nodes to the graph
    G.add_nodes_from(["A", "B", "C", "D"])

    # Add edges with length and highway data values
    edge_data = [
        {"source": "A", "target": "B", "length": 1, "highway": "S"},
        {"source": "A", "target": "C", "length": 2, "highway": "S"},
        {"source": "A", "target": "D", "length": 20, "highway": "L"},
        {"source": "B", "target": "C", "length": 16, "highway": "L"},
        {"source": "B", "target": "D", "length": 1, "highway": "S"},
    ]

    G.add_edges_from([(edge["source"], edge["target"], edge) for edge in edge_data])

    # call function
    hwy_dict = core.find_hwy_distances_by_class(G)
    # assert things
    assert "S" in hwy_dict.keys()
    assert "L" in hwy_dict.keys()
    assert hwy_dict["S"] == 1 + 2 + 1
    assert hwy_dict["L"] == 20 + 16


def test_random_disrupt():
    """Test the randomly_disrupt_network() function."""
    # create a networkx graph object
    G = nx.Graph()

    # Add four nodes to the graph
    G.add_nodes_from(["A", "B", "C", "D"])

    # Add 4 edges with edge_id values
    edge_data = [
        {"source": "A", "target": "B", "edge_id": 1, "time": 10},
        {"source": "A", "target": "C", "edge_id": 2, "time": 20},
        {"source": "A", "target": "D", "edge_id": 3, "time": 30},
        {"source": "B", "target": "C", "edge_id": 4, "time": 40},
    ]

    G.add_edges_from([(edge["source"], edge["target"], edge) for edge in edge_data])

    # call function 50% reduction
    new_G, d_list = core.randomly_disrupt_network(G, 50, -1)
    assert len(d_list) == 2
    # call function with 25% reduction
    new_G, d_list = core.randomly_disrupt_network(G, 25, -1)
    assert len(d_list) == 1
    # 75% reduction
    new_G, d_list = core.randomly_disrupt_network(G, 75, -1)
    assert len(d_list) == 3
    # check flooring so 40% should be same as 25%
    new_G, d_list = core.randomly_disrupt_network(G, 40, -1)
    assert len(d_list) == 1
    assert len(new_G.edges) == 4
    one_edge = new_G.get_edge_data("A", "B")
    assert "source" in one_edge.keys()
    assert "target" in one_edge.keys()
    assert "edge_id" in one_edge.keys()
    assert "time" in one_edge.keys()

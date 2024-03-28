import pytest  # noqa: F401
from GOSTnets import osm_parser
from unittest import mock
import geopandas as gpd
from shapely.geometry import LineString


def test_haversine():
    """Test the haversine function."""
    # define the lon/lat points
    lon1 = 0.00
    lat1 = 0.00
    lon2 = 0.00
    lat2 = 0.00
    # calculate the haversine distance
    distance = osm_parser.haversine(lon1, lat1, lon2, lat2)
    # check the result
    assert distance == 0.0


def mocked_urllib_request_urlopen(url):
    """Mock the urllib.request.urlopen function."""
    return "test"


@mock.patch("urllib.request.urlopen", mocked_urllib_request_urlopen)
def test_download_osm():
    """Test the download_osm function."""
    # define inputs
    left, bottom, right, top = (0, 0, 0, 0)
    # run the function
    result = osm_parser.download_osm(left, bottom, right, top)
    # check the result
    assert result == "test"


def test_read_osm():
    """Test the read_osm function."""
    pass


def test_node():
    """Test the Node class."""
    node = osm_parser.Node(10, 3.3, 4.4)
    assert node.id == 10
    assert node.lon == 3.3
    assert node.lat == 4.4
    assert isinstance(node.tags, dict)


def test_way():
    """Test the Way class."""
    way = osm_parser.Way("id", 5)
    assert way.id == "id"
    assert way.osm == 5
    assert isinstance(way.nds, list)
    assert isinstance(way.tags, dict)


def test_way_split():
    """Test the Way class' split() method."""
    way = osm_parser.Way("id", 5)
    way.nds = [1, 2, 3]
    # split the way
    result = way.split({1: 1, 2: 3, 3: 5})
    # check the result
    assert len(result) == 2
    assert isinstance(result, list)
    assert isinstance(result[0], osm_parser.Way)
    assert isinstance(result[1], osm_parser.Way)


def test_osm():
    """Test the OSM class."""
    pass


def test_fetch_roads_OSM():
    """Test the fetch_roads_OSM function."""
    data_path = "./Tutorials/tutorial_data/iceland-latest.osm.pbf"
    # run the function
    result = osm_parser.fetch_roads_OSM(data_path)
    # check the result
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.columns[0] == "osm_id"
    assert result.columns[1] == "infra_type"
    assert result.columns[2] == "geometry"


def test_line_length():
    """Test the line_length function."""
    # make a linestring
    line = LineString([(0, 0), (1, 1)])
    # run the function
    result = osm_parser.line_length(line)
    # check the result
    assert isinstance(result, float)
    assert result > 0

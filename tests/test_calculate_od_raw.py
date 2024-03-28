import pytest  # noqa: F401
from GOSTnets import calculate_od_raw
from unittest import mock
from unittest.mock import MagicMock
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd


def mocked_pandana_snap(G, point_gdf):
    """Mock the pandana_snap function."""
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(0, 1)], "NN": [0, 1], "NN_dist": [0, 1]},
        crs="epsg:4326",
    )
    return gdf


def mocked_calc_od(G, oNodes, dNodes, fail_value):
    """Mock the calc_od function."""
    return np.array([[0, 1], [1, 0]])


@mock.patch("GOSTnets.calculate_od_raw.pandana_snap", mocked_pandana_snap)
@mock.patch("GOSTnets.calculate_od_raw.calc_od", mocked_calc_od)
def test_calculateOD_gdf(tmp_path):
    """Test the calculateOD_gdf function."""
    # generate inputs for the function
    G = nx.Graph(
        x=0,
        y=0,
    )
    origins = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(0, 1)]}, crs="epsg:4326"
    )
    destinations = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(0, 1)]}, crs="epsg:4326"
    )
    calculate_snap = False
    wgs84 = "epsg:4326"
    # Run the function
    result = calculate_od_raw.calculateOD_gdf(
        G,
        origins,
        destinations,
        calculate_snap=calculate_snap,
        wgs84=wgs84,
    )
    # Check the result
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_calculateOD_csv(tmp_path):
    """Test the calculateOD_csv function."""
    # Define inputs
    G = None
    # make test csv files
    originCSV = tmp_path / "otest_points.csv"
    originCSV.write_text("Lat,Lon\n0,0\n1,1\n2,2\n3,3\n4,4\n5,5\n6,6\n7,7\n8,8\n9,9\n")
    destinationCSV = tmp_path / "dtest_points.csv"
    destinationCSV.write_text(
        "Lat,Lon\n0,0\n1,1\n2,2\n3,3\n4,4\n5,5\n6,6\n7,7\n8,8\n9,9\n"
    )
    fail_value = 999999
    weight = "length"
    calculate_snap = True
    # mock the calculateOD_gdf function
    calculate_od_raw.calculateOD_gdf = MagicMock(return_value=1)

    # Run the function
    result = calculate_od_raw.calculateOD_csv(
        G,
        originCSV,
        destinationCSV=destinationCSV,
        fail_value=fail_value,
        weight=weight,
        calculate_snap=calculate_snap,
    )

    # Check the result
    assert result == 1
    assert calculate_od_raw.calculateOD_gdf.call_count == 1


def test_calculate_gravity():
    """Test the calculate_gravity function."""
    # make inputs
    od = np.array([[0, 1], [1, 0]])
    # Run the function
    result = calculate_od_raw.calculate_gravity(od)
    # Check the result
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 9)

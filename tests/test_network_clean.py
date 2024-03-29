import pytest  # noqa: F401
from unittest import mock
from GOSTnets import network_clean


# mock the functions that the clean_network() function calls
def mocked_simplify_junctions(G, UTM, WGS, junctdist):
    """Mock the simplify_junctions function."""
    return G


def mocked_add_missing_reflected_edges(G):
    """Mock the add_missing_reflected_edges function."""
    return G


class nodes:
    def __init__(self):
        pass

    def edges(self, data=False):
        return [(1, 2, {"Wkt": "LINESTRING(0 0, 1 1)"})]

    def number_of_edges(self):
        return 1


def mocked_custom_simplify(G):
    """Mock the custom_simplify function."""
    return nodes()


def mocked_unbundle_geometry(wkt):
    """Mock the unbundle_geometry function."""
    return wkt


def mocked_convert_to_MultiDiGraph(G):
    """Mock the convert_to_MultiDiGraph function."""
    return G


def mocked_remove_duplicate_edges(G):
    """Mock the remove_duplicate_edges function."""
    return G


@mock.patch("GOSTnets.network_clean.simplify_junctions", mocked_simplify_junctions)
@mock.patch(
    "GOSTnets.network_clean.add_missing_reflected_edges",
    mocked_add_missing_reflected_edges,
)
@mock.patch("GOSTnets.network_clean.custom_simplify", mocked_custom_simplify)
@mock.patch("GOSTnets.network_clean.unbundle_geometry", mocked_unbundle_geometry)
@mock.patch(
    "GOSTnets.network_clean.convert_to_MultiDiGraph", mocked_convert_to_MultiDiGraph
)
@mock.patch(
    "GOSTnets.network_clean.remove_duplicate_edges", mocked_remove_duplicate_edges
)
def test_clean_network():
    """Test the clean_network function."""
    sg = network_clean.clean_network(
        G=nodes(),
        UTM=None,
        WGS=None,
        junctdist=None,
        verbose=None,
        wpath=None,
        output_file_name=None,
    )
    # make assertions
    assert isinstance(sg, nodes)
    assert sg.number_of_edges() == 1

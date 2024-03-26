import pytest  # noqa: F401
from GOSTnets import calculate_od_raw
from unittest.mock import MagicMock


def test_calculateOD_gdf(tmp_path):
    """Test the calculateOD_gdf function."""
    pass


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
    pass

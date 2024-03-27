import pytest  # noqa: F401
from GOSTnets.conversion_utils import rasterize_od_results
import geopandas as gpd
from shapely.geometry import Point


def test_rasterize_od_results(tmp_path):
    """Test the rasterize_od_results function."""
    # make input geopandas dataframe
    gdf = gpd.GeoDataFrame(
        {
            "field1": [1, 2, 3],
            "geometry": [
                Point(0, 0),
                Point(0, 1),
                Point(1, 1),
            ],
        }
    )
    # call the function
    rasterize_od_results(gdf, tmp_path / "output.tif", "field1")
    # check that the output file was created
    assert (tmp_path / "output.tif").exists()

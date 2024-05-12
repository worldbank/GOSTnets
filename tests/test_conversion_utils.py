import pytest  # noqa: F401
import numpy as np
from unittest import mock
from unittest.mock import MagicMock
from GOSTnets.conversion_utils import rasterize_od_results
import geopandas as gpd
from shapely.geometry import Point


def test_rasterize_od_results(tmp_path):
    """Test the rasterize_od_results function, no template."""
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


def mocked_rasterio_open(
    template,
    t="w",
    driver="gtiff",
    height="y",
    width="x",
    count=1,
    dtype="dtype",
    crs="crs",
    transform="t",
):
    """Mocked function for rasterio.open()"""

    class tmpOutput:
        def __init__(self):
            self.crs = "EPSG:4326"
            self.meta = MagicMock()
            self.res = (0, 1)
            self.transform = "transform"
            self.shape = (10, 20)

        def read(self):
            raster = np.zeros((10, 10))
            raster[:5, :5] = 4
            raster[5:, 5:] = 3
            raster = np.reshape(raster, [1, 10, 10])
            return raster

        def write_band(self, x, b):
            return print("write")

        def close(self):
            return print("close")

    return_val = tmpOutput()
    return return_val


def mocked_rasterize(shapes="a", fill=0, out_shape=(1, 2), transform="t"):
    """Mock the rasterio.feature.rasterize function."""
    return np.zeros((2, 2))


@mock.patch("rasterio.open", mocked_rasterio_open)
@mock.patch("rasterio.features.rasterize", mocked_rasterize)
def test_rasterize_od_results_with_template(capfd):
    """Test the rasterize_od_results function with a template."""
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
    rasterize_od_results(gdf, "output.tif", "field1", template="template")
    # check text output by the mocked functions
    captured = capfd.readouterr()
    assert captured.out[:5] == "write"
    assert captured.out[7:11] == "lose"

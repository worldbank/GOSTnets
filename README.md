# GOSTnets: Build, process, and analyze networks

[GOSTnets](https://github.com/worldbank/GOSTnets) is built on top of geopandas, networkx, osmnx, and rtree.

## Installation

### From PyPI
The first line includes conda install of several libraries; this is to support Windows users for whom gdal and geopandas do not install cleanly through pip.
```
conda create --name gostnets geopandas gdal osmnx -c conda-forge
conda activate gostnets
pip install GOSTnets
```
Additionally, there are two extensions you may want to include.
```shell
pip install GOSTnets[tutorials]
```
```shell
pip install GOSTnets[dev]
```

### From Source
1. Clone or download this repository to your local machine. Then, navigate to the root directory of the repository:

    ```shell
    git clone https://github.com/worldbank/GOSTnets.git
    cd GOSTnets
    ```
2. Create a virtual environment (optional but recommended):

    ```shell
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the package in editable mode with dependencies:

    ```shell
    pip install -e .
    ```

    The `-e` flag stands for "editable," meaning changes to the source code will immediately affect the installed package.


## Usage

Every function contains a docstring which can be brought up in use to check the inputs for various functions. For example:

```python
import GOSTnets as gn
gn.edge_gdf_from_graph?
```
returns:
```python
Signature: gn.edge_gdf_from_graph(G, crs={'init': 'epsg:4326'}, attr_list=None, geometry_tag='geometry', xCol='x', yCol='y')
#### Function for generating a GeoDataFrame from a networkx Graph object ###
 REQUIRED: a graph object G
 OPTIONAL: crs - projection of format {'init' :'epsg:4326'}. Defaults to
           WGS84. Note: here we are defining the crs of the input geometry -
           we do NOT reproject to this crs. To reproject, consider using
           geopandas' to_crs method on the returned gdf.
           attr_list: list of the keys which you want to be moved over to
           the GeoDataFrame.
           geometry_tag - the key in the data dictionary for each edge which
           contains the geometry info.
           xCol - if no geometry is present in the edge data dictionary, the
           function will try to construct a straight line between the start
           and end nodes, if geometry information is present in their data
           dictionaries.  Pass the Longitude info as 'xCol'.
           yCol - likewise, determining the Latitude tag for the node's data
           dictionary allows us to make a straight line geometry where an
           actual geometry is missing.
 RETURNS: a GeoDataFrame object of the edges in the graph
#-------------------------------------------------------------------------#
```
## License
This licensed under the [**MIT License**](https://opensource.org/license/mit). This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

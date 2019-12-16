# GOSTNets

**Python for network analysis**

Builds, process, and analyze networks. GOSTNets is built on top of geopandas, networkx, osmnx, and peartree.

### Installation
Eventually we will have the tool available on pip and conda, but for now, please use the setup.py in this repository

```
conda create --name test
conda activate test
conda install -c conda-forge rtree
git clone https://github.com/worldbank/GOSTnets.git
python setup.py build
python setup.py install
```

## Documentation

Documentation available at [readthedocs](https://gostnets.readthedocs.io/)

Plenty of examples and tutorials using Jupyter Notebooks live inside of the Implementations folder within the [GOST_PublicGoods Github repo](https://github.com/worldbank/GOST_PublicGoods)

### how to autobuild docs:
in the docs/source dir, run: 
```
sphinx-apidoc -f -o . ../../GOSTnets
```
in the docs dir, run:
```
make html
```

## Usage

Every function contains a docstring which can be brought up in use to check the inputs for various functions. For example: 

```python
import GOSTnets as gn
gn.edge_gdf_from_graph?
```

returns: 

```
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

These docstrings have been written for every function, and should help new and old users alike with the options and syntax.

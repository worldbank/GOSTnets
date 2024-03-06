# GOSTnets: Build, process, and analyze networks

[GOSTnets](https://github.com/worldbank/GOSTnets) is built on top of geopandas, networkx, osmnx, and rtree.

## Installation

### From PyPI

#### *pypi support in development*

<!-- ```
conda create --name test python=3.8
conda activate test
conda install -c conda-forge rtree=0.9.3 geopandas rasterio geojson
pip install GOSTnets
``` -->

### From `conda-forge`

#### *conda support in development*

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

<!-- ### From Docker

#### pull image from DockerHub

Clone this repo in your local environment (for example in: C:\repos\GOSTnets). Then run the docker container:

```
docker run -i -t -p 8888:8888 -v ${PWD}:/home -v C:\repos\GOSTnets:/GOSTnets --name anaconda3_GOSTnets_c1 d3netxer/anaconda3_GOSTnets_v1
```

note in the docker command how you are mapping the 8888 port in the docker container to your local machine. You are also creating a volume to the [GOSTnets](https://github.com/worldbank/GOSTnets) repository code. You are also creating another volume in your present working directory, this is where your project code should be. Then within your container first activate the 'geo_env' anaconda environment ```conda activate geo_env```. Then use the following command to launch jupyter notebook from the container:

```
jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root --notebook-dir=/home
```

It will read from your present working directory and the notebook will be exposed through the mapped 8888 port, for you to open with your browser. A tip is that a great development set-up is to your VS Code and install the docker extensions. Once the container is running you can attach to it using VS Code, then you are able to easily use VS Code to write your code and run commands in your docker container.

note: graph-tool is also installed in this docker container.

#### Build GOSTnets container from scratch

First you will run the continuumio/anaconda3 docker container:

```
docker run -i -t -p 8888:8888 -v ${PWD}:/home --name anaconda3 continuumio/anaconda3
```

Then inside the container you will install the dependencies (followed these instructions: https://geopandas.org/en/stable/getting_started/install.html)

```
conda create -n geo_env
conda activate geo_env
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install python=3 geopandas rasterio geojson git gdal geopy boltons pulp jupyterlab osmnx
```

optional: you can also install graph-tool using Conda and these instructions: https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions

Then you will commit your image. -->

## Alternative Installations With Optional Dependencies

The following are optional dependencies that can be installed with [GOSTnets](https://github.com/worldbank/GOSTnets) - which not required to use [GOSTnets](https://github.com/worldbank/GOSTnets), but they may be useful for some users, as they enable additional functionality. All of these alternative installs use the same `pip install` command as above. The examples show how to install these from PyPI, but the same commands can be used to install from source, replacing the package name `GOSTnets` with a `.` similar to the example above.

### OSM Support (Needed to run functions from `load_osm.py`)

```shell
pip install GOSTnets[osm]
```

### Optimization Support (Needed to run functions from `optimization.py`)

```shell
pip install GOSTnets[opt]
```

### Development Support

```shell
pip install GOSTnets[dev]
```

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

These docstrings have been written for every function, and should help new and old users alike with the options and syntax.

## License

This licensed under the [**MIT License**](https://opensource.org/license/mit). This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

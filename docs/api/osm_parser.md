# osm_parser.py

source: https://gist.github.com/Tofull/49fbb9f3661e376d2fe08c2e9d64320e

### `haversine(lon1, lat1, lon2, lat2, unit_m=True)`

**Description**
Calculate the great circle distance between two points

**Parameters**
- `lon1` (float): longitude of the first point
- `lat1` (float): latitude of the first point
- `lon2` (float): longitude of the second point
- `lat2` (float): latitude of the second point
- `unit_m` (bool, optional): if True, return the distance in meters (default) if False, return the distance in kilometers

**Returns**
- float: distance between the two points

---

### `download_osm(left, bottom, right, top, proxy=False, proxyHost='10.0.4.2', proxyPort='3128', cache=False, cacheTempDir='/tmp/tmpOSM/', verbose=True)`

**Description**
Downloads OpenStreetMap data for a given bounding box.

**Parameters**
- `left` (float): The left longitude of the bounding box.
- `bottom` (float): The bottom latitude of the bounding box.
- `right` (float): The right longitude of the bounding box.
- `top` (float): The top latitude of the bounding box.
- `proxy` (bool, optional): Whether to use a proxy for the request. Defaults to False.
- `proxyHost` (str, optional): The proxy host address. Defaults to "10.0.4.2".
- `proxyPort` (str, optional): The proxy port number. Defaults to "3128".
- `cache` (bool, optional): Whether to cache the downloaded tile. Defaults to False.
- `cacheTempDir` (str, optional): The directory to store the cached tile. Defaults to "/tmp/tmpOSM/".
- `verbose` (bool, optional): Whether to print progress messages. Defaults to True.

**Returns**
- file-like object: The downloaded OpenStreetMap tile.

---

### `read_osm(filename_or_stream, only_roads=True)`

**Description**
Read graph in OSM format from file specified by name or by stream object.

**Parameters**
- `filename_or_stream` (string or file): The filename or stream to read. File can be either a filename or stream/file object.
- `only_roads` (bool, optional): Whether to only read roads. Defaults to True.

**Returns**
- networkx multidigraph: The graph from the OSM file.
- Examples
- >>> G=nx.read_osm(nx.download_osm(-122.33,47.60,-122.31,47.61))
- >>> import matplotlib.pyplot as plt
- >>> plt.plot([G.node[n]['lat']for n in G], [G.node[n]['lon'] for n in G], 'o', color='k')
- >>> plt.show()

---

### `fetch_roads_OSM(osm_path, acceptedRoads=['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link'])`

**Description**
Returns a GeoDataFrame of OSM roads from an OSM file

**Parameters**
- `osm_path` (str): path to OSM file
- `acceptedRoads` (list, optional): list of OSM road types

**Returns**
- gpd.GeoDataFrame: A GeoDataFrame of OSM roads

---

### `line_length(line, ellipsoid='WGS-84')`

**Description**
Returns length of a line in kilometers, given in geographic

**Parameters**
- `line` (shapely.geometry.LineString): A shapely LineString object with WGS-84 coordinates
- `ellipsoid` (str, optional): string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance)

**Returns**
- float: Length of line in kilometers

---

## Class Node

Represents a node in the OpenStreetMap data.

## Class Way

Represents a way in the OpenStreetMap data.

### `Way.split(dividers)`

**Description**
Splits the way into multiple smaller ways based on the given dividers.

**Parameters**
- `dividers` (dict): A dictionary containing the number of occurrences of each node reference.

**Returns**
- list: A list of new Way objects, each representing a slice of the original way.

---

## Class OSM

Represents an OpenStreetMap (OSM) data structure.

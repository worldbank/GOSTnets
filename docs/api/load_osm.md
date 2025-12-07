# load_osm.py

## Class OSM_to_network

Object to load OSM PBF to networkX objects.

### `OSM_to_network.generateRoadsGDF(in_df=None, outFile='', verbose=False)`

**Description**
post-process roads GeoDataFrame adding additional attributes

**Parameters**
- `in_df` (GeoDataFrame, optional): Optional input GeoDataFrame
- `outFile` (string, optional): optional parameter to output a csv with the processed roads
- `verbose` (optional): 

**Returns**
- float: Length of line in kilometers

---

### `OSM_to_network.filterRoads(acceptedRoads=['primary', 'primary_link', 'secondary', 'secondary_link', 'motorway', 'motorway_link', 'trunk', 'trunk_link'])`

**Description**
Extract certain times of roads from the OSM before the netowrkX conversion

**Parameters**
- `acceptedRoads` (list of strings, optional): list of accepted road types

**Returns**
- None: the raw roads are filtered based on the list of accepted roads

---

### `OSM_to_network.fetch_roads(data_path)`

**Description**
Extracts roads from an OSM PBF

**Parameters**
- `data_path` (string): The directory of the shapefiles consisting of edges and nodes

**Returns**
- GeoDataFrame: a road GeoDataFrame

---

### `OSM_to_network.fetch_roads_and_ferries(data_path)`

**Description**
Extracts roads and ferries from an OSM PBF

**Parameters**
- `data_path` (string): The directory of the shapefiles consisting of edges and nodes

**Returns**
- GeoDataFrame: a road GeoDataFrame

---

### `OSM_to_network.line_length(line, ellipsoid='WGS-84')`

**Description**
Returns length of a line in kilometers, given in geographic coordinates. Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

**Parameters**
- `line` (LineString): a shapely LineString object with WGS-84 coordinates
- `ellipsoid` (str, optional): string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance)

**Returns**
- float: Length of line in kilometers

---

### `OSM_to_network.get_all_intersections(shape_input, idx_osm=None, unique_id='osm_id', verboseness=False)`

**Description**
Processes GeoDataFrame and splits edges as intersections

**Parameters**
- `shape_input` (GeoDataFrame): Input GeoDataFrame
- `idx_osm` (spatial index, optional): The geometry index name
- `unique_id` (string, optional): The unique id field name
- `verboseness` (optional): 

**Returns**
- GeoDataFrame: returns processed GeoDataFrame

---

### `OSM_to_network.initialReadIn(fpath=None, wktField='Wkt')`

**Description**
Convert the OSM object to a networkX object

**Parameters**
- `fpath` (string, optional): path to CSV file with roads to read in
- `wktField` (string, optional): wktField name

**Returns**
- nx.MultiDiGraph: a networkX MultiDiGraph object

---

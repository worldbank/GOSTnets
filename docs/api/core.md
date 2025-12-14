# core.py

## `convert(x, u_tag, v_tag, geometry_tag, attr_list)`

**Description:**

Convert a row of a pandas dataframe to a tuple of (u, v, data) for use in a networkx graph

**Parameters:**

- `x` (pandas.Series): a row of a pandas dataframe
- `u_tag` (str): the column name for the u node
- `v_tag` (str): the column name for the v node
- `geometry_tag` (str): the column name for the geometry
- `attr_list` (list): a list of the columns to be included in the data dictionary

**Returns:**

- tuple: a tuple of (u, v, data) for use in a networkx graph

---

## `combo_csv_to_graph(fpath, u_tag='u', v_tag='v', geometry_tag='Wkt', largest_G=False)`

**Description:**

Function for generating a G object from a saved combo .csv

**Parameters:**

- `fpath` (str): path to a .csv containing edges (WARNING: COMBO CSV only)
- `u_tag` (str, optional): specify column containing u node ID if not labelled 'u'
- `v_tag` (str, optional): specify column containing v node ID if not labelled 'v'
- `geometry_tag` (str, optional): specify column containing geometry if not "Wkt"
- `largest_G` (bool, optional): Boolean that if True, returns only the largest sub-graph, default is False

**Returns:**

- nx.MultiDiGraph

---

## `check(x, chck_set)`

**Description:**

Check if a value is in a set

**Parameters:**

- `x` (any): the value to be checked
- `chck_set` (set): the set to be checked against

**Returns:**

- int: 1 if the value is in the set, 0 if not

---

## `selector(x)`

**Description:**

Selects and returns an integer if the input is an integer, otherwise returns the input as is.

**Parameters:**

- `x` (int or any): The input value to be selected.

**Returns:**

- int or any: The selected value.

---

## `edges_and_nodes_gdf_to_graph(nodes_df, edges_df, node_tag='node_ID', u_tag='stnode', v_tag='endnode', geometry_tag='Wkt', largest_G=False, discard_node_col=[], checks=False, add_missing_reflected_edges=False, oneway_tag=None)`

**Description:**

Function for generating a G object from dataframse of nodes and one of edges

**Parameters:**

- `nodes_df` (str): Pandas DataFrame with node information
- `edges_df` (str): Pandas DataFrame with edges information
- `node_tag` (optional):
- `u_tag` (str, optional): optional. specify column containing u node ID if not labelled 'stnode'
- `v_tag` (str, optional): specify column containing v node ID if not labelled 'endnode'
- `geometry_tag` (str, optional): specify column containing geometry if not labelled 'Wkt'
- `largest_G` (bool, optional): If largest_G is true, then only the largest graph will be returned
- `discard_node_col` (list, optional): default is empty, all columns in the nodes_df will be copied to the nodes in the graph. If a list is filled, all the columns specified will be dropped.
- `checks` (bool, optional): if True, will perform a validation checks and return the nodes_df with a 'node_in_edge_df' column
- `add_missing_reflected_edges` (bool, optional): if contains a tag, then the oneway column is used to see whether reverse edges need to be added. This is much faster than using the add_missing_reflected_edges after a graph is already created.
- `oneway_tag` (str, optional): if oneway_tag exists, then missing reflected edges won't be added where an edge's oneway_tag equals True

**Returns:**

- nx.MultiDiGraph

---

## `edges_and_nodes_csv_to_graph(fpath_nodes, fpath_edges, u_tag='stnode', v_tag='endnode', geometry_tag='Wkt', largest_G=False)`

**Description:**

Function for generating a G object from a saved .csv of edges

**Parameters:**

- `fpath_nodes` (str): path to a .csv containing nodes
- `fpath_edges` (str): path to a .csv containing edges
- `u_tag` (str, optional): optional. specify column containing u node ID if not labelled 'stnode'
- `v_tag` (str, optional): specify column containing v node ID if not labelled 'endnode'
- `geometry_tag` (str, optional): specify column containing geometry if not labelled 'Wkt'
- `largest_G` (optional):

**Returns:**

- nx.MultiDiGraph

---

## `flatten(line)`

**Description:**

Flattens a nested list into a single list.

**Parameters:**

- `line` (list): The nested list to be flattened.

**Returns:**

- list: The flattened list.

---

## `node_gdf_from_graph(G, crs='epsg:4326', attr_list=None, geometry_tag='geometry', xCol='x', yCol='y')`

**Description:**

Function for generating GeoDataFrame from Graph

**Parameters:**

- `G` (nx.Graph): a graph object G
- `crs` (str, optional): projection of format 'epsg:4326'. Defaults to WGS84. note: here we are defining the crs of the input geometry - we do NOT reproject to this crs. To reproject, consider using geopandas' to_crs method on the returned gdf.
- `attr_list` (list, optional): list of the keys which you want to be moved over to the GeoDataFrame, if not all. Defaults to None, which will move all.
- `geometry_tag` (str, optional): specify geometry attribute of graph, default 'geometry'
- `xCol` (str, optional): if no shapely geometry but Longitude present, assign here
- `yCol` (str, optional): if no shapely geometry but Latitude present, assign here

**Returns:**

- gpd.GeoDataFrame: a geodataframe of the node objects in the graph

---

## `edge_gdf_from_graph(G, crs='EPSG:4326', attr_list=None, geometry_tag='geometry', xCol='x', yCol='y', oneway_tag='oneway', single_edge=False)`

**Description:**

Function for generating a GeoDataFrame from a networkx Graph object

**Parameters:**

- `G` (nx.Graph): (required) a graph object G
- `crs` (str, optional): (optional) projection of format 'epsg:4326'. Defaults to WGS84. Note: here we are defining the crs of the input geometry -we do NOT reproject to this crs. To reproject, consider using geopandas' to_crs method on the returned gdf.
- `attr_list` (list, optional): (optional) list of the keys which you want to be moved over to the GeoDataFrame.
- `geometry_tag` (str, optional): (optional) the key in the data dictionary for each edge which contains the geometry info.
- `xCol` (str, optional): (optional) if no geometry is present in the edge data dictionary, the function will try to construct a straight line between the start and end nodes, if geometry information is present in their data dictionaries.  Pass the Longitude info as 'xCol'.
- `yCol` (str, optional): (optional) likewise, determining the Latitude tag for the node's data dictionary allows us to make a straight line geometry where an actual geometry is missing.
- `oneway_tag` (optional):
- `single_edge` (bool, optional): If True then one edge/row in the returned GeoDataFrame will represent a bi-directional edge. An extra 'oneway' column will be added

**Returns:**

- gpd.GeoDataFrame: a GeoDataFrame object of the edges in the graph

---

## `chck(x, poly)`

**Description:**

Check if a point is within a polygon

**Parameters:**

- `x` (shapely.geometry.Point): a point object
- `poly` (shapely.geometry.Polygon): a polygon object

**Returns:**

- int: 1 if the point is within the polygon, 0 if not

---

## `graph_nodes_intersecting_polygon(G, polygons, crs=None)`

**Description:**

Function for identifying nodes of a graph that intersect polygon(s). Ensure any GeoDataFrames are in the same projection before using function, or pass a crs.

**Parameters:**

- `G` (nx.Graph or gpd.GeoDataFrame): a Graph object OR node geodataframe
- `polygons` (gpd.GeoDataFrame): a GeoDataFrame containing one or more polygons
- `crs` (str, optional): a crs object of form 'epsg:XXXX'. If passed, matches both inputs to this crs.

**Returns:**

- list: a list of the nodes intersecting the polygons

---

## `graph_edges_intersecting_polygon(G, polygons, mode, crs=None, fast=True)`

**Description:**

Function for identifying edges of a graph that intersect polygon(s). Ensure any GeoDataFrames are in the same projection before using function, or pass a crs.

**Parameters:**

- `G` (nx.Graph): a Graph object
- `polygons` (gpd.GeoDataFrame): a GeoDataFrame containing one or more polygons
- `mode` (str): a string, either 'contains' or 'intersecting'
- `crs` (dict, optional): If passed, will reproject both polygons and graph edge gdf to this projection.
- `fast` (bool, optional): (default: True): we can cheaply test whether an edge intersects a polygon gdf by checking whether either the start or end nodes are within a polygon. If both are, then we return 'contained'; if at least one is, we can return 'intersects'. If we set fast to False, then we iterate through each geometry one at a time, and check to see whether the geometry object literally intersects the polygon geodataframe, one at a time. May be computationally intensive!

**Returns:**

- geopandas.GeoDataFrame: a GeoDataFrame containing the edges intersecting the polygons

---

## `sample_raster(G, tif_path, property_name='RasterValue')`

**Description:**

Function for attaching raster values to corresponding graph nodes.

**Parameters:**

- `G` (nx.Graph): a graph containing one or more nodes
- `tif_path` (str): a raster or path to a tif
- `property_name` (str, optional): a property name for the value of the raster attached to the node

**Returns:**

- nx.Graph: The original graph with a new data property for the nodes included in the raster

---

## `generate_isochrones(G, origins, thresh, weight=None, stacking=False)`

**Description:**

Function for generating isochrones from one or more graph nodes. Ensure any GeoDataFrames / graphs are in the same projection before using function, or pass a crs

**Parameters:**

- `G` (nx.Graph): a graph containing one or more nodes
- `origins` (list): a list of node IDs that the isochrones are to be generated from
- `thresh` (str): The time threshold for the calculation of the isochrone
- `weight` (str, optional): Name of edge weighting for calculating 'distances'. For isochrones, should be time expressed in seconds. Defaults to time expressed in seconds.
- `stacking` (bool, optional): If True, returns number of origins that can be reached from that node. If false, max = 1

**Returns:**

- nx.Graph: The original graph with a new data property for the nodes and edges included in the isochrone

---

## `make_iso_polys(G, origins, trip_times, edge_buff=10, node_buff=25, infill=False, weight='time', measure_crs='epsg:4326', edge_filters=None)`

**Description:**

Function for adding a time value to edge dictionaries

**Parameters:**

- `G` (nx.Graph): a graph object
- `origins` (list): a list object of node IDs from which to generate an isochrone poly object
- `trip_times` (list): a list object containing the isochrone values
- `edge_buff` (int, optional): the thickness with witch to buffer included edges
- `node_buff` (int, optional): the thickness with witch to buffer included nodes
- `infill` (bool, optional): If True, will remove any holes in isochrones
- `weight` (str, optional): The edge weight to use when appraising travel times.
- `measure_crs` (str, optional): measurement crs, object of form 'epsg:XXXX'
- `edge_filters` (dict, optional): you can optionally add a dictionary with key values, where the key is the attribute and the value you want to ignore from creating isochrones. An example might be an underground subway line.

**Returns:**

- gpd.GeoDataFrame: a GeoDataFrame object of the isochrone polygons

---

## `make_iso_polys_original(G, origins, trip_times, edge_buff=10, node_buff=25, infill=False, weight='time', measure_crs='epsg:4326')`

**Description:**

Function for adding a time value to edge dictionaries

**Parameters:**

- `G` (nx.Graph): a graph object
- `origins` (list): a list object of node IDs from which to generate an isochrone poly object
- `trip_times` (list): a list object containing the isochrone values
- `edge_buff` (int, optional): the thickness with witch to buffer included edges
- `node_buff` (int, optional): the thickness with witch to buffer included nodes
- `infill` (bool, optional): If True, will remove any holes in isochrones
- `weight` (str, optional): The edge weight to use when appraising travel times.
- `measure_crs` (str, optional): measurement crs, object of form 'epsg:XXXX'

**Returns:**

- gpd.GeoDataFrame: GeoDataFrame object of the isochrone polygons

---

## `find_hwy_distances_by_class(G, distance_tag='length')`

**Description:**

Function for finding out the different highway classes in the graph and their respective lengths

**Parameters:**

- `G` (nx.Graph): a graph object
- `distance_tag` (str, optional): specifies which edge attribute represents length

**Returns:**

- dict: a dictionary that has each class and the total distance per class

---

## `find_graph_avg_speed(G, distance_tag, time_tag)`

**Description:**

Function for finding the average speed per km for the graph. It will sum up the total meters in the graph and the total time (in sec).     Then it will convert m/sec to km/hr. This function needs the 'convert_network_to_time' function to have run previously.

**Parameters:**

- `G` (nx.Graph): a graph containing one or more nodes
- `distance_tag` (str): the key in the dictionary for the field currently containing a distance in meters
- `time_tag` (str): time to traverse the edge in seconds

**Returns:**

- float: The average speed for the whole graph in km per hr

---

## `example_edge(G, n=1)`

**Description:**

Prints out an example edge

**Parameters:**

- `G` (nx.Graph): a graph object
- `n` (int, optional): n - number of edges to print

**Returns:**

- None: Prints out the edge data

---

## `example_node(G, n=1)`

**Description:**

Prints out an example node

**Parameters:**

- `G` (nx.Graph): a graph object
- `n` (int, optional): number of nodes to print

**Returns:**

- None: Prints out the node data

---

## `convert_network_to_time(G, distance_tag, graph_type='drive', road_col='highway', output_time_col='time', speed_dict=None, walk_speed=4.5, factor=1, default=20)`

**Description:**

Function for adding a time value to graph edges. Ensure any graphs are in the same projection before using function, or pass a crs.

**Parameters:**

- `G`:
- `distance_tag`:
- `graph_type` (optional):
- `road_col` (optional):
- `output_time_col` (optional):
- `speed_dict` (optional):
- `walk_speed` (optional):
- `factor` (optional):
- `default` (optional):

**Returns:**

---

## `first_val(x)`

**Description:**

Get the first value of a list, or the value itself if it is not a list

**Parameters:**

- `x` (list or str): a list or string

**Returns:**

- str: the first value of the list, or the string itself if it is not a list

---

## `assign_traffic_times(G, mb_token, accepted_road_types=['trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'motorway', 'motorway_link'], verbose=False, road_col='infra_type', id_col='id')`

**Description:**

Function for querying travel times from the Mapbox "driving traffic" API. Queries are only made for the specified road types.

**Parameters:**

- `G` (nx.Graph): a graph object of the road network
- `mb_token` (str): Mapbox token (retrieve from Mapbox account, starts with "pk:")
- `accepted_road_types` (optional):
- `verbose` (bool, optional): Set to true to monitor progress of queries and notify if any queries failed, defaults to False
- `road_col` (str, optional): key for the road type in the edge data dictionary, defaults to 'infra_type'
- `id_col` (str, optional): key for the id in the edge data dictionary, defaults to 'id'

**Returns:**

- nx.Graph: The original graph with two new data properties for the edges: 'mapbox_api' (a boolean set to True if the edge successfully received a traffic time value) and 'time_traffic' (travel time in seconds)

---

## `calculate_OD(G, origins, destinations, fail_value, weight='time', weighted_origins=False, one_way_roads_exist=False, verbose=False)`

**Description:**

Function for generating an origin: destination matrix

**Parameters:**

- `G` (nx.Graph): a graph containing one or more nodes
- `origins` (list): a list of the node IDs to treat as origins points
- `destinations` (list): a list of the node IDs to treat as destinations
- `fail_value` (int): the value to return if the trip cannot be completed (implies some sort of disruption / disconnected nodes)
- `weight` (str, optional): use edge weight of 'time' unless otherwise specified
- `weighted_origins` (bool, optional): equals 'true' if the origins have weights. If so, the input to 'origins' must be dictionary instead of a list, where the keys are the origin IDs and the values are the weighted demands.
- `one_way_roads_exist` (bool, optional): If the value is 'True', then even if there are more origins than destinations, it will not do a flip during processing.
- `verbose` (bool, optional): Set to true to monitor progress of queries and notify if any queries failed, defaults to False

**Returns:**

- numpy matrix: a numpy matrix of format `OD[o][d]` = shortest time possible

---

## `disrupt_network(G, property, thresh, fail_value)`

**Description:**

Function for disrupting a graph given a threshold value against a node's value. Any edges which bind to broken nodes have their 'time' property set to fail_value

**Parameters:**

- `G` (nx.Graph): REQUIRED a graph containing one or more nodes and one or more edges
- `property` (str): the element in the data dictionary for the edges to test
- `thresh` (int): values of data[property] above this value are disrupted
- `fail_value` (int): The data['time'] property is set to this value to simulate the removal of the edge

**Returns:**

- nx.Graph: a modified graph with the edited 'time' attribute

---

## `randomly_disrupt_network(G, edge_frac, fail_value)`

**Description:**

Function for randomly disurpting a network. NOTE: requires the graph to have an 'edge_id' value in the edge data dictionary. This DOES NOT have to be unique.

**Parameters:**

- `G` (nx.Graph): a graph containing one or more nodes and one or more edges
- `edge_frac` (int): the percentage of edges to destroy. Integer rather than decimal, e.g. 5 = 5% of edges
- `fail_value` (int): the data['time'] property is set to this value to simulate the removal of the edge

**Returns:**

- tuple (nx.Graph, destroy_list): nx.Graph a modified graph with the edited 'time' attribute the list of edge IDs randomly chosen for destruction destroy_list list of the integers corresponding to the edge indices that were 'destroyed'

---

## `gravity_demand(G, origins, destinations, weight, maxtrips=100, dist_decay=1, fail_value=99999999999)`

**Description:**

Function for generating a gravity-model based demand matrix. Note: 1 trip will always be returned between an origin and a destination, even if weighting would otherwise be 0.

**Parameters:**

- `G`:
- `origins` (list): a list of node IDs. Must be in G.
- `destinations` (list): a list of node IDs Must be in G.
- `weight` (str): the gravity weighting of the nodes in the model, e.g. population
- `maxtrips` (int, optional): normalize the number of trips in the resultant function to this number of trip_times
- `dist_decay` (int, optional): parameter controlling the aggresion of discounting based on distance
- `fail_value` (int, optional): the data['time'] property is set to this value to simulate the removal of the edge

**Returns:**

- numpy array: a numpy array describing the demand between o and d in terms of number of trips

---

## `unbundle_geometry(c)`

**Description:**

Function for unbundling complex geometric objects. Note: shapely MultiLineString objects quickly get complicated. They may not show up when you plot them in QGIS. This function aims to make a .csv 'plottable'

**Parameters:**

- `c` (object): any object. This helper function is usually applied in lambda format against a pandas / geopandas dataframe. The idea is to try to return more simple versions of complex geometries for LineString and MultiLineString type objects.

**Returns:**

- geometry: an unbundled geometry value that can be plotted.

---

## `save(G, savename, wpath, pickle=True, edges=True, nodes=True)`

**Description:**

function used to save a graph object in a variety of handy formats

**Parameters:**

- `G` (nx.Graph): a graph object
- `savename` (str): the filename, WITHOUT extension
- `wpath` (str): the write path for where the user wants the files saved
- `pickle` (bool, optional): if set to false, will not save a pickle of the graph
- `edges` (bool, optional): if set to false, will not save an edge gdf
- `nodes` (bool, optional): if set to false, will not save a node gdf

**Returns:**

- None: saves files to the write path

---

## `add_missing_reflected_edges(G, one_way_tag=None, verbose=False)`

**Description:**

function for adding any missing reflected edges - makes all edges bidirectional. This is essential for routing with simplified graphs

**Parameters:**

- `G` (nx.Graph): a graph object
- `one_way_tag` (str, optional): if exists, then values that are True are one-way and will not be reflected
- `verbose` (bool, optional): Set to true to monitor progress of queries and notify if any queries failed, defaults to False

**Returns:**

- nx.Graph: a modified graph with the edited 'time' attribute

---

## `remove_duplicate_edges(G, max_ratio=1.5)`

**Description:**

function for deleting duplicated edges - where there is more than one edge connecting a node pair. USE WITH CAUTION - will change both topological relationships and node maps

**Parameters:**

- `G` (nx.Graph): a graph object
- `max_ratio` (float, optional): most of the time we see duplicate edges that are clones of each other. Sometimes, however, there are valid duplicates. These occur if multiple roads connect two junctions uniquely and without interruption - e.g. two roads running either side of a lake which meet at either end. The idea here is that valid 'duplicate edges' will have geometries of materially different length. Hence, we include a ratio - defaulting to 1.5 - beyond which we are sure the duplicates are valid edges, and will not be deleted.

**Returns:**

- nx.Graph: a modified graph with the edited 'time' attribute

---

## `convert_to_MultiDiGraph(G)`

**Description:**

takes any graph object, loads it into a MultiDiGraph type Networkx object

**Parameters:**

- `G` (nx.Graph): a graph object

**Returns:**

- nx.MultiDiGraph: a MultiDiGraph object

---

## `simplify_junctions(G, measure_crs, in_crs='epsg:4326', thresh=25, verbose=False)`

**Description:**

simplifies topology of networks by simplifying node clusters into single nodes.

**Parameters:**

- `G` (nx.Graph): a graph object
- `measure_crs` (str): the crs to make the measurements inself.
- `in_crs` (str, optional): the current crs of the graph's geometry properties. By default, assumes WGS 84 (epsg 4326)
- `thresh` (int, optional): the threshold distance in which to simplify junctions. By default, assumes 25 meters
- `verbose` (optional):

**Returns:**

- nx.Graph: a modified graph with simplified junctions

---

## `get_paths_to_simplify(G, strict=True)`

**Description:**

Create a list of all the paths to be simplified between endpoint nodes.

**Parameters:**

- `G` (networkx multidigraph): networkx multidigraph
- `strict` (bool, optional): if False, allow nodes to be end points even if they fail all other rules but have edges with different OSM IDs

**Returns:**

- list: paths to be simplified

---

## `is_endpoint(G, node, strict=True)`

**Description:**

Return True if the node is a "real" endpoint of an edge in the network,     otherwise False. OSM data includes lots of nodes that exist only as points     to help streets bend around curves. An end point is a node that either:     1) is its own neighbor, ie, it self-loops.     2) or, has no incoming edges or no outgoing edges, ie, all its incident         edges point inward or all its incident edges point outward.     3) or, it does not have exactly two neighbors and degree of 2 or 4.     4) or, if strict mode is false, if its edges have different OSM IDs.

**Parameters:**

- `G` (networkx multidigraph): the input graph
- `node` (int): the node to examine
- `strict` (bool, optional): if False, allow nodes to be end points even if they fail all other rules         but have edges with different OSM IDs

**Returns:**

- bool: whether the node is a real endpoint

---

## `build_path(G, node, endpoints, path)`

**Description:**

Recursively build a path of nodes until you hit an endpoint node.

**Parameters:**

- `G` (networkx multidigraph): networkx multidigraph
- `node` (int): the current node to start from
- `endpoints` (set): the set of all nodes in the graph that are endpoints
- `path` (list): the list of nodes in order in the path so far

**Returns:**

- list: paths_to_simplify

---

## `custom_simplify(G, strict=True)`

**Description:**

Simplify a graph's topology by removing all nodes that are not intersections or dead-ends. Create an edge directly between the end points that encapsulate them, but retain the geometry of the original edges, saved as attribute in new edge.

**Parameters:**

- `G` (networkx multidigraph): networkx multidigraph
- `strict` (bool, optional): if False, allow nodes to be end points even if they fail all other rules but have edges with different OSM IDs

**Returns:**

- networkx multidigraph: simplified networkx multidigraph

---

## `cut(line, distance)`

**Description:**

Cuts a line in two at a distance from its starting point

**Parameters:**

- `line` (LineString): a shapely LineString object
- `distance` (float): distance from start of line to cut

**Returns:**

- list: list of two LineString objects

---

## `salt_long_lines(G, source, target, thresh=5000, factor=1, attr_list=None, geometry_tag='Wkt')`

**Description:**

Adds in new nodes to edges greater than a given length

**Parameters:**

- `G` (nx.Graph): a graph object
- `source` (str): crs object in format 'epsg:4326'
- `target` (str): crs object in format 'epsg:32638'
- `thresh` (int, optional): distance in metres after which to break edges.
- `factor` (int, optional): edge lengths can be returned in units other than metres by specifying a numerical multiplication factor. Factor behavior divides rather than multiplies.
- `attr_list` (optional):
- `geometry_tag` (optional):

**Returns:**

- nx.Graph: a modified graph with the edited 'time' attribute

---

## `pandana_snap(G, point_gdf, source_crs='epsg:4326', target_crs='epsg:4326', add_dist_to_node_col=True, time_it=False)`

**Description:**

snaps points to a graph at very high speed

**Parameters:**

- `G` (nx.Graph): a graph object, or the node geodataframe of a graph
- `point_gdf` (gpd.GeoDataFrame): a geodataframe of points, in the same source crs as the geometry of the graph object
- `source_crs` (str, optional): The crs for the input G and input point_gdf in format 'epsg:32638'
- `target_crs` (str, optional): The measure crs how distances between points are calculated. The returned point GeoDataFrame's CRS does not get modified. The crs object in format 'epsg:32638'
- `add_dist_to_node_col` (bool, optional): return distance to nearest node in the units of the target_crs
- `time_it` (optional):

**Returns:**

- GeoDataFrame: returns a GeoDataFrame that is the same as the input point_gdf but adds a column containing the id of the nearest node in the graph, and the distance if add_dist_to_node_col is True

---

## `pandana_snap_c(G, point_gdf, source_crs='epsg:4326', target_crs='epsg:4326', add_dist_to_node_col=True, time_it=False)`

**Description:**

snaps points to a graph at a faster speed than pandana_snap.

**Parameters:**

- `G` (nx.Graph): a graph object, or the node geodataframe of a graph
- `point_gdf` (gpd.GeoDataFrame): a geodataframe of points, in the same source crs as the geometry of the graph object
- `source_crs` (str, optional): The crs for the input G and input point_gdf in format 'epsg:32638'
- `target_crs` (str, optional): The measure crs how distances between points are calculated. The returned point GeoDataFrame's CRS does not get modified. The crs object in format 'epsg:32638'
- `add_dist_to_node_col` (bool, optional): return distance to nearest node in the units of the target_crs
- `time_it` (bool, optional): return time to complete function

**Returns:**

- GeoDataFrame: returns a GeoDataFrame that is the same as the input point_gdf but adds a column containing the id of the nearest node in the graph, and the distance if add_dist_to_node_col is True

---

## `pandana_snap_to_many(G, point_gdf, source_crs='epsg:4326', target_crs='epsg:4326', add_dist_to_node_col=True, time_it=False, k_nearest=5, origin_id='index')`

**Description:**

snaps points their k nearest neighbors in the graph.

**Parameters:**

- `G` (nx.Graph): a graph object
- `point_gdf` (gpd.GeoDataFrame): a geodataframe of points, in the same source crs as the geometry of the graph object
- `source_crs` (str, optional): The crs for the input G and input point_gdf in format 'epsg:32638'
- `target_crs` (str, optional): The desired crs returned point GeoDataFrame. The crs object in format 'epsg:32638'
- `add_dist_to_node_col` (bool, optional): return distance to nearest node in the units of the target_crs
- `time_it` (bool, optional): return time to complete function
- `k_nearest` (int, optional): k nearest neighbors to query for the nearest node
- `origin_id` (str, optional): key for the id in the points_gdf input, defaults to 'index'

**Returns:**

- dict: returns a dictionary of the k nearest nodes to each origin point

---

## `pandana_snap_single_point(G, shapely_point, source_crs='epsg:4326', target_crs='epsg:4326')`

**Description:**

snaps a point to a graph at very high speed

**Parameters:**

- `G` (nx.Graph): a graph object
- `shapely_point` (shapely Point): a shapely point (ex. Point(x, y)), in the same source crs as the geometry of the graph object
- `source_crs` (str, optional): crs object in format 'epsg:32638'
- `target_crs` (str, optional): crs object in format 'epsg:32638'

**Returns:**

- object: returns the id of the nearest node in the graph, could be an integer, float, string, or really any object

---

## `pandana_snap_points(source_gdf, target_gdf, source_crs='epsg:4326', target_crs='epsg:4326', add_dist_to_node_col=True)`

**Description:**

snaps points to another GeoDataFrame at very high speed

**Parameters:**

- `source_gdf` (gpd.GeoDataFrame): a geodataframe of points
- `target_gdf` (gpd.GeoDataFrame): a geodataframe of points, in the same source crs as the geometry of the source_gdfsg:32638'
- `source_crs` (str, optional): crs object in format 'epsg:32638', by default 'epsg:4326'
- `target_crs` (str, optional): crs object in format 'epsg:32638', by default 'epsg:4326'
- `add_dist_to_node_col` (bool, optional): return distance in metres to nearest node

**Returns:**

- return: returns a GeoDataFrame that is the same as the input source_gdf but adds a column containing the id of the nearest node in the target_gdf, and the distance if add_dist_to_node_col is True

---

## `join_networks(base_net, new_net, measure_crs, thresh=500)`

**Description:**

joins two networks together within a binding threshold

**Parameters:**

- `base_net` (nx.MultiDiGraph): a base network object (nx.MultiDiGraph)
- `new_net` (nx.MultiDiGraph): the network to add on to the base (nx.MultiDiGraph)
- `measure_crs` (int): the crs number of the measurement (epsg code)
- `thresh` (int, optional): binding threshold - unit of the crs - default 500m

**Returns:**

- nx.MultiDiGraph: returns a new network object that is the combination of the two input networks

---

## `clip(G, bound, source_crs='epsg:4326', target_crs='epsg:4326', geom_col='geometry', largest_G=True)`

**Description:**

Removes any edges that fall beyond a polygon, and shortens any other edges that do so

**Parameters:**

- `G` (nx.MultiDiGraph): a graph object.
- `bound` (shapely Polygon or MultiPolygon): a shapely polygon object
- `source_crs` (str, optional): crs object in format 'epsg:4326'
- `target_crs` (str, optional): crs object in format 'epsg:4326'
- `geom_col` (str, optional): label name for geometry object
- `largest_G` (bool, optional): if True, takes largest remaining subgraph of G as G

**Returns:**

- nx.MultiDiGraph: returns a new graph object that is the clipped version of the input graph

---

## `new_edge_generator(passed_geom, infra_type, iterator, existing_legitimate_point_geometries, geom_col, project_WGS_UTM)`

**Description:**

Generates new edge and node geometries based on a passed geometry. WARNING: This is a child process of clip(), and shouldn't be run on its own

**Parameters:**

- `passed_geom` (shapely LineString): a shapely Linestring object
- `infra_type` (str): the road / highway class of the passed geometry
- `iterator` (int): helps count the new node IDs to keep unique nodes
- `existing_legitimate_point_geometries` (dict): a dictionary of points already created / valid in [u:geom] format
- `geom_col` (str): label name for geometry object
- `project_WGS_UTM` (object): projection object to transform passed geometries

**Returns:**

- list: returns a list of new nodes and edges to be added to the graph

---

## `project_gdf(gdf, to_crs=None, to_latlong=False)`

**Description:**

Taken from OSMNX

**Parameters:**

- `gdf` (geopandas.GeoDataFrame): geopandas.GeoDataFrame the GeoDataFrame to be projected
- `to_crs` (str, optional): string or pyproj. CRS if None, project to UTM zone in which gdf's centroid lies, otherwise project to this CRS
- `to_latlong` (bool, optional): bool if True, project to settings.default_crs and ignore to_crs

**Returns:**

- geopandas.GeoDataFrame: the projected GeoDataFrame

---

## `gn_project_graph(G, to_crs=None)`

**Description:**

Taken from OSMNX. Project graph from its current CRS to another.

**Parameters:**

- `G` (networkx.MultiDiGraph): networkx.MultiDiGraph the graph to be projected
- `to_crs` (str, optional): string or pyproj.CRS if None, project graph to UTM zone in which graph centroid lies, otherwise project graph to this CRS

**Returns:**

- networkx.MultiDiGraph: the projected graph

---

## `euclidean_distance(lat1, lon1, lat2, lon2)`

**Description:**

Calculate the great circle distance between two points on the earth (specified in decimal degrees)

**Parameters:**

- `lat1` (float): lat1
- `lon1` (float): lon1
- `lat2` (float): lat2
- `lon2` (float): lon2

**Returns:**

- float: returns the distance between two points in km

---

## `utm_of_graph(G)`

**Description:**

Calculates the UTM coordinate reference system (CRS) for a given graph.

**Parameters:**

- `G` (networkx.Graph): The input graph.

**Returns:**

- str: The UTM CRS string.

---

## `find_kne(point, lines, near_idx)`

**Description:**

Find the nearest edge (kne) to a given point from a set of lines.

**Parameters:**

- `point` (Point): The point for which to find the nearest edge.
- `lines` (GeoDataFrame): The set of lines representing edges.
- `near_idx` (array-like): The array-like object containing the indices of the nearest edges.

**Returns:**

- kne_idx: int The index of the nearest edge.
- kne: Series The geometry of the nearest edge.

---

## `get_pp(point, line)`

**Description:**

Get the projected point (pp) of 'point' on 'line'.

**Parameters:**

- `point` (Point): The point to be projected.
- `line` (LineString): The line on which the point is projected.

**Returns:**

- Point: The projected point on the line.

---

## `split_line(line, pps)`

**Description:**

Split 'line' by all intersecting 'pps' (as multipoint).

**Parameters:**

- `line` (LineString): The line to be split.
- `pps` (MultiPoint): The multipoint object containing all the points at which to split the line.

**Returns:**

- list: a list of all line segments after the split

---

## `update_nodes(nodes, new_points, ptype, road_col, node_key_col, poi_key_col, node_highway_pp, node_highway_poi, measure_crs='epsg:3857', osmid_prefix=9990000000)`

**Description:**

Update nodes with a list (pp) or a GeoDataFrame (poi) of new_points.

**Parameters:**

- `nodes` (GeoDataFrame): The original nodes GeoDataFrame.
- `new_points` (GeoDataFrame or list): The new points to be added to the nodes.
- `ptype` (str): type of Point list to append, 'pp' or 'poi'
- `road_col`:
- `node_key_col`:
- `poi_key_col`:
- `node_highway_pp`:
- `node_highway_poi`:
- `measure_crs` (str, optional): the crs of the measure (epsg code)
- `osmid_prefix` (optional):

**Returns:**

- GeoDataFrame: The updated nodes GeoDataFrame.

---

## `update_edges(edges, new_lines, replace=True, nodes_meter=None, pois_meter=None)`

**Description:**

Update edge info by adding new_lines; or,

**Parameters:**

- `edges` (GeoDataFrame): The original edges GeoDataFrame.
- `new_lines` (list): The new line segments to be added to the edges.
- `replace` (bool, optional): treat new_lines (flat list) as newly added edges if False, else replace existing edges with new_lines (often a nested list)
- `nodes_meter` (GeoDataFrame, optional): The nodes GeoDataFrame.
- `pois_meter` (GeoDataFrame, optional): The POIs GeoDataFrame.

**Returns:**

- GeoDataFrame: The updated edges GeoDataFrame.
- Note: kne_idx refers to 'fid in Rtree'/'label'/'loc', not positional iloc

---

## `nearest_edge(row, Rtree, knn, edges_meter)`

**Description:**

Finds the nearest edge(s) to a given point.

**Parameters:**

- `row` (pandas.Series): A row containing the point geometry.
- `Rtree`:
- `knn`:
- `edges_meter`:

**Returns:**

- tuple: A tuple containing the indices of the nearest edges and their corresponding geometries.

---

## `advanced_snap(G, pois, u_tag='stnode', v_tag='endnode', node_key_col='osmid', edge_key_col='osmid', poi_key_col=None, road_col='highway', oneway_tag='oneway', path=None, threshold=500, knn=5, measure_crs='epsg:3857', factor=1, verbose=False)`

**Description:**

Connect and integrate a set of POIs into an existing road network.

**Parameters:**

- `G`:
- `pois` (GeoDataFrame): a gdf of POI (geom: Point)
- `u_tag` (optional):
- `v_tag` (optional):
- `node_key_col` (str, optional): The node tag id in the returned graph
- `edge_key_col` (str, optional): The edge tag id in the returned graph
- `poi_key_col` (str, optional): The tag to be used for oneway edges
- `road_col` (optional):
- `oneway_tag` (optional):
- `path` (str, optional): directory path to use for saving optional shapefiles (nodes and edges). Outputs will NOT be saved if this arg is not specified.
- `threshold` (int, optional): the max length of a POI connection edge, POIs withconnection edge beyond this length will be removed. The unit is in meters as crs epsg is set to 3857 by default during processing.
- `knn` (int, optional): k nearest neighbors to query for the nearest edge. Consider increasing this number up to 10 if the connection output is slightly unreasonable. But higher knn number will slow down the process.
- `measure_crs` (int, optional): preferred EPSG in meter units. Suggested to use the correct UTM projection.
- `factor` (int, optional): allows you to scale up / down unit of returned new_footway_edges if other than meters. Set to 1000 if length in km.
- `verbose` (optional):

**Returns:**

- graph: the original gdf with POIs and PPs appended and with connection edges appended and existing edges updated (if PPs are present)pois_meter (GeoDataFrame): gdf of the POIs along with extra columns, such as the associated nearest lines and PPs new_footway_edges (GeoDataFrame): gdf of the new footway edges that connect the POIs to the original graph

---

## `add_intersection_delay(G, intersection_delay=7, time_col='time', highway_col='highway', filter=['projected_footway', 'motorway'])`

**Description:**

Find node intersections. For all intersection nodes, if directed edge is going into the intersection then add delay to the edge.

**Parameters:**

- `G` (nx.MultiDiGraph): a base network object (nx.MultiDiGraph)
- `intersection_delay` (int, optional): The number of seconds to delay travel time at intersections
- `time_col` (optional):
- `highway_col` (optional):
- `filter` (list, optional): The filter is a list of highway values where the type of highway does not get an intersection delay.

**Returns:**

- nx.MultiDiGraph: a base network object (nx.MultiDiGraph)

---

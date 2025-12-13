# network_clean.py

## `clean_network(G, wpath='', output_file_name='', UTM='epsg:3857', WGS='epsg:4326', junctdist=50, verbose=False)`

**Description:**

Topologically simplifies an input graph object by collapsing junctions and removing interstital nodes

**Parameters:**

- `G` (networkx.graph object): a graph object containing nodes and edges. Edges should have a property called 'Wkt' containing geometry objects describing the roads.
- `wpath` (str, optional): the write path - a drive directory for inputs and output
- `output_file_name` (str, optional): This will be the output file name with '_processed' appended
- `UTM` (dict, optional): The epsg code of the projection, in metres, to apply the junctdist
- `WGS` (dict, optional): the current crs of the graph's geometry properties. By default, assumes WGS 84 (epsg 4326)
- `junctdist` (int, float, optional): distance within which to collapse neighboring nodes. simplifies junctions. Set to 0.1 if not simplification desired. 50m good for national (primary / secondary) networks
- `verbose` (boolean, optional): if True, saves down intermediate stages for dissection

**Returns:**

- nx.MultiDiGraph: A simplified graph object

---

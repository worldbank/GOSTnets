# graphtool.py

## `get_prop_type(value, key=None)`

**Description:**

Performs typing and value conversion for the graph_tool PropertyMap class.

**Parameters:**

- `value` (any): The value to be typed and converted
- `key` (any, optional): The key to be typed and converted, if provided, defaults to None

**Returns:**

- tuple: A tuple of the type name, value, and key

---

## `nx2gt(nxG)`

**Description:**

Converts a networkx graph to a graph-tool graph.

**Parameters:**

- `nxG` (networkx.Graph): The networkx graph to be converted.

**Returns:**

- gtG: graph_tool.Graph The graph-tool graph converted from the networkx graph.

---

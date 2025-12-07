# fetch_pois.py

## Class OsmObject

Represents an object for fetching and processing OpenStreetMap Points of Interest (POIs).

### `OsmObject.RelationtoPoint(string)`

**Description**
Converts a relation geometry to a point geometry.

**Parameters**
- `string` (shapely.geometry): The relation geometry to be converted.

**Returns**
- shapely.geometry.Point: The centroid of the relation geometry if it is a Polygon, otherwise the centroid of the MultiPolygon formed by the relation geometry's constituent geometries.

---

### `OsmObject.GenerateOSMPOIs()`

**Description**
Generates OpenStreetMap Points of Interest (POIs) within a given bounding box.

**Parameters**
- None

**Returns**
- pandas.DataFrame: A DataFrame containing the generated POIs.

---

### `OsmObject.RemoveDupes(buf_width, crs='epsg:4326')`

**Description**
Remove duplicate geometries from the GeoDataFrame.

**Parameters**
- `buf_width` (float): The buffer width used for checking intersection.
- `crs` (str, optional): The coordinate reference system. Defaults to "epsg:4326".

**Returns**
- pandas.DataFrame: The GeoDataFrame with duplicate geometries removed.

---

### `OsmObject.prepForMA()`

**Description**
Prepare results data frame for use in the OSRM functions in OD.

**Parameters**
- None

**Returns**
- pandas.DataFrame: The modified data frame with added fields and removed geometry fields.

---

### `OsmObject.Save(outFolder)`

**Description**
Save the dataframe as a CSV file in the specified output folder.

**Parameters**
- `outFolder` (str): The name of the output folder.

**Returns**
- None

---

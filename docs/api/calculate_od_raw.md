# calculate_od_raw.py

### `calculateOD_gdf(G, origins, destinations, fail_value=-1, weight='time', calculate_snap=False, wgs84='epsg:4326')`

**Description**
Calculate Origin destination matrix from GeoDataframes

**Parameters**
- `G` (networkx graph): describes the road network. Often extracted using OSMNX
- `origins` (geopandas dataframe): source locations for calculating access
- `destinations` (geopandas dataframe): destination locations for calculating access
- `fail_value` (optional): 
- `weight` (optional): 
- `calculate_snap` (boolean, optional): variable to add snapping distance to travel time, default is false
- `wgs84` (CRS dictionary, optional): CRS of road network to which the GDFs are projected

**Returns**
- numpy array: 2d OD matrix with columns as index of origins and rows as index of destinations

---

### `calculateOD_csv(G, originCSV, destinationCSV='', oLat='Lat', oLon='Lon', dLat='Lat', dLon='Lon', crs='epsg:4326', fail_value=-1, weight='time', calculate_snap=False)`

**Description**
Calculate OD matrix from csv files of points

**Parameters**
- `G`: describes the road network. Often extracted using OSMNX
- `originCSV`: 
- `destinationCSV` (optional): 
- `oLat` (str, optional): Origin latitude field
- `oLon` (str, optional): Origin longitude field
- `dLat` (str, optional): Destination latitude field
- `dLon` (str, optional): Destination longitude field
- `crs` (str, optional): crs of input origins and destinations, defaults to 'epsg:4326'
- `fail_value` (optional): 
- `weight` (str, optional): variable in G used to define edge impedance, defaults to 'time'
- `calculate_snap` (bool, optional): variable to add snapping distance to travel time, default is false

**Returns**
- numpy array: 2d OD matrix with columns as index of origins and rows as index of destinations

---

### `calculate_gravity(od, oWeight=[], dWeight=[], decayVals=[0.01, 0.005, 0.001, 0.0007701635, 0.0003850818, 0.0001925409, 9.62704e-05, 3.85082e-05, 1e-05])`

**Description**
Calculate gravity model values for origin-destination (OD) matrix.

**Parameters**
- `od` (numpy.ndarray): Origin-destination matrix.
- `oWeight` (list, optional): List of weights for each origin. Defaults to an empty list.
- `dWeight` (list, optional): List of weights for each destination. Defaults to an empty list.
- `decayVals` (list, optional): List of decay values for market access. Defaults to a predefined list.

**Returns**
- pandas.DataFrame: DataFrame containing gravity model values for each origin-destination pair.

---

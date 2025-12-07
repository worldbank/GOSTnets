# fetch_od.py

### `CreateODMatrix(infile, infile_2, lat_name='Lat', lon_name='Lon', UID='ID', Pop=None, call_type='OSRM', rescue=0, rescue_num=0, MB_Token='', sleepTime=5, osrmHeader='')`

**Description**
Create an Origin-Destination matrix from a list of origins and destinations.

**Parameters**
- `infile` (string or geodataframe): string for folder path containing input data of the origins. This can also be a geodataframe of the data instead.
- `infile_2` (string or geodataframe): string for folder path containing input data of the destinations. This can also be a geodataframe of the data instead.
- `lat_name` (string, optional): Latitude column name.
- `lon_name` (string, optional): Longitude column name
- `UID` (string, optional): Origin Unique Identifier column name (e.g. District, Name, Object ID...). This is mainly helpful for joining the output back to the input data / a shapefile, and is non-essential in terms of the calculation. It can be text or a number.
- `Pop` (string, optional): Population / weighting column name
- `call_type` (string, optional): Server call type - "OSRM" for OSRM, "MB" for Mapbox, "MBT" for Mapbox traffic, or "Euclid" for Euclidean distances (as the crow flies)
- `rescue` (int, optional): Save - input latest save number to pick up matrix construction process from there.
- `rescue_num` (int, optional): Rescue number parameter - If you have already re-started the download process, denote how many times. First run = 0, restarted once = 1...
- `MB_Token` (string, optional): Mapbox private key if using the "MB" or "MBT" call types
- `sleepTime` (int, optional): When making calls to OSRM, a sleep time is required to avoid DDoS
- `osrmHeader` (string, optional): optional parameter to set OSRM source

**Returns**
- pandas.DataFrame: DataFrame containing the OD matrix.

---

### `MarketAccess(new, lambder_list=[0.01, 0.005, 0.001, 0.0007701635, 0.0003850818, 0.0001925409, 9.62704e-05, 3.85082e-05, 1e-05])`

**Description**
Calculate Market Access for a given range of lambdas.

**Parameters**
- `new` (pd.DataFrame): DataFrame containing the data for market access calculation.
- `lambder_list` (list, optional): List of lambda values to be used for market access calculation.

**Returns**
- pd.DataFrame: DataFrame containing the market access values for each lambda.

---

### `ReadMe(ffpath)`

**Description**


**Parameters**
- `ffpath`: 

**Returns**

---

# optimization.py

### `optimize_facility_locations(OD, facilities, p, existing_facilities=None, verbose=False, execute=True, write='')`

**Description**
Function for identifying spatially optimal locations of facilities (P-median problem)

**Parameters**
- `OD` (pd.DataFrame): an Origin:Destination matrix, origins as rows, destinations as columns, in pandas DataFrame format.
- `facilities` (list): The 'destinations' of the OD-Matrix. MUST be a list of objects included in OD.columns (or subset) if certain nodes are unsuitable for facility locations
- `p` (int): the number of facilities to solve for
- `existing_facilities` (list, optional): facilities to always include in the solution. MUST be in 'facilities' list
- `verbose` (bool, optional): print a bunch of status updates
- `execute` (bool, optional): should the problem be executed
- `write` (str, optional): outPath to write problem

**Returns**
- ans: list a list of the optimal facility locations

---

### `optimize_set_coverage(OD, max_coverage=2000, existing_facilities=None)`

**Description**
Determine the minimum number of facilities and their locations in order to cover all demands within a pre-specified maximum distance (or time) coverage (Location Set-Covering Problem).

**Parameters**
- `OD` (pd.DataFrame): An Origin:Destination matrix, origins as rows, destinations as columns, in pandas DataFrame format.
- `max_coverage` (int, optional): The pre-specified maximum distance (or time) coverage.
- `existing_facilities` (list, optional): Facilities to always include in the solution. Must be in 'facilities' list.

**Returns**
- list: A list of facility locations that provide coverage to all demands within the maximum coverage distance.

---

### `optimize_partial_set_coverage(OD, pop_coverage=0.8, max_coverage=2000, origins_pop_series=None, existing_facilities=None)`

**Description**
Function to determine the minimum number of facilities and their locations in order to cover a given fraction of the population within a pre-specified maximum distance (or time) coverage (Partial Set-Covering Problem). Do not use a demand-weighted OD matrix as an input.

**Parameters**
- `OD` (pandas.DataFrame): An Origin:Destination matrix, origins as rows, destinations as columns.
- `pop_coverage` (float, optional): The given fraction of the population that should be covered. Defaults to 0.8.
- `max_coverage` (int, optional): The pre-specified maximum distance (or time) coverage. Defaults to 2000.
- `origins_pop_series` (pandas.Series, optional): A series that contains each origin as the key, and each origin's population as the value. Defaults to None.
- `existing_facilities` (list, optional): Facilities to always include in the solution. Defaults to None.

**Returns**
- list: A list of facility locations that cover the specified fraction of the population.

---

### `optimize_max_coverage(OD, p_facilities=5, max_coverage=2000, origins_pop_series=None, existing_facilities=None)`

**Description**
Determine the location of P facilities in order to maximize the demand covered within a pre-specified maximum distance coverage (Max Cover).

**Parameters**
- `OD` (pandas.DataFrame): An Origin:Destination matrix, origins as rows, destinations as columns.
- `p_facilities` (int, optional): The number of facilities to locate. Default is 5.
- `max_coverage` (int, optional): The pre-specified maximum distance (or time) coverage. Default is 2000.
- `origins_pop_series` (pandas.Series, optional): Series containing population data for each origin. Default is None.
- `existing_facilities` (list, optional): List of facilities to always include in the solution. Default is None.

**Returns**
- list: List of facility locations that maximize the demand covered.

---

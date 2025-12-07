# conversion_utils.py

### `rasterize_od_results(inD, outFile, field, template=None)`

**Description**
Convert gridded point data frame to raster of commensurate size and resolution

**Parameters**
- `inD` (geopandas data frame): OD matrix as point data frame
- `outFile` (string): path to save output raster
- `field` (string): field to rasterize
- `template` (string, optional): path to raster template file, by default None

**Returns**
- None

---

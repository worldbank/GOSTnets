import os

import shapely

import geopandas as gpd
import osmnx as ox  ### Make sure to install osmnx with -c conda-forge to get newest version
import pandas as pd

from shapely.geometry import box
from shapely.ops import unary_union


class OsmObject:
    """
    Represents an object for fetching and processing OpenStreetMap Points of Interest (POIs).

    Attributes
    ----------
    tags : dict
        A dictionary of tags used to filter the POIs.
    name : str
        The name of the amenity.
    bbox : shapely Polygon
        The area within which to search for POIs.
    path : str
        The output folder where results are saved.

    Methods
    -------
    RelationtoPoint(string): Converts a relation geometry to a point geometry.
    GenerateOSMPOIs(): Generates the OSM POIs within the specified area.
    RemoveDupes(buf_width, crs): Removes duplicate POIs within a buffer width.
    prepForMA(): Prepares the results data frame for use in OSRM functions.
    Save(outFolder): Saves the POIs to a CSV file in the specified output folder.

    Examples
    --------
    >>> education = {'amenity':['school', 'kindergarten','university', 'college']}
    >>> health = {'amenity':['clinic', 'pharmacy', 'hospital', 'health']}
    >>> crs = 'epsg:4326'
    >>> buf_width = 0.0005
    >>> for a in amenities:
    ...     curr_amenity = amenities[a]
    ...     current = AmenityObject(a, bbox, tags, path)
    ...     current.GenerateOSMPOIs()
    ...     current.RemoveDupes(buf_width, crs)
    ...     current.Save(a)
    """

    def __init__(self, a, poly, tags, path=""):
        """
        Initialize the OsmObject class.

        Parameters
        ----------
        a : string
            name of the amenity
        poly : Shapely Polygon
            area within which to search for POIs
        tags : list of strings
            list of official OSM features to extract
        path : string
            outFolder where results are saved

        """
        self.tags = tags
        self.name = a
        self.bbox = poly
        self.path = path

    def RelationtoPoint(self, string):
        """
        Converts a relation geometry to a point geometry.

        Parameters
        ----------
        string : shapely.geometry
            The relation geometry to be converted.

        Returns
        -------
        shapely.geometry.Point
            The centroid of the relation geometry if it is a Polygon,
            otherwise the centroid of the MultiPolygon formed by the relation
            geometry's constituent geometries.

        """
        lats, lons = [], []

        # It is possible that a relation might be a Polygon instead of a MultiPolygon
        if isinstance(string, shapely.geometry.polygon.Polygon):
            return string.centroid

        for i in string.geoms:
            lons.append(i.bounds[0])
            lats.append(i.bounds[1])
            lons.append(i.bounds[2])
            lats.append(i.bounds[3])

        point = box(min(lons), min(lats), max(lons), max(lats)).centroid

        return point

    def GenerateOSMPOIs(self):
        """
        Generates OpenStreetMap Points of Interest (POIs) within a given bounding box.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the generated POIs.

        """
        # old way in OSMNX
        # df = ox.pois_from_polygon(polygon = self.bbox, amenities = self.tags)

        # note that even as of Dec, 2020 the code below will be depreciated, as OSMNX deleted the poi modeule in favor of the new geometries module
        # df = ox.pois_from_polygon(polygon = self.bbox, tags = {'amenity':self.current_amenity} )

        df = ox.geometries_from_polygon(self.bbox, self.tags).reset_index()

        print(f"is df empty: {df.empty}")
        if df.empty is True:
            return df

        points = df.copy()
        points = points.loc[points["element_type"] == "node"]

        polygons = df.copy()
        polygons = polygons.loc[polygons["element_type"] == "way"]
        polygons["geometry"] = polygons.centroid

        multipolys = df.copy()
        multipolys = multipolys.loc[multipolys["element_type"] == "relation"]
        multipolys["geometry"] = multipolys["geometry"].apply(
            lambda x: self.RelationtoPoint(x)
        )

        df = pd.concat(
            [pd.DataFrame(points), pd.DataFrame(polygons), pd.DataFrame(multipolys)],
            ignore_index=True,
        )

        self.df = df
        return df

    def RemoveDupes(self, buf_width, crs="epsg:4326"):
        """
        Remove duplicate geometries from the GeoDataFrame.

        Parameters
        ----------
        buf_width : float
            The buffer width used for checking intersection.
        crs : str, optional
            The coordinate reference system. Defaults to "epsg:4326".

        Returns
        -------
        pandas.DataFrame
            The GeoDataFrame with duplicate geometries removed.

        """
        df = self.df
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        gdf["buffer"] = gdf["geometry"].buffer(buf_width)
        df_l = pd.DataFrame()
        for i in gdf.index:
            row = gdf.loc[i]
            if len(df_l) == 0:
                df_l = pd.concat([df_l, row.to_frame().T], ignore_index=True)
            else:
                current_points = unary_union(df_l["buffer"])
                if row["buffer"].intersects(current_points):
                    pass
                else:
                    df_l = pd.concat([df_l, row.to_frame().T], ignore_index=True)
        gdf = gdf.to_crs(crs)
        self.df = df_l
        return df_l

    def prepForMA(self):
        """
        Prepare results data frame for use in the OSRM functions in OD.

        Steps:
        1. Add 'Lat' and 'Lon' fields based on the x and y coordinates of the geometry.
        2. Add a unique identifier 'mID' to each row.
        3. Remove the 'geometry' and 'buffer' fields from the data frame.

        Returns
        -------
        pandas.DataFrame
            The modified data frame with added fields and removed geometry fields.
        """

        def tryLoad(x):
            """
            Tries to load the x-coordinate and y-coordinate from an object.

            Parameters
            ----------
            x : object
                An object with x and y attributes.

            Returns
            -------
            list
                A list containing the x-coordinate and y-coordinate of the object.
                If the object does not have x and y attributes, [0, 0] is returned.
            """
            try:
                return [x.x, x.y]
            except Exception:
                return [0, 0]

        curDF = self.df
        allShapes = [tryLoad(x) for x in curDF.geometry]
        Lon = [x[0] for x in allShapes]
        Lat = [x[1] for x in allShapes]
        curDF["Lat"] = Lat
        curDF["Lon"] = Lon
        curDF["mID"] = range(0, curDF.shape[0])
        curDF = curDF.drop(["geometry", "buffer"], axis=1)
        return curDF

    def Save(self, outFolder):
        """
        Save the dataframe as a CSV file in the specified output folder.

        Parameters
        ----------
        outFolder : str
            The name of the output folder.

        Returns
        -------
        None
        """
        out = os.path.join(self.path, outFolder)
        if not os.path.exists(out):
            os.mkdir(out)
        self.df.to_csv(os.path.join(out, "%s.csv" % self.name), encoding="utf-8")

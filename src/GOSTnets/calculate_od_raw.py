import pandas as pd
import geopandas as gpd
import numpy as np

from shapely.geometry import Point
from .core import pandana_snap
from .core import calculate_OD as calc_od


def calculateOD_gdf(
    G,
    origins,
    destinations,
    fail_value=-1,
    weight="time",
    calculate_snap=False,
    wgs84="epsg:4326",
):
    """Calculate Origin destination matrix from GeoDataframes

    Parameters
    ----------
    G : networkx graph
        describes the road network. Often extracted using OSMNX
    origins : geopandas dataframe
        source locations for calculating access
    destinations : geopandas dataframe
        destination locations for calculating access
    calculate_snap : boolean, optional
        variable to add snapping distance to travel time, default is false
    wgs84 : CRS dictionary, optional
        CRS of road network to which the GDFs are projected

    Returns
    -------
    numpy array
        2d OD matrix with columns as index of origins and rows as index of
        destinations
    """
    # Get a list of originNodes and destinationNodes
    if origins.crs != wgs84:
        origins = origins.to_crs(wgs84)
    if destinations.crs != wgs84:
        destinations = destinations.to_crs(wgs84)
    origins = pandana_snap(G, origins)
    destinations = pandana_snap(G, destinations)
    oNodes = origins["NN"].unique()
    dNodes = destinations["NN"].unique()
    od = calc_od(G, oNodes, dNodes, fail_value)
    origins["OD_O"] = origins["NN"].apply(lambda x: np.where(oNodes == x)[0][0])
    destinations["OD_D"] = destinations["NN"].apply(
        lambda x: np.where(dNodes == x)[0][0]
    )
    outputMatrix = od[origins["OD_O"].values][destinations["OD_D"].values]
    if calculate_snap:
        originsUTM = pandana_snap(G, origins, target_crs="epsg:3857")
        destinationsUTM = pandana_snap(G, destinations, target_crs="epsg:3857")
        originsUTM["tTime_sec"] = (
            originsUTM["NN_dist"] / 1000 / 5 * 60 * 60
        )  # Convert snap distance to walking time in seconds
        destinationsUTM["tTime_sec"] = (
            destinationsUTM["NN_dist"] / 1000 / 5 * 60 * 60
        )  # Convert snap distance to walking time in seconds
        originsUTM.reset_index(inplace=True)
        for idx, row in originsUTM.iterrows():
            outputMatrix[idx, :] = outputMatrix[idx, :] + row["tTime_sec"]
        outputMatrix = outputMatrix
    return outputMatrix


def calculateOD_csv(
    G,
    originCSV,
    destinationCSV="",
    oLat="Lat",
    oLon="Lon",
    dLat="Lat",
    dLon="Lon",
    crs="epsg:4326",
    fail_value=-1,
    weight="time",
    calculate_snap=False,
):
    """
    Calculate OD matrix from csv files of points

    Parameters
    ----------
    G :
        describes the road network. Often extracted using OSMNX
    origins : str
        path to csv file with locations for calculating access
    destinations : str
        path to csv with destination locations for calculating access
    oLat : str
        Origin latitude field
    oLon : str
        Origin longitude field
    dLat : str
        Destination latitude field
    dLon : str
        Destination longitude field
    crs : str
        crs of input origins and destinations, defaults to 'epsg:4326'
    fail-value : int
        value to put in OD matrix if no route found, defaults to -1
    weight : str
        variable in G used to define edge impedance, defaults to 'time'
    calculate_snap : bool
        variable to add snapping distance to travel time, default is false

    Returns
    -------
    numpy array
        2d OD matrix with columns as index of origins and rows as index of
        destinations
    """

    originPts = pd.read_csv(originCSV)
    pts = [Point(x) for x in zip(originPts[oLon], originPts[oLat])]
    originGDF = gpd.GeoDataFrame(originPts, geometry=pts, crs=crs)

    if destinationCSV == "":
        destinationGDF = originGDF.copy()
    else:
        originPts = pd.read_csv(destinationCSV)
        pts = [Point(x) for x in zip(originPts[dLon], originPts[dLat])]
        destinationGDF = gpd.GeoDataFrame(originPts, geometry=pts, crs=crs)
    OD = calculateOD_gdf(
        G, originGDF, destinationGDF, fail_value, weight, calculate_snap=calculate_snap
    )

    return OD


def calculate_gravity(
    od,
    oWeight=[],
    dWeight=[],
    decayVals=[
        0.01,
        0.005,
        0.001,
        0.0007701635,  # Market access halves every 15 mins
        0.0003850818,  # Market access halves every 30 mins
        0.0001925409,  # Market access halves every 60 mins
        0.0000962704,  # Market access halves every 120 mins
        0.0000385082,  # Market access halves every 300 mins
        0.00001,
    ],
):
    """
    Calculate gravity model values for origin-destination (OD) matrix.

    Parameters
    ----------
    od : numpy.ndarray
        Origin-destination matrix.
    oWeight : list, optional
        List of weights for each origin. Defaults to an empty list.
    dWeight : list, optional
        List of weights for each destination. Defaults to an empty list.
    decayVals : list, optional
        List of decay values for market access. Defaults to a predefined list.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing gravity model values for each origin-destination
        pair.
    """
    if len(oWeight) != od.shape[0]:
        oWeight = [1] * od.shape[0]
    if len(dWeight) != od.shape[1]:
        dWeight = [1] * od.shape[1]
    allRes = []

    od_df = pd.DataFrame(od)

    def decayFunction(x, decay):
        return np.exp(-1 * decay * x)

    for dist_decay in decayVals:
        summedVals = (
            np.sum(decayFunction(od_df, dist_decay) * dWeight, axis=1) * oWeight
        )

        allRes.append(summedVals)

    res = pd.DataFrame(allRes).transpose()
    res.columns = ["d_%s" % d for d in decayVals]

    return res

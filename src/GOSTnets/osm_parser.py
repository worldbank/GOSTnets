"""
source: https://gist.github.com/Tofull/49fbb9f3661e376d2fe08c2e9d64320e
"""

## Modules
# Elementary modules
from math import radians, cos, sin, asin, sqrt
import copy

# Graph module
import networkx

# Specific modules
import xml.sax  # parse osm file
from pathlib import Path  # manage cached tiles

try:
    from osgeo import ogr
except ImportError:
    try:
        import ogr
    except ImportError:
        print("GDAL is not installed - OGR functionality not available")

from shapely.wkt import loads
import geopandas as gpd
from boltons.iterutils import pairwise
import geopy.distance as distance


def haversine(lon1, lat1, lon2, lat2, unit_m=True):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    default unit : m

    Parameters
    ----------
    lon1 : float
        longitude of the first point
    lat1 : float
        latitude of the first point
    lon2 : float
        longitude of the second point
    lat2 : float
        latitude of the second point
    unit_m : bool
        if True, return the distance in meters (default) if False,
        return the distance in kilometers

    Returns
    -------
    float
        distance between the two points

    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    if unit_m:
        r *= 1000
    return c * r


def download_osm(
    left,
    bottom,
    right,
    top,
    proxy=False,
    proxyHost="10.0.4.2",
    proxyPort="3128",
    cache=False,
    cacheTempDir="/tmp/tmpOSM/",
    verbose=True,
):
    """
    Downloads OpenStreetMap data for a given bounding box.

    Parameters
    ----------
    left : float
        The left longitude of the bounding box.
    bottom : float
        The bottom latitude of the bounding box.
    right : float
        The right longitude of the bounding box.
    top : float
        The top latitude of the bounding box.
    proxy : bool, optional
        Whether to use a proxy for the request. Defaults to False.
    proxyHost : str, optional
        The proxy host address. Defaults to "10.0.4.2".
    proxyPort : str, optional
        The proxy port number. Defaults to "3128".
    cache : bool, optional
        Whether to cache the downloaded tile. Defaults to False.
    cacheTempDir : str, optional
        The directory to store the cached tile. Defaults to "/tmp/tmpOSM/".
    verbose : bool, optional
        Whether to print progress messages. Defaults to True.

    Returns
    -------
    file-like object
        The downloaded OpenStreetMap tile.

    """
    import urllib.request  # To request the web

    if cache:
        ## cached tile filename
        cachedTileFilename = "osm_map_{:.8f}_{:.8f}_{:.8f}_{:.8f}.map".format(
            left, bottom, right, top
        )

        if verbose:
            print("Cached tile filename :", cachedTileFilename)

        Path(cacheTempDir).mkdir(
            parents=True, exist_ok=True
        )  ## Create cache path if not exists

        osmFile = Path(
            cacheTempDir + cachedTileFilename
        ).resolve()  ## Replace the relative cache folder path to absolute path

        if osmFile.is_file():
            # download from the cache folder
            if verbose:
                print("Tile loaded from the cache folder.")

            fp = urllib.request.urlopen("file://" + str(osmFile))
            return fp

    if proxy:
        # configure the urllib request with the proxy
        proxy_handler = urllib.request.ProxyHandler(
            {
                "https": "https://" + proxyHost + ":" + proxyPort,
                "http": "http://" + proxyHost + ":" + proxyPort,
            }
        )
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)

    request = "http://api.openstreetmap.org/api/0.6/map?bbox=%f,%f,%f,%f" % (
        left,
        bottom,
        right,
        top,
    )

    if verbose:
        print("Download the tile from osm web api ... in progress")
        print("Request :", request)

    fp = urllib.request.urlopen(request)

    if verbose:
        print("OSM Tile downloaded")

    if cache:
        if verbose:
            print("Write osm tile in the cache")
        content = fp.read()
        with open(osmFile, "wb") as f:
            f.write(content)

        if osmFile.is_file():
            if verbose:
                print("OSM tile written in the cache")

            fp = urllib.request.urlopen(
                "file://" + str(osmFile)
            )  ## Reload the osm tile from the cache (because fp.read moved the cursor)
            return fp

    return fp


def read_osm(filename_or_stream, only_roads=True):
    """
    Read graph in OSM format from file specified by name or by stream object.

    Parameters
    ----------
    filename_or_stream : string or file
        The filename or stream to read. File can be either a filename
        or stream/file object.
    only_roads : bool, optional
        Whether to only read roads. Defaults to True.

    Returns
    -------
    networkx multidigraph
        The graph from the OSM file.

    Examples
    --------
    >>> G=nx.read_osm(nx.download_osm(-122.33,47.60,-122.31,47.61))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([G.node[n]['lat']for n in G], [G.node[n]['lon'] for n in G], 'o', color='k')
    >>> plt.show()

    """
    osm = OSM(filename_or_stream)
    G = networkx.DiGraph()

    ## Add ways
    for w in osm.ways.values():
        if only_roads and "highway" not in w.tags:
            continue

        if "oneway" in w.tags:
            if w.tags["oneway"] == "yes":
                # ONLY ONE DIRECTION
                G.add_path(w.nds, id=w.id, highway=w.tags["highway"])
            else:
                # BOTH DIRECTION
                G.add_path(w.nds, id=w.id, highway=w.tags["highway"])
                G.add_path(w.nds[::-1], id=w.id, highway=w.tags["highway"])
        else:
            # BOTH DIRECTION
            G.add_path(w.nds, id=w.id, highway=w.tags["highway"])
            G.add_path(w.nds[::-1], id=w.id, highway=w.tags["highway"])

    ## Complete the used nodes' information
    for n_id in G.nodes.keys():
        n = osm.nodes[n_id]
        G.node[n_id]["lat"] = n.lat
        G.node[n_id]["lon"] = n.lon
        G.node[n_id]["id"] = n.id

    ## Estimate the length of each way
    for u, v in G.edges():
        distance = haversine(
            G.node[u]["lon"],
            G.node[u]["lat"],
            G.node[v]["lon"],
            G.node[v]["lat"],
            unit_m=True,
        )  # Give a realistic distance estimation (neither EPSG nor projection nor reference system are specified)

        G.add_weighted_edges_from([(u, v, distance)], weight="length")

    return G


class Node:
    """
    Represents a node in the OpenStreetMap data.

    Attributes
    ----------
    id : int
        The unique identifier of the node.
    lon : float
        The longitude coordinate of the node.
    lat : float
        The latitude coordinate of the node.
    tags : dict
        A dictionary containing additional tags associated with the
        node.

    """

    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.tags = {}

    def __str__(self):
        return "Node (id : %s) lon : %s, lat : %s " % (self.id, self.lon, self.lat)


class Way:
    """
    Represents a way in the OpenStreetMap data.

    Attributes
    ----------
    id : str
        The unique identifier of the way.
    osm : object
        The OpenStreetMap object that the way belongs to.
    nds : list
        The list of node references that make up the way.
    tags : dict
        The dictionary of tags associated with the way.

    """

    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}

    def split(self, dividers):
        """
        Splits the way into multiple smaller ways based on the given dividers.

        Parameters
        ----------
        dividers : dict
            A dictionary containing the number of occurrences of each
            node reference.

        Returns
        -------
        list
            A list of new Way objects, each representing a slice of the
            original way.

        """

        # slice the node-array using this nifty recursive function
        def slice_array(ar, dividers):
            for i in range(1, len(ar) - 1):
                if dividers[ar[i]] > 1:
                    left = ar[: i + 1]
                    right = ar[i:]

                    rightsliced = slice_array(right, dividers)

                    return [left] + rightsliced
            return [ar]

        slices = slice_array(self.nds, dividers)

        # create a way object for each node-array slice
        ret = []
        i = 0
        for slice in slices:
            littleway = copy.copy(self)
            littleway.id += "-%d" % i
            littleway.nds = slice
            ret.append(littleway)
            i += 1

        return ret


class OSM:
    """
    Represents an OpenStreetMap (OSM) data structure.

    Parameters
    ----------
    filename_or_stream : str or file object
        The OSM data file name or stream.

    Attributes
    ----------
    nodes : dict
        A dictionary of OSM nodes, where the key is the node ID and the
        value is the Node object.
    ways : dict
        A dictionary of OSM ways, where the key is the way ID and the
        value is the Way object.
    """

    def __init__(self, filename_or_stream):
        """
        Initializes an instance of the OSM class.

        Parameters
        ----------
        filename_or_stream : str or file object
            The OSM data file name or stream.

        """
        nodes = {}
        ways = {}

        superself = self

        class OSMHandler(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self, loc):
                pass

            @classmethod
            def startDocument(self):
                pass

            @classmethod
            def endDocument(self):
                pass

            @classmethod
            def startElement(self, name, attrs):
                if name == "node":
                    self.currElem = Node(
                        attrs["id"], float(attrs["lon"]), float(attrs["lat"])
                    )
                elif name == "way":
                    self.currElem = Way(attrs["id"], superself)
                elif name == "tag":
                    self.currElem.tags[attrs["k"]] = attrs["v"]
                elif name == "nd":
                    self.currElem.nds.append(attrs["ref"])

            @classmethod
            def endElement(self, name):
                if name == "node":
                    nodes[self.currElem.id] = self.currElem
                elif name == "way":
                    ways[self.currElem.id] = self.currElem

            @classmethod
            def characters(self, chars):
                pass

        xml.sax.parse(filename_or_stream, OSMHandler)

        self.nodes = nodes
        self.ways = ways

        # count times each node is used
        node_histogram = dict.fromkeys(self.nodes.keys(), 0)
        for way in self.ways.values():
            if (
                len(way.nds) < 2
            ):  # if a way has only one node, delete it out of the osm collection
                del self.ways[way.id]
            else:
                for node in way.nds:
                    node_histogram[node] += 1

        # use that histogram to split all ways, replacing the member set of ways
        new_ways = {}
        for id, way in self.ways.items():
            split_ways = way.split(node_histogram)
            for split_way in split_ways:
                new_ways[split_way.id] = split_way
        self.ways = new_ways


def fetch_roads_OSM(
    osm_path,
    acceptedRoads=[
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    ],
):
    """
    Returns a GeoDataFrame of OSM roads from an OSM file

    Parameters
    ----------
    osm_path : str
        path to OSM file
    acceptedRoads : list
        list of OSM road types

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame of OSM roads

    """
    driver = ogr.GetDriverByName("OSM")
    data = driver.Open(osm_path)

    sql_lyr = data.ExecuteSQL(
        "SELECT osm_id,highway FROM lines WHERE highway IS NOT NULL"
    )

    roads = []
    for feature in sql_lyr:
        if feature.GetField("highway") is not None:
            osm_id = feature.GetField("osm_id")
            shapely_geo = loads(feature.geometry().ExportToWkt())
            if shapely_geo is None:
                continue
            highway = feature.GetField("highway")
            if acceptedRoads != []:
                if highway in acceptedRoads:
                    roads.append([osm_id, highway, shapely_geo])
            else:
                roads.append([osm_id, highway, shapely_geo])
    if len(roads) > 0:
        road_gdf = gpd.GeoDataFrame(
            roads,
            columns=["osm_id", "infra_type", "geometry"],
            crs="epsg:4326",
        )
        return road_gdf
    else:
        print("No roads in {}".format(roads))


def line_length(line, ellipsoid="WGS-84"):
    """
    Returns length of a line in kilometers, given in geographic
    coordinates. Adapted from
    https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Parameters
    ----------
    line : shapely.geometry.LineString
        A shapely LineString object with WGS-84 coordinates
    ellipsoid : str
        string name of an ellipsoid that `geopy` understands (see
        http://geopy.readthedocs.io/en/latest/#module-geopy.distance)

    Returns
    -------
    float
        Length of line in kilometers

    """
    if line.geometryType() == "MultiLineString":
        return sum(line_length(segment) for segment in line)

    return sum(
        distance.geodesic(
            tuple(reversed(a)), tuple(reversed(b)), ellipsoid=ellipsoid
        ).km
        for a, b in pairwise(line.coords)
    )

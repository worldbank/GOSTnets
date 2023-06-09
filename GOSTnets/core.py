import os, sys, logging, warnings, time

import pyproj
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np

from scipy import spatial
from functools import partial
from shapely.wkt import loads, dumps
from shapely.geometry import Point, LineString, MultiLineString, MultiPoint, box
from shapely.ops import linemerge, unary_union, transform
from collections import Counter

import math

def combo_csv_to_graph(fpath, u_tag = 'u', v_tag = 'v', geometry_tag = 'Wkt', largest_G = False):
    """
    Function for generating a G object from a saved combo .csv

    :param fpath: path to a .csv containing edges (WARNING: COMBO CSV only)
    :param u_tag: specify column containing u node ID if not labelled 'u'
    :param v_tag: specify column containing u node ID if not labelled 'v'
    :param geometry_tag: specify column containing u node ID if not
    :returns: a multidigraph object
    """

    edges_1 = pd.read_csv(os.path.join(fpath))

    edges = edges_1.copy()

    node_bunch = list(set(list(edges[u_tag]) + list(edges[v_tag])))

    node_bunch2 = []

    for node in node_bunch:
        #print(type(node))
        if isinstance(node, int):
            node_bunch2.append(node)
        elif node.isnumeric():
            node_bunch2.append(int(node))
        else:
            node_bunch2.append(node)

    col_list = list(edges.columns)
    drop_cols = [u_tag, v_tag, geometry_tag]
    attr_list = [col_entry for col_entry in col_list if col_entry not in drop_cols]

    def convert(x, attr_list):
        u = x[u_tag]
        v = x[v_tag]

        if isinstance(u, int):
            u = u
        elif u.isnumeric():
            u = int(u)
        else:
            u = u

        if isinstance(v, int):
            v = v
        elif v.isnumeric():
            v = int(v)
        else:
            v = v

        data = {'Wkt':loads(x[geometry_tag])}
        for i in attr_list:
            data[i] = x[i]

        return (u, v, data)

    edge_bunch = edges.apply(lambda x: convert(x, attr_list), axis = 1).tolist()

    G = nx.MultiDiGraph()

    G.add_nodes_from(node_bunch2)
    G.add_edges_from(edge_bunch)

    # print("print node bunch")
    # print(G.nodes)

    # for u, data in G.nodes(data = True):
    #     q = tuple(float(x) for x in u[1:-1].split(','))
    #     #q = tuple(x for x in u[1:-1].split(','))
    #     data['x'] = q[0]
    #     data['y'] = q[1]

    #G = nx.convert_node_labels_to_integers(G)


    if largest_G == True:
        list_of_subgraphs = list(nx.strongly_connected_component_subgraphs(G))
        l = 0
        cur_max = 0
        for i in list_of_subgraphs:
            if i.number_of_edges() > cur_max:
                cur_max = i.number_of_edges()
                max_ID = l
            l +=1
        G = list_of_subgraphs[max_ID]

    return G

def edges_and_nodes_gdf_to_graph(nodes_df, edges_df, node_tag = 'node_ID', u_tag = 'stnode', v_tag = 'endnode', geometry_tag = 'Wkt', largest_G = False, discard_node_col=[], checks=False, add_missing_reflected_edges=False, oneway_tag=None):

    """
    Function for generating a G object from a saved .csv of edges

    :param fpath_nodes: 
      path to a .csv containing nodes
    :param fpath_edges: 
      path to a .csv containing edges
    :param u_tag: 
      optional. specify column containing the node ID. This is used to only include entries that have a value.
    :param u_tag: 
      optional. specify column containing u node ID if not labelled 'stnode'
    :param v_tag: 
      specify column containing v node ID if not labelled 'endnode'
    :param geometry_tag: 
      specify column containing geometry if not labelled 'Wkt'
    :param largest_G: 
      If largest_G is true, then only the largest graph will be returned
    :param discard_node_col:
      default is empty, all columns in the nodes_df will be copied to the nodes in the graph. If a list is filled, all the columns specified will be dropped.
    :checks:
      if True, will perfrom a validation checks and return the nodes_df with a 'node_in_edge_df' column
    :add_missing_reflected_edges:
      if contains a tag, then the oneway column is used to see whether reverse edges need to be added. This is much faster than using the add_missing_reflected_edges after a graph is already created.
    :oneway_tag:
      if oneway_tag exists, then missing reflected edges won't be added where an edge's oneway_tag equals True
    :returns: 
      a multidigraph object
    """

    if checks == True:

        # chck_set = list(set(list(edges_df[u_tag]) + list(edges_df[v_tag])))
        # same thing, but easier to understand?
        chck_set = list(edges_df[u_tag])
        chck_set.extend(list(edges_df[v_tag]))
        chck_set = list(set(chck_set))

        def check(x, chck_set):
            if x in chck_set:
                return 1
            else:
                return 0

        nodes_df['node_in_edge_df'] = nodes_df[node_tag].apply(lambda x: check(x, chck_set))
        
        unique, counts = np.unique(nodes_df['node_in_edge_df'].values, return_counts=True)
       
        print("validation check")
        print(f"counts of nodes in edges_df or not: {dict(zip(unique, counts))}")

        # This can be improved by doing another test in reverse: nodes found in edge_df that are within the nodes_df or not

        return nodes_df['node_in_edge_df']

    #nodes_df = nodes_df.drop(columns=['node_in_edge_df'])

    # creating a node bunch isn't needed
    # def convert_nodes(x):
    #     u = x[node_tag]
    #     data = {'x':x.x,
    #            'y':x.y}
    #     return (u, data)
    # node_bunch = nodes_df.apply(lambda x: convert_nodes(x), axis = 1).tolist()

    col_list = list(edges_df.columns)
    drop_cols = [u_tag, v_tag, geometry_tag]
    attr_list = [col_entry for col_entry in col_list if col_entry not in drop_cols]

    edge_bunch_reverse_edges  = []

    def convert_edges(x):
        u = x[u_tag]
        v = x[v_tag]

        if isinstance(u, int):
            u = u
        # elif u.isnumeric():
        #     u = int(u)
        else:
            u = u

        if isinstance(v, int):
            v = v
        # elif v.isnumeric():
        #     v = int(v)
        else:
            v = v

        data = {geometry_tag:loads(str(x[geometry_tag]))}
        for i in attr_list:
            data[i] = x[i]
        
        if add_missing_reflected_edges:
            if oneway_tag:
                if x[oneway_tag] == False:
                    edge_bunch_reverse_edges.append((v, u, data))
            else:
                edge_bunch_reverse_edges.append((v, u, data))

        return (u, v, data)

    # This will create edges and nodes
    edge_bunch = edges_df.apply(lambda x: convert_edges(x), axis = 1).tolist()

    G = nx.MultiDiGraph()

    #G.add_nodes_from(node_bunch)
    # just needs edges to build graph with edges and nodes
    G.add_edges_from(edge_bunch)

    if len(edge_bunch_reverse_edges) > 0:
        G.add_edges_from(edge_bunch_reverse_edges)

    # discard columns if specified
    if len(discard_node_col) > 0:
        nodes_df = nodes_df.drop(columns=discard_node_col)
    
    # consider dropping na values
    # nodes_df.dropna(axis='columns', inplace=True)

    # add nodes' attributes to graph using nodes_df
    # This way works, as of Networkx 2.0
    # https://stackoverflow.com/questions/54497929/networkx-setting-node-attributes-from-dataframe
    node_attr = nodes_df.set_index(node_tag)

    #perform checks if x and y columns are fully populated, if not then copy them from the geometry column
    if node_attr.x.isnull().values.any() or node_attr.x.isnull().values.any():
        node_attr['x'] = node_attr.geometry.x
        node_attr['y'] = node_attr.geometry.y

    node_attr_dict = node_attr.to_dict('index')

    #https://stackoverflow.com/questions/9442724/how-can-i-use-if-else-in-a-dictionary-comprehension

    #node_attr = {(int(item[0]) if item[0].isnumeric() else item[0]):item[1] for item in node_attr.items() }

    def selector(x):
        if isinstance(x, int):
            return x
        # elif x.isnumeric():
        #     return int(x)
        else:
            return x


    node_attr_dict = { selector(item[0]):item[1] for item in node_attr_dict.items()}


    nx.set_node_attributes(G, node_attr_dict)

    # we want to keep the original node labels
    #G = nx.convert_node_labels_to_integers(G)

    if largest_G == True:
        list_of_subgraphs = list(nx.strongly_connected_component_subgraphs(G))
        l = 0
        cur_max = 0
        for i in list_of_subgraphs:
            if i.number_of_edges() > cur_max:
                cur_max = i.number_of_edges()
                max_ID = l
            l +=1
        G = list_of_subgraphs[max_ID]

    return G

def edges_and_nodes_csv_to_graph(fpath_nodes, fpath_edges, u_tag = 'stnode', v_tag = 'endnode', geometry_tag = 'Wkt', largest_G = False):

    """
    Function for generating a G object from a saved .csv of edges

    :param fpath_nodes: 
      path to a .csv containing nodes
    :param fpath_edges: 
      path to a .csv containing edges
    :param u_tag: 
      optional. specify column containing u node ID if not labelled 'stnode'
    :param v_tag: 
      specify column containing v node ID if not labelled 'endnode'
    :param geometry_tag: 
      specify column containing geometry if not labelled 'Wkt'
    :returns: 
      a multidigraph object
    """

    nodes_df = pd.read_csv(fpath_nodes)
    edges_df = pd.read_csv(fpath_edges)

    G = edges_and_nodes_gdf_to_graph(nodes_df, edges_df, u_tag = u_tag, v_tag = v_tag, geometry_tag = geometry_tag, largest_G = largest_G)

    return G

def node_gdf_from_graph(G, crs = 'epsg:4326', attr_list = None, geometry_tag = 'geometry', xCol='x', yCol='y'):
    """
    Function for generating GeoDataFrame from Graph

    :param G: a graph object G
    :param crs: projection of format {'init' :'epsg:4326'}. Defaults to WGS84. note: here we are defining the crs of the input geometry - we do NOT reproject to this crs. To reproject, consider using geopandas' to_crs method on the returned gdf.
    :param attr_list: list of the keys which you want to be moved over to the GeoDataFrame, if not all. Defaults to None, which will move all.
    :param geometry_tag: specify geometry attribute of graph, default 'geometry'
    :param xCol: if no shapely geometry but Longitude present, assign here
    :param yCol: if no shapely geometry but Latitude present, assign here
    :returns: a geodataframe of the node objects in the graph
    """

    nodes = []
    keys = []

    # finds all of the attributes
    if attr_list is None:
        for u, data in G.nodes(data = True):
            keys.append(list(data.keys()))
        flatten = lambda l: [item for sublist in l for item in sublist]
        attr_list = list(set(flatten(keys)))

    if geometry_tag in attr_list:
        non_geom_attr_list = attr_list
        non_geom_attr_list.remove(geometry_tag)
    else:
        non_geom_attr_list = attr_list

    if 'node_ID' in attr_list:
        non_geom_attr_list = attr_list
        non_geom_attr_list.remove('node_ID')

    z = 0

    for u, data in G.nodes(data = True):

        if geometry_tag not in attr_list and xCol in attr_list and yCol in attr_list :
            try:
                new_column_info = {
                'node_ID': u,
                'geometry': Point(data[xCol], data[yCol]),
                'x': data[xCol],
                'y': data[yCol]}
            except:
                print('Skipped due to missing geometry data:',(u, data))
        else:
            try:
                new_column_info = {
                'node_ID': u,
                'geometry': data[geometry_tag],
                'x':data[geometry_tag].x,
                'y':data[geometry_tag].y}
            except:
                print((u, data))

        for i in non_geom_attr_list:
            try:
                new_column_info[i] = data[i]
            except:
                pass

        nodes.append(new_column_info)
        z += 1

    nodes_df = pd.DataFrame(nodes)
    nodes_df = nodes_df[['node_ID', *non_geom_attr_list, geometry_tag]]
    nodes_df = nodes_df.drop_duplicates(subset = ['node_ID'], keep = 'first')
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry = nodes_df.geometry, crs = crs)

    return nodes_gdf

def edge_gdf_from_graph(G, crs = 'EPSG:4326', attr_list = None, geometry_tag = 'geometry', xCol='x', yCol = 'y', oneway_tag = 'oneway', single_edge = False):
    """
    Function for generating a GeoDataFrame from a networkx Graph object

    :param G: (required) a graph object G
    :param crs: (optional) projection of format {'init' :'epsg:4326'}. Defaults to WGS84. Note: here we are defining the crs of the input geometry -we do NOT reproject to this crs. To reproject, consider using geopandas' to_crs method on the returned gdf.
    :param attr_list: (optional) list of the keys which you want to be moved over to the GeoDataFrame.
    :param geometry_tag: (optional) the key in the data dictionary for each edge which contains the geometry info.
    :param xCol: (optional) if no geometry is present in the edge data dictionary, the function will try to construct a straight line between the start and end nodes, if geometry information is present in their data dictionaries.  Pass the Longitude info as 'xCol'.
    :param yCol: (optional) likewise, determining the Latitude tag for the node's data dictionary allows us to make a straight line geometry where an actual geometry is missing.
    :param single_edge: If True then one edge/row in the returned GeoDataFrame will represent a bi-directional edge. An extra 'oneway' column will be added
    :returns: a GeoDataFrame object of the edges in the graph
    """

    edges = []
    keys = []

    if attr_list is None:
        for u, v, data in G.edges(data = True):
            keys.append(list(data.keys()))
        flatten = lambda l: [item for sublist in l for item in sublist]
        keys = list(set(flatten(keys)))
        if geometry_tag in keys:
            keys.remove(geometry_tag)
        if 'geometry' in keys:
            keys.remove('geometry')
        attr_list = keys
        if single_edge == True:
            if oneway_tag not in keys:
                attr_list.append(oneway_tag)

    def add_edge_attributes(data, stnode=u, endnode=v):
        if geometry_tag in data:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            # geom = str(data[geometry_tag])
            geom = data[geometry_tag]

        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[stnode][xCol]
            y1 = G.nodes[stnode][yCol]
            #print(endnode)
            x2 = G.nodes[endnode][xCol]
            y2 = G.nodes[endnode][yCol]
            geom = LineString([(x1, y1), (x2, y2)])

        new_column_info = {
            'stnode':stnode,
            'endnode':endnode,
            geometry_tag:geom}

        for i in attr_list:
            try:
                new_column_info[i] = data[i]
            except:
                pass
        
        return new_column_info

    if single_edge == False:

        for u, v, data in G.edges(data=True):

            new_column_info = add_edge_attributes(data, stnode=u, endnode=v)

            edges.append(new_column_info)
    
    else:

        unique_edges = []

        for u, v, data in G.edges(data=True):

            if G.has_edge(u,v) and G.has_edge(v,u):
                # two-way
                if (u, v) not in unique_edges and (v, u) not in unique_edges:

                    unique_edges.append((u,v))

                    new_column_info = add_edge_attributes(data, stnode=u, endnode=v)

                    new_column_info[oneway_tag] = False
                    edges.append(new_column_info)
            else:
                # one-way
                new_column_info = add_edge_attributes(data, stnode=u, endnode=v)

                new_column_info[oneway_tag] = True
                edges.append(new_column_info)

    edges_df = pd.DataFrame(edges)

    # make sure the all attributes are in edges_df, or else it may break for example if the 'tunnel' key appeared in the attr_list but does not appear in the edges_df
    for attr in attr_list:
        if attr not in edges_df.keys():
            attr_list.remove(attr)

    edges_df = edges_df[['stnode','endnode',*attr_list,geometry_tag]]
    if type(edges_df.iloc[0][geometry_tag]) == str:
        edges_df[geometry_tag] = edges_df[geometry_tag].apply(str)
        edges_df[geometry_tag] = edges_df[geometry_tag].apply(loads)
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry = geometry_tag, crs = crs)

    return edges_gdf

def graph_nodes_intersecting_polygon(G, polygons, crs = None):

    """
    Function for generating GeoDataFrame from Graph. Note: ensure any GeoDataFrames are in the same projection before using function, or pass a crs

    :param G: a Graph object OR node geodataframe
    :param crs: a crs object of form {'init':'epsg:XXXX'}. If passed, matches both inputs to this crs.
    :returns: a list of the nodes intersecting the polygons
    """

    if type(G) == nx.classes.multidigraph.MultiDiGraph:
        graph_gdf = node_gdf_from_graph(G)

    elif type(G) == gpd.geodataframe.GeoDataFrame:
        graph_gdf = G

    else:
        raise ValueError('Expecting a graph or node geodataframe for G!')

    if type(polygons) != gpd.geodataframe.GeoDataFrame:
        raise ValueError('Expecting a geodataframe for polygon(s)!')

    if crs != None and graph_gdf.crs != crs:
            graph_gdf = graph_gdf.to_crs(crs)

    if crs != None and polygons.crs != crs:
            polygons = polygons.to_crs(crs)

    if polygons.crs != graph_gdf.crs:
        raise ValueError('crs mismatch detected! aborting process')

    intersecting_nodes = []
    for poly in polygons.geometry:

        def chck(x, poly):
            if poly.contains(x):
                return 1
            else:
                return 0

        graph_gdf['intersecting'] = graph_gdf['geometry'].apply(lambda x: chck(x, poly))
        intersecting_nodes.append(list(graph_gdf['node_ID'].loc[graph_gdf['intersecting'] == 1]))

    intersecting_nodes = [j for i in intersecting_nodes for j in i]
    unique_intersecting_nodes = list(set(intersecting_nodes))
    return unique_intersecting_nodes

def graph_edges_intersecting_polygon(G, polygons, mode, crs = None, fast = True):
    """
    Function for identifying edges of a graph that intersect polygon(s). Ensure any GeoDataFrames are in the same projection before using function, or pass a crs.

    :param G: a Graph object
    :param polygons: a GeoDataFrame containing one or more polygons
    :param mode: a string, either 'contains' or 'intersecting'
    :param crs: If passed, will reproject both polygons and graph edge gdf to this projection.
    :param fast: (default: True): we can cheaply test whether an edge intersects a polygon gdf by checking whether either the start or end nodes are within a polygon. If both are, then we return 'contained'; if at least one is, we can return 'intersects'. If we set fast to False, then we iterate through each geometry one at a time, and check to see whether the geometry object literally intersects the polygon geodataframe, one at a time. May be computationally intensive!
    :returns: a list of the edges intersecting the polygons
    """

    if type(G) == nx.classes.multidigraph.MultiDiGraph:
        node_graph_gdf = node_gdf_from_graph(G)
        edge_graph_gdf = edge_gdf_from_graph(G)
    else:
        raise ValueError('Expecting a graph for G!')

    if type(polygons) != gpd.geodataframe.GeoDataFrame:
        raise ValueError('Expecting a geodataframe for polygon(s)!')

    if crs != None and node_graph_gdf.crs != crs:
            node_graph_gdf = node_graph_gdf.to_crs(crs)

    if crs != None and polygons.crs != crs:
            polygons = polygons.to_crs(crs)

    if polygons.crs != node_graph_gdf.crs:
        raise ValueError('crs mismatch detected! aborting process')

    intersecting_nodes = graph_nodes_intersecting_polygon(node_graph_gdf, polygons, crs)

    if fast == True:

        if mode == 'contains':
            edge_graph_gdf = edge_graph_gdf.loc[(edge_graph_gdf.stnode.isin(intersecting_nodes)) &
                                     (edge_graph_gdf.endnode.isin(intersecting_nodes))]
        elif mode == 'intersects':
            edge_graph_gdf = edge_graph_gdf.loc[(edge_graph_gdf.stnode.isin(intersecting_nodes)) |
                                     (edge_graph_gdf.endnode.isin(intersecting_nodes))]
    elif fast == False:
        poly = unary_union(polygons.geometry)

        if mode == 'contains':
            edge_graph_gdf = edge_graph_gdf.loc[(edge_graph_gdf.within(poly))]

        elif mode == 'intersects':
            edge_graph_gdf = edge_graph_gdf.loc[(edge_graph_gdf.intersects(poly))]

    else:
        raise ValueError("'fast' requires a boolean input!!")

    return edge_graph_gdf

def sample_raster(G, tif_path, property_name = 'RasterValue'):
    """
    Function for attaching raster values to corresponding graph nodes. Ensure any GeoDataFrames / graphs are in the same projection before using function, or pass a crs

    :param G: a graph containing one or more nodes
    :param tif_path: a raster or path to a tif
    :param property_name: a property name for the value of the raster attached to the node
    :returns: a graph
    """

    import rasterio

    if type(G) == nx.classes.multidigraph.MultiDiGraph or type(G) == nx.classes.digraph.DiGraph:
        pass
    else:
        raise ValueError('Expecting a graph or geodataframe for G!')

    # generate dictionary of {node ID: point} pairs
    try:
        list_of_nodes = {}
        for u, data in G.nodes(data=True):
            list_of_nodes.update({u:(data['x'], data['y'])})
    except:
        raise ValueError('loading point geometry went wrong. Ensure node data dict includes x, y values!')

    # load raster
    try:
        dataset = rasterio.open(os.path.join(tif_path))
    except:
        raise ValueError('Expecting a path to a .tif file!')

    # create list of values, throw out nodes that don't intersect the bounds of the raster
    b = dataset.bounds
    datasetBoundary = box(b[0], b[1], b[2], b[3])
    selKeys = []
    selPts = []
    for key, pt in list_of_nodes.items():
        if Point(pt[0], pt[1]).intersects(datasetBoundary):
            selPts.append(pt)
            selKeys.append(key)
    raster_values = list(dataset.sample(selPts))
    raster_values = [x[0] for x in raster_values]

    # generate new dictionary of {node ID: raster values}
    ref = dict(zip(selKeys, raster_values))

    # load new values onto node data dictionary
    missedCnt = 0
    for u, data in G.nodes(data=True):
        try:
            data[property_name] = ref[u]
        except:
            missedCnt += 1
            logging.info("Could not add raster value to node %s" % u)
    logging.info("Number of original nodes: %s" % len(G.nodes))
    logging.info("Number of missed nodes in raster: %d" % missedCnt)
    logging.info("Number of nodes that intersected raster: %d" % len(selKeys))

    return G

def generate_isochrones(G, origins, thresh, weight = None, stacking = False):
    """
    Function for generating isochrones from one or more graph nodes. Ensure any GeoDataFrames / graphs are in the same projection before using function, or pass a crs

    :param G: a graph containing one or more nodes
    :param orgins: a list of node IDs that the isochrones are to be generated from
    :param thresh: The time threshold for the calculation of the isochrone
    :param weight: Name of edge weighting for calculating 'distances'. For isochrones, should be time expressed in seconds. Defaults to time expressed in seconds.
    :param stacking: If True, returns number of origins that can be reached from that node. If false, max = 1
    :returns: The original graph with a new data property for the nodes and edges included in the isochrone
    """

    if type(origins) == list and len(origins) >= 1:
        pass
    else:
        raise ValueError('Ensure isochrone centers (origins object) is a list containing at least one node ID!')

    ddict = list(G.nodes(data = True))[:1][0][1]

    if weight == None:
        if 'time' not in ddict.keys():
            raise ValueError('need "time" key in edge value dictionary!')
        else:
            weight = 'time'

    sub_graphs = []
    for node in origins:
        sub_graphs.append(nx.ego_graph(G, node, thresh, distance = weight))

    reachable_nodes = []
    for graph in sub_graphs:
        reachable_nodes.append(list(graph.nodes))

    reachable_nodes = [j for i in reachable_nodes for j in i]

    if stacking == False:

        reachable_nodes = set(reachable_nodes)

        for u, data in G.nodes(data=True):
            if u in reachable_nodes:
                data[thresh] = 1
            else:
                data[thresh] = 0

    elif stacking == True:

        reachable_nodes = Counter(reachable_nodes)

        for u, data in G.nodes(data=True):
            if u in reachable_nodes:
                data[thresh] = reachable_nodes[u]
            else:
                data[thresh] = 0
    else:
        raise ValueError('stacking must either be True or False!')

    return G


def make_iso_polys(G, origins, trip_times, edge_buff=10, node_buff=25, infill=False, weight = 'time', measure_crs = 'epsg:4326', edge_filters=None):
    """
    Function for adding a time value to edge dictionaries

    :param G: a graph object
    :param origins: a list object of node IDs from which to generate an isochrone poly object
    :param trip_times: a list object containing the isochrone values
    :param edge_buff: the thickness with witch to buffer included edges
    :param node_buff: the thickness with witch to buffer included nodes
    :param infill: If True, will remove any holes in isochrones
    :param weight: The edge weight to use when appraising travel times.
    :param measure_crs: measurement crs, object of form {'init':'epsg:XXXX'}
    :edge_filters: you can optionally add a dictionary with key values, where the key is the attribute and the value you want to ignore from creating isochrones. An example might be an underground subway line.
    """

    default_crs = 'epsg:4326'

    if type(origins) == list and len(origins) >= 1:
        pass
    else:
        raise ValueError('Ensure isochrone centers ("origins" object) is a list containing at least one node ID!')

    isochrone_polys, nodez, tt = [], [], []

    for trip_time in sorted(trip_times, reverse=True):
        count = 0
        for _node_ in origins:
            subgraph = nx.ego_graph(G, _node_, radius = trip_time, distance = weight)
            #subgraph = nx.ego_graph(G_service0002, _node_, radius = 3600, distance = 'length')
            node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]

            if len(node_points) > 1:
                count += 1
                if count == 1:
                    # create initial GDFs

                    nodes_gdf = gpd.GeoDataFrame({'id': subgraph.nodes()}, geometry=node_points, crs = default_crs)
                    nodes_gdf = nodes_gdf.set_index('id')
                    nodes_gdf['coords'] = nodes_gdf['geometry'].map(lambda x: x.coords[0])

                    edge_gdf = edge_gdf_from_graph(subgraph)
                    if edge_filters:
                        for edge_filter in edge_filters.items():
                            edge_gdf = edge_gdf.loc[edge_gdf[edge_filter[0]] != edge_filter[1]]
                    edge_gdf = edge_gdf[['geometry']]
                    edge_gdf['coords'] = edge_gdf.geometry.apply(lambda geometry: str(geometry.coords[0])+','+str(geometry.coords[1]))
    
                else:
                    
                    new_nodes_gdf = gpd.GeoDataFrame({'id': subgraph.nodes()}, geometry=node_points, crs = default_crs)
                    new_nodes_gdf = new_nodes_gdf.set_index('id')
                    new_nodes_gdf['coords'] = new_nodes_gdf['geometry'].map(lambda x: x.coords[0])
                    
                    new_edge_gdf = edge_gdf_from_graph(subgraph)
                    if edge_filters:
                        for edge_filter in edge_filters.items():
                            new_edge_gdf = new_edge_gdf.loc[new_edge_gdf[edge_filter[0]] != edge_filter[1]]
                    new_edge_gdf = new_edge_gdf[['geometry']]
                    new_edge_gdf['coords'] = new_edge_gdf.geometry.apply(lambda geometry: str(geometry.coords[0])+','+str(geometry.coords[1]))
                    
                    # discard pp that have the same coordinate of an existing node
                    nodes_gdf = pd.concat([nodes_gdf,new_nodes_gdf], ignore_index = True)
                    edge_gdf = pd.concat([edge_gdf,new_edge_gdf], ignore_index = True)
                    
            else:
                pass

        print("merge all edges and nodes")

        #drop duplicates
        nodes_gdf.drop_duplicates(inplace=True, subset="coords")
        edge_gdf.drop_duplicates(inplace=True, subset="coords") 

        if measure_crs != None and nodes_gdf.crs != measure_crs:
            nodes_gdf = nodes_gdf.to_crs(measure_crs)
            edge_gdf = edge_gdf.to_crs(measure_crs)

        n = nodes_gdf.buffer(node_buff).geometry
        e = edge_gdf.buffer(edge_buff).geometry

        all_gs = list(n) + list(e)

        print("unary_union")
        new_iso = gpd.GeoSeries(all_gs).unary_union

        # If desired, try and "fill in" surrounded
        # areas so that shapes will appear solid and blocks
        # won't have white space inside of them

        if infill:
            new_iso = Polygon(new_iso.exterior)

        isochrone_polys.append(new_iso)
        nodez.append(str(_node_))
        tt.append(trip_time)
            
        
    gdf = gpd.GeoDataFrame({'geometry':isochrone_polys,'thresh':tt,'nodez':_node_}, crs = measure_crs, geometry = 'geometry')
    gdf = gdf.to_crs(default_crs)

    return gdf

# probably will depreciate soon
def make_iso_polys_original(G, origins, trip_times, edge_buff=10, node_buff=25, infill=False, weight = 'time', measure_crs = 'epsg:4326'):
    """
    Function for adding a time value to edge dictionaries

    :param G: a graph object
    :param origins: a list object of node IDs from which to generate an isochrone poly object
    :param trip_times: a list object containing the isochrone values
    :param edge_buff: the thickness with witch to buffer included edges
    :param node_buff: the thickness with witch to buffer included nodes
    :param infill: If True, will remove any holes in isochrones
    :param weight: The edge weight to use when appraising travel times.
    :param measure_crs: measurement crs, object of form {'init':'epsg:XXXX'}
    """

    default_crs = 'epsg:4326'

    if type(origins) == list and len(origins) >= 1:
        pass
    else:
        raise ValueError('Ensure isochrone centers ("origins" object) is a list containing at least one node ID!')

    isochrone_polys, tt, nodez = [], [], []

    for trip_time in sorted(trip_times, reverse=True):

        for _node_ in origins:

            #print(f"print _node_: {_node_}")

            subgraph = nx.ego_graph(G, _node_, radius = trip_time, distance = weight)
            node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]

            if len(node_points) >= 1:

                nodes_gdf = gpd.GeoDataFrame({'id': subgraph.nodes()}, geometry=node_points, crs = default_crs)
                nodes_gdf = nodes_gdf.set_index('id')

                edge_lines = []

                for n_fr, n_to in subgraph.edges():
                    f = nodes_gdf.loc[n_fr].geometry
                    t = nodes_gdf.loc[n_to].geometry
                    edge_lines.append(LineString([f,t]))

                edge_gdf = gpd.GeoDataFrame({'geoms':edge_lines}, geometry = 'geoms', crs = default_crs)

                if measure_crs != None and nodes_gdf.crs != measure_crs:
                    nodes_gdf = nodes_gdf.to_crs(measure_crs)
                    edge_gdf = edge_gdf.to_crs(measure_crs)

                n = nodes_gdf.buffer(node_buff).geometry
                e = edge_gdf.buffer(edge_buff).geometry

                all_gs = list(n) + list(e)

                new_iso = gpd.GeoSeries(all_gs).unary_union

                # If desired, try and "fill in" surrounded
                # areas so that shapes will appear solid and blocks
                # won't have white space inside of them

                if infill:
                    new_iso = Polygon(new_iso.exterior)

                isochrone_polys.append(new_iso)
                nodez.append(str(_node_))
                tt.append(trip_time)
            else:
                pass

    gdf = gpd.GeoDataFrame({'geometry':isochrone_polys,'thresh':tt,'nodez':nodez}, crs = measure_crs, geometry = 'geometry')
    gdf = gdf.to_crs(default_crs)

    return gdf

def find_hwy_distances_by_class(G, distance_tag='length'):
    """
    Function for finding out the different highway classes in the graph and their respective lengths

    :param G: a graph object
    :param distance_tag: specifies which edge attribute represents length
    :returns: a dictionary that has each class and the total distance per class
    """

    if type(G) == nx.classes.multidigraph.MultiDiGraph or type(G) == nx.classes.digraph.DiGraph:
        pass
    else:
        raise ValueError('Expecting a graph or geodataframe for G!')

    G_adj = G.copy()

    class_list = []

    for u, v, data in G_adj.edges(data=True):
        #print(data['highway'])
        if type(data['highway']) == list:
                if data['highway'][0] not in class_list:
                    class_list.append(data['highway'][0])
        else:
            if data['highway'] not in class_list:
                class_list.append(data['highway'])
    
    class_dict = { i : 0 for i in class_list }

    for i in class_list:
        for u, v, data in G_adj.edges(data=True):
            if type(data['highway']) == list:
                if data['highway'][0] == i:
                    class_dict[i] += data[distance_tag]
            else:
                if data['highway'] == i:
                    class_dict[i] += data[distance_tag]

    return class_dict

def find_graph_avg_speed(G, distance_tag, time_tag):
    """
    Function for finding the average speed per km for the graph. It will sum up the total meters in the graph and the total time (in sec). \
    Then it will convert m/sec to km/hr. This function needs the 'convert_network_to_time' function to have run previously.
    
    :param G:
      a graph containing one or more nodes
    :param distance_tag:
      the key in the dictionary for the field currently containing a distance in meters
    :param time_tag:
      time to traverse the edge in seconds
    :returns:
      The average speed for the whole graph in km per hr
    """

    if type(G) == nx.classes.multidigraph.MultiDiGraph or type(G) == nx.classes.digraph.DiGraph:
        pass
    else:
        raise ValueError('Expecting a graph or geodataframe for G!')

    G_adj = G.copy()

    total_meters = 0
    total_sec = 0

    for u, v, data in G_adj.edges(data=True):

        total_meters = total_meters + data[distance_tag]
        total_sec = total_sec + data[time_tag]

    # perform conversion
    # ex. 5m/1sec = .005/.00027 = 18.51 kph
    avg_speed_kmph = (total_meters/1000)/(total_sec/3600)

    return avg_speed_kmph

def example_edge(G, n=1):
    """
    Prints out an example edge

    :param G: a graph object
    :param n: n - number of edges to print
    """
    i = list(G.edges(data = True))[:n]
    for j in i:
        print(j)

def example_node(G, n=1):
    """
    Prints out an example node

    :param G: a graph object
    :param n: number of nodes to print
    """

    i = list(G.nodes(data = True))[:n]
    for j in i:
        print(j)

def convert_network_to_time(G, distance_tag, graph_type = 'drive', road_col = 'highway', output_time_col = 'time', speed_dict = None, walk_speed = 4.5, factor = 1, default = 20):
    """
    Function for adding a time value to graph edges. Ensure any graphs are in the same projection before using function, or pass a crs.

    DEFAULT SPEEDS:

               speed_dict = {
               'residential': 20,  # kmph
               'primary': 40, # kmph
               'primary_link':35,
               'motorway':50,
               'motorway_link': 45,
               'trunk': 40,
               'trunk_link':35,
               'secondary': 30,
               'secondary_link':25,
               'tertiary':30,
               'tertiary_link': 25,
               'unclassified':20,
               'projected_footway':3.5
               }

    :param G: a graph containing one or more nodes
    :param distance_tag: the key in the dictionary for the field currently
               containing a distance in meters
    :param road_col: key for the road type in the edge data dictionary
    :param road_col: key for the time value in the output graph
    :param graph_type: set to either 'drive' or 'walk'. IF walk - will set time = walking time across all segments, using the supplied walk_speed. IF drive - will use a speed dictionary for each road type, or defaults as per the note below.
    :param speed_dict: speed dictionary to use. If not supplied, reverts to
               defaults
    :param walk_speed: specify a walkspeed in km/h
    :param factor: allows you to scale up / down distances if saved in a unit other than meters. Set to 1000 if length in km.
    :param default: if highway type not in the speed_dict, use this speed as the default. If default is None, then the conversion will be skipped
    :returns: The original graph with a new data property for the edges called 'time'
    """

    if type(G) == nx.classes.multidigraph.MultiDiGraph or type(G) == nx.classes.digraph.DiGraph:
        pass
    else:
        raise ValueError('Expecting a graph for G!')

    import warnings

    try:
        # checks the first edge to see if the 'time' attribute already exists
        if list(G.edges(data = True))[0][2]['time']:
          warnings.warn('Aree you sure you want to convert length to time? This graph already has a time attribute')
    except:
        pass

    G_adj = G.copy()

    for u, v, data in G_adj.edges(data=True):

        orig_len = data[distance_tag] * factor

        # Note that this is a MultiDiGraph so there could
        # be multiple indices here, I naively assume this is not
        # the case
        #actually saves the length in meters
        data['length'] = orig_len

        # get appropriate speed limit
        if graph_type == 'walk':
            speed = walk_speed

        elif graph_type == 'drive':

            if speed_dict == None:
                speed_dict = {
                'residential': 20,  # kmph
                'primary': 40, # kmph
                'primary_link':35,
                'motorway':50,
                'motorway_link': 45,
                'trunk': 40,
                'trunk_link':35,
                'secondary': 30,
                'secondary_link':25,
                'tertiary':30,
                'tertiary_link': 25,
                'unclassified':20,
                'projected_footway':3.5
                }

            highwayclass = data[road_col]

            if type(highwayclass) == list:
                highwayclass = highwayclass[0]

            if highwayclass in speed_dict.keys():
                speed = speed_dict[highwayclass]
            else:
                if default == None:
                    continue
                else:
                    speed = default

        else:
            raise ValueError('Expecting either a graph_type of "walk" or "drive"!')

        # perform conversion
        kmph = (orig_len / 1000) / speed
        in_seconds = kmph * 60 * 60
        data[output_time_col] = in_seconds

        # And state the mode, too
        data['mode'] = graph_type

    return G_adj

def assign_traffic_times(G, mb_token, accepted_road_types = ['trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','motorway','motorway_link'], verbose = False, road_col = 'infra_type', id_col = 'id'):
    """
    Function for querying travel times from the Mapbox "driving traffic" API. Queries are only made for the specified road types.

    :param G: a graph object of the road network
    :param mb_token: Mapbox token (retrieve from Mapbox account, starts with "pk:")
    :param road_types: a list of OSM road types for which to query traffic-aware travel time, defaults to main roads
    :param verbose: Set to true to monitor progress of queries and notify if any queries failed, defaults to False
    :param road_col: key for the road type in the edge data dictionary, defaults to 'infra_type'
    :param id_col: key for the id in the edge data dictionary, defaults to 'id'
    :returns: The original graph with two new data properties for the edges: 'mapbox_api' (a boolean set to True if the edge succesfuly received a trafic time value) and 'time_traffic' (travel time in seconds)
    """

    import json, time
    import urllib.request as url

    edges_all = edge_gdf_from_graph(G)

    def first_val(x):
      if isinstance(x, list):
        return x[0]
      else:
        return x

    edges_all[road_col] = edges_all[road_col].apply(lambda x: first_val(x))

    # print('print edges_all')
    # print(edges_all[road_col][390:400])

    print('print unique roads')
    # may not of orginally worked because some fields can contain multiple road tags in a list. Ex. [motorway, trunk]. need to do pre-processing
    print(edges_all[road_col].unique())

    print('print accepted_road_types')
    print(accepted_road_types)

    # pre-process the id_col to make sure it has only one value, sometimes the osmid column can contain a list
    edges_all[id_col] = edges_all[id_col].apply(lambda x: first_val(x))

    # specific rows can be selected by using .isin method on a series.
    edges = edges_all[edges_all[road_col].isin(accepted_road_types)].copy()

    base_url = 'https://api.mapbox.com/directions/v5/mapbox/driving-traffic/'
    end_url = f'?&access_token={mb_token}'
    numcalls = 0
    loop_count = 1

    function_start = time.time()
    start = time.time()

    for idx, row in edges.iterrows():

        # build request
        start_x = G.nodes[row.stnode]['x']
        start_y = G.nodes[row.stnode]['y']
        end_x = G.nodes[row.endnode]['x']
        end_y = G.nodes[row.endnode]['y']
        coordinates = str(start_x)+','+str(start_y)+';'+str(end_x)+','+str(end_y)
        request = base_url+coordinates+end_url
        r = url.urlopen(request)

        try:
            data = json.loads(r.read().decode('utf-8'))['routes'][0]['duration']
        except:
            data = np.nan

        # print(data)

        # assign response duration value to edges df
        edges.at[idx,'duration'] = data

        numcalls += 1
        if numcalls == 299:

            elapsed_seconds = (time.time() - start)%60
            # print('print elapsed_seconds without %: ' + str((time.time() - start)))
            # print('print elapsed_seconds: ' + str(elapsed_seconds))
            if verbose == True: print(f"Did {numcalls+1} calls in {elapsed_seconds:.2f} seconds, now wait {60-elapsed_seconds:.2f}, {(300*loop_count)/len(edges):.2%} complete")
            time.sleep(60-elapsed_seconds)

            # reset count
            numcalls = 0
            start = time.time()
            loop_count += 1

    edges['newID'] = edges['stnode'].astype(str)+"_"+edges['endnode'].astype(str)+"_"+edges[id_col].astype(str)
    edges_duration = edges[['newID','duration']].copy()
    edges_duration = edges_duration.set_index('newID')
    n_null = edges_duration.isnull().sum()['duration']

    if verbose == True and n_null > 0: print(f'query failed {n_null} times')

    edges_duration = edges_duration.dropna()

    for u, v, data in G.edges(data = True):
        newID = str(u) + "_" + str(v) + "_" + str(data[id_col])
        if newID in edges_duration.index:
            data['time_mapbox'] = edges_duration.loc[newID,'duration']
            data['mapbox_api'] = True
        else:
            data['mapbox_api'] = False

    print('complete function time: ' + str(time.time() - function_start))

    return G

def calculate_OD(G, origins, destinations, fail_value, weight = 'time', weighted_origins = False, one_way_roads_exist = False, verbose = False):
    """
    Function for generating an origin: destination matrix

    :param G: a graph containing one or more nodes
    :param fail_value: the value to return if the trip cannot be completed (implies some sort of disruption / disconnected nodes)
    :param origins: a list of the node IDs to treat as origins points
    :param destinations: a list of the node IDs to treat as destinations
    :param weight: use edge weight of 'time' unless otherwise specified
    :param weighted_origins: equals 'true' if the origins have weights. If so, the input to 'origins' must be dictionary instead of a list, where the keys are the origin IDs and the values are the weighted demands.
    :one_way_roads_exist: If the value is 'True', then even if there are more origins than destinations, it will not do a flip during processing.
    :returns: a numpy matrix of format OD[o][d] = shortest time possible
    """

    # Error checking
    G_edges = edge_gdf_from_graph(G)
    if len(G_edges.loc[G_edges[weight].isnull()]) > 0:
        raise ValueError('One or more of your edges has a null weight value')
    if len(G_edges.loc[G_edges[weight]==0]) > 0:
        raise ValueError('One or more of your edges has a 0 weight value')
    if len(G_edges.loc[G_edges['stnode'].isnull()]) > 0:
        raise ValueError('One or more of your edges has a null stnode')
    if len(G_edges.loc[G_edges['endnode'].isnull()]) > 0:
        raise ValueError('One or more of your edges has a null endnode')

    count = 0
    start = time.time()

    if weighted_origins == True:
        print('weighted_origins equals true')
        OD = np.zeros((len(origins), len(destinations)))
        #dictionary key length
        o = 0
        #loop through dictionary
        for key,value in origins.items():
            origin = key
            for d in range(0,len(destinations)):
                destination = destinations[d]
                #find the shortest distance between the origin and destination
                distance = nx.dijkstra_path_length(G, origin, destination, weight = weight)
                # calculate weighted distance
                weighted_distance = distance * float(value)
                OD[o][d] = weighted_distance
            o += 1

    else:
        flip = 0
        if one_way_roads_exist == False:
            if len(origins) > len(destinations):
                flip = 1
                o_2 = destinations
                destinations = origins
                origins = o_2

        #origins will be number or rows, destinations will be number of columns
        OD = np.zeros((len(origins), len(destinations)))

        for o in range(0, len(origins)):
            origin = origins[o]

            if count % 1000 == 0 and verbose == True:
                print("Processing %s of %s" % (count, len(origins)))
                print('seconds elapsed: ' + str(time.time() - start))
            count += 1

            try:
                results_dict = nx.single_source_dijkstra_path_length(G, origin, cutoff = None, weight = weight)
            except Exception as e:
                print(f"error: printing origin: {origin}")
                print(e)

            for d in range(0, len(destinations)):
                destination = destinations[d]
                if destination in results_dict.keys():
                    OD[o][d] = results_dict[destination]
                else:
                    OD[o][d] = fail_value

        if flip == 1:
            OD = np.transpose(OD)

    return OD

def disrupt_network(G, property, thresh, fail_value):
    """
    Function for disrupting a graph given a threshold value against a node's value. Any edges which bind to broken nodes have their 'time' property set to fail_value

    :param G: REQUIRED a graph containing one or more nodes and one or more edges
    :param property: the element in the data dictionary for the edges to test
    :param thresh: values of data[property] above this value are disrupted
    :param fail_value: The data['time'] property is set to this value to simulate the removal of the edge
    :returns: a modified graph with the edited 'time' attribute
    """
    G_copy = G.copy()

    broken_nodes = []

    for u, data in G_copy.nodes(data = True):

        if data[property] > thresh:

            broken_nodes.append(u)

    print('nodes disrupted: %s' % len(broken_nodes))
    i = 0
    for u, v, data in G_copy.edges(data = True):

        if u in broken_nodes or v in broken_nodes:

            data['time'] = fail_value
            i+=1

    print('edges disrupted: %s' % i)
    return G_copy

def randomly_disrupt_network(G, edge_frac, fail_value):
    """
    Function for randomly disurpting a network. NOTE: requires the graph to have an 'edge_id' value in the edge data dictionary. This DOES NOT have to be unique.

    :param G: a graph containing one or more nodes and one or more edges
    :param edge_frac: the percentage of edges to destroy. Integer rather than decimal, e.g. 5 = 5% of edges
    :param fail_value: the data['time'] property is set to this value to simulate the removal of the edge      
    :returns: a modified graph with the edited 'time' attribute the list of edge IDs randomly chosen for destruction
    """

    edgeid = []

    for u,v, data in G.edges(data = True):
        edgeid.append(data['edge_id'])

    num_to_destroy = math.floor(len(edgeid) / 2 * (edge_frac / 100))

    destroy_list = list(np.random.randint(low = 0, high = max(edgeid), size = [num_to_destroy]))

    G_adj = G.copy()

    for u, v, data in G_adj.edges(data = True):
        if data['edge_id'] in destroy_list:
            data['time'] = fail_value

    return G_adj, destroy_list

def gravity_demand(G, origins, destinations, weight, maxtrips = 100, dist_decay = 1, fail_value = 99999999999):
    """
    Function for generating a gravity-model based demand matrix. Note: 1 trip will always be returned between an origin and a destination, even if weighting would otherewise be 0.
    :param origins: a list of node IDs. Must be in G.
    :param destinations: a list of node IDs Must be in G.
    :param weight: the gravity weighting of the nodes in the model, e.g. population
    :param fail_value: the data['time'] property is set to this value to simulate the removal of the edge
    :param maxtrips: normalize the number of trips in the resultant function to this number of trip_times
    :param dist_decay: parameter controlling the aggresion of discounting based on distance
    :returns: a numpy array describing the demand between o and d in terms of number of trips
    """

    maxtrips = 100
    dist_decay = 1

    demand = np.zeros((len(origins), len(destinations)))

    shortest_time = Calculate_OD(G, origins, destinations, fail_value)

    for o in range(0, len(origins)):
        for d in range(0, len(destinations)):
            if origins == destinations and o == d:
                demand[o][d] = 0
            else:
                normalized_dist = shortest_time[o][d] / shortest_time.max()
                demand[o][d] = (
                (G.node[origins[o]][weight] *
                G.node[destinations[d]][weight]) *
                np.exp(-1 * dist_decay * normalized_dist)
                )

    demand = ((demand / demand.max()) * maxtrips)
    demand = np.ceil(demand).astype(int)
    return demand

def unbundle_geometry(c):
    """
    Function for unbundling complex geometric objects. Note: shapely MultiLineString objects quickly get complicated. They may not show up when you plot them in QGIS. This function aims to make a .csv 'plottable'

    :param c: any object. This helper function is usually applied in lambda format against a pandas / geopandas dataframe. The idea is to try to return more simple versions of complex geometries for LineString and MultiLineString type objects.
    :returns: an unbundled geometry value that can be plotted.
    """

    if type(c) == list:
        objs = []
        for i in c:
            if type(i) == str:
                J = loads(i)
                if type(J) == LineString:
                    objs.append(J)
                if type(J) == MultiLineString:
                    for j in J:
                        objs.append(j)
            elif type(i) == MultiLineString:
                for j in i:
                    objs.append(j)
            elif type(i) == LineString:
                objs.append(i)
            else:
                pass
            mls = MultiLineString(objs)
            ls = linemerge(mls)
        return ls
    elif type(c) == str:
        return loads(c)
    else:
        return c

def save(G, savename, wpath, pickle = True, edges = True, nodes = True):
    """
    function used to save a graph object in a variety of handy formats

    :param G: a graph object
    :param savename: the filename, WITHOUT extension
    :param wpath: the write path for where the user wants the files saved
    :param pickle: if set to false, will not save a pickle of the graph
    :param edges: if set to false, will not save an edge gdf
    :param nodes: if set to false, will not save a node gdf
    """

    if nodes == True:
        new_node_gdf = node_gdf_from_graph(G)
        new_node_gdf.to_csv(os.path.join(wpath, '%s_nodes.csv' % savename))
    if edges == True:
        new_edge_gdf = edge_gdf_from_graph(G)
        new_edge_gdf.to_csv(os.path.join(wpath, '%s_edges.csv' % savename))
    if pickle == True:
        nx.write_gpickle(G, os.path.join(wpath, '%s.pickle' % savename))

def add_missing_reflected_edges(G, one_way_tag = None, verbose = False):
    """
    function for adding any missing reflected edges - makes all edges bidirectional. This is essential for routing with simplified graphs

    :param G: a graph object
    :param one_way_tag: if exists, then values that are True are one-way and will not be reflected
    """
    #unique_edges = []
    missing_edges = []

#     for u, v in G.edges(data = False):
#         unique_edges.append((u,v))

    edgeLength = G.number_of_edges()
    count = 0
    start = time.time()
    for u, v, data in G.edges(data = True):
        if count % 10000 == 0 and verbose == True:
                print("Processing %s of %s" % (count, edgeLength))
                print('seconds elapsed: ' + str(time.time() - start))
        count += 1
        if one_way_tag:
            # print("print one_way_tag")
            # print(one_way_tag)
            # print("print data")
            # print(data)
            # print("data[one_way_tag]")
            # print(data[one_way_tag])
            if data[one_way_tag] == False:
                #print("2-way road")
                #if (v, u) not in unique_edges:
                    #print("appending to missing_edges")
                missing_edges.append((v,u,data))
        else:
            #if (v, u) not in unique_edges:
            missing_edges.append((v,u,data))

    G2 = G.copy()
    G2.add_edges_from(missing_edges)
    print(f"completed processing {G2.number_of_edges()} edges")
    return G2

def add_missing_reflected_edges_old(G, one_way_tag=None):
    """
    to-do: delete this function, it is slower, creating a unique edge list slows things down with a big graph

    function for adding any missing reflected edges - makes all edges bidirectional. This is essential for routing with simplified graphs

    :param G: a graph object
    :param one_way_tag: if exists, then values that are True are one-way and will not be reflected
    """
    unique_edges = []
    missing_edges = []

    for u, v in G.edges(data = False):
        unique_edges.append((u,v))

    for u, v, data in G.edges(data = True):
        if one_way_tag:
            # print("print one_way_tag")
            # print(one_way_tag)
            # print("print data")
            # print(data)
            # print("data[one_way_tag]")
            # print(data[one_way_tag])
            if data[one_way_tag] == False:
                #print("2-way road")
                if (v, u) not in unique_edges:
                    #print("appending to missing_edges")
                    missing_edges.append((v,u,data))
        else:
            if (v, u) not in unique_edges:
                missing_edges.append((v,u,data))

    G2 = G.copy()
    G2.add_edges_from(missing_edges)
    print(G2.number_of_edges())
    return G2

def remove_duplicate_edges(G, max_ratio = 1.5):
    """
    function for deleting duplicated edges - where there is more than one edge connecting a node pair. USE WITH CAUTION - will change both topological relationships and node maps
    :param G: a graph object
    :param max_ratio: most of the time we see duplicate edges that are clones of each other. Sometimes, however, there are valid duplicates. These occur if multiple roads connect two junctions uniquely and without interruption - e.g. two roads running either side of a lake which meet at either end. The idea here is that valid 'duplicate edges' will have geometries of materially different length. Hence, we include a ratio - defaulting to 1.5 - beyond which we are sure the duplicates are valid edges, and will not be deleted.
    """

    G2 = G.copy()
    uniques = []
    deletes = []
    for u, v, data in G2.edges(data = True):
        if (u,v) not in uniques:
            uniques.append((v,u))
            t = G2.number_of_edges(u, v)
            lengths = []
            for i in range(0,t):
                lengths.append(G2.edges[u,v,i]['length'])
            if max(lengths) / min(lengths) >= max_ratio:
                pass
            else:
                deletes.append((u,v))

    for d in deletes:
        G2.remove_edge(d[0],d[1])
    print(G2.number_of_edges())
    return G2

def convert_to_MultiDiGraph(G):
    """
    takes any graph object, loads it into a MultiDiGraph type Networkx object
    :param G: a graph object
    """
    a = nx.MultiDiGraph()

    node_bunch = []
    for u, data in G.nodes(data = True):
        node_bunch.append((u,data))

    a.add_nodes_from(node_bunch)

    edge_bunch = []
    for u, v, data in G.edges(data = True):
        if 'Wkt' in data.keys():
            data['Wkt'] = str(data['Wkt'])
        edge_bunch.append((u,v,data))

    a.add_edges_from(edge_bunch)
    return a

#### NETWORK SIMPLIFICATION ####

def simplify_junctions(G, measure_crs, in_crs = {'init': 'epsg:4326'}, thresh = 25, verbose = False):
    """
    simplifies topology of networks by simplifying node clusters into single nodes.

    :param G: a graph object
    :param measure_crs: the crs to make the measurements inself.
    :param in_crs: the current crs of the graph's geometry properties. By default, assumes WGS 84 (epsg 4326)
    :param thresh: the threshold distance in which to simplify junctions. By default, assumes 25 meters
    """

    G2 = G.copy()

    gdfnodes = node_gdf_from_graph(G2)
    gdfnodes_proj_buffer = gdfnodes.to_crs(measure_crs)
    gdfnodes_proj_buffer = gdfnodes_proj_buffer.buffer(thresh)

    # Get the version of Pandas
    pandas_version = pd.__version__

    # Compare the major version number
    if int(pandas_version.split('.')[0]) >= 2:
        # passing index to be compatible with Pandas ver 2.0
        juncs_pd = pd.DataFrame({'geometry':unary_union(gdfnodes_proj_buffer)}, index=[0])
    else:
        juncs_pd = pd.DataFrame({'geometry':unary_union(gdfnodes_proj_buffer)})

    juncs_gdf = gpd.GeoDataFrame(juncs_pd, crs = measure_crs, geometry = 'geometry')
    juncs_gdf['area'] = juncs_gdf.area

    juncs_gdf_2 = juncs_gdf.copy()
    juncs_gdf_2 = juncs_gdf_2.loc[juncs_gdf_2.area > int(juncs_gdf.area.min() + 1)]
    juncs_gdf = juncs_gdf_2
    juncs_gdf = juncs_gdf.reset_index()
    juncs_gdf['obj_ID'] = juncs_gdf.index
    juncs_gdf['obj_ID'] = 'new_obj_'+juncs_gdf['obj_ID'].astype(str)

    juncs_gdf_unproj = juncs_gdf.to_crs(in_crs)
    juncs_gdf_unproj['centroid'] = juncs_gdf_unproj.centroid
    juncs_gdf_bound = gpd.sjoin(juncs_gdf_unproj, gdfnodes, how='left', op='intersects', lsuffix='left', rsuffix='right')
    juncs_gdf_bound = juncs_gdf_bound[['obj_ID','centroid','node_ID']]

    node_map = juncs_gdf_bound[['obj_ID','node_ID']]
    node_map = node_map.set_index('node_ID')
    node_dict = node_map['obj_ID'].to_dict()
    nodes_to_be_destroyed = list(node_dict.keys())

    centroid_map = juncs_gdf_bound[['obj_ID','centroid']]
    centroid_map = centroid_map.set_index('obj_ID')
    centroid_dict = centroid_map['centroid'].to_dict()
    new_node_IDs = list(centroid_dict.keys())

    # Add the new centroids of the junction areas as new nodes
    new_nodes = []
    for i in new_node_IDs:
        new_nodes.append((i, {'x':centroid_dict[i].x, 'y':centroid_dict[i].y}))
    G2.add_nodes_from(new_nodes)

    # modify edges - delete those where both u and v are to be removed, edit the others
    edges_to_be_destroyed = []
    new_edges = []

    count = 0
    start = time.time()
    edgeLength = G2.number_of_edges()

    for u, v, data in G2.edges(data = True):

        if count % 10000 == 0 and verbose == True:
            print("Processing %s of %s" % (count, edgeLength))
            print('seconds elapsed: ' + str(time.time() - start))
        count += 1

        if type(data['Wkt']) == LineString:
            l = data['Wkt']
        else:
            l = loads(data['Wkt'])

        line_to_be_edited = l.coords

        if u in nodes_to_be_destroyed and v in nodes_to_be_destroyed:
            if node_dict[u] == node_dict[v]:
                edges_to_be_destroyed.append((u,v))

            else:
                new_ID_u = node_dict[u]
                new_point_u = centroid_dict[new_ID_u]
                new_ID_v = node_dict[v]
                new_point_v = centroid_dict[new_ID_v]

                if len(line_to_be_edited) > 2:
                    data['Wkt'] = LineString([new_point_u, *line_to_be_edited[1:-1], new_point_v])
                else:
                    data['Wkt'] = LineString([new_point_u, new_point_v])
                data['Type'] = 'dual_destruction'

                new_edges.append((new_ID_u,new_ID_v,data))
                edges_to_be_destroyed.append((u,v))

        else:

            if u in nodes_to_be_destroyed:
                new_ID_u = node_dict[u]
                u = new_ID_u

                new_point = centroid_dict[new_ID_u]
                coords = [new_point, *line_to_be_edited[1:]]
                data['Wkt'] = LineString(coords)
                data['Type'] = 'origin_destruction'

                new_edges.append((new_ID_u,v,data))
                edges_to_be_destroyed.append((u,v))

            elif v in nodes_to_be_destroyed:
                new_ID_v = node_dict[v]
                v = new_ID_v

                new_point = centroid_dict[new_ID_v]
                coords = [*line_to_be_edited[:-1], new_point]
                data['Wkt'] = LineString(coords)
                data['Type'] = 'destination_destruction'

                new_edges.append((u,new_ID_v,data))
                edges_to_be_destroyed.append((u,v))

            else:
                data['Type'] = 'legitimate'
                pass

    # remove old edges that connected redundant nodes to each other / edges where geometry needed to be changed
    G2.remove_edges_from(edges_to_be_destroyed)

    # ... and add any corrected / new edges
    G2.add_edges_from(new_edges)

    # remove now redundant nodes
    G2.remove_nodes_from(nodes_to_be_destroyed)

    print(G2.number_of_edges())

    return G2

def custom_simplify(G, strict=True):
    """
    Simplify a graph's topology by removing all nodes that are not intersections or dead-ends. Create an edge directly between the end points that encapsulate them, but retain the geometry of the original edges, saved as attribute in new edge.

    :param G: networkx multidigraph
    :param bool strict: if False, allow nodes to be end points even if they fail all other rules but have edges with different OSM IDs
    :returns: networkx multidigraph
    """

    def get_paths_to_simplify(G, strict=True):

        """
        Create a list of all the paths to be simplified between endpoint nodes.

        The path is ordered from the first endpoint, through the interstitial nodes,
        to the second endpoint. If your street network is in a rural area with many
        interstitial nodes between true edge endpoints, you may want to increase
        your system's recursion limit to avoid recursion errors.

        Parameters
        ----------
        G : networkx multidigraph
        strict : bool
            if False, allow nodes to be end points even if they fail all other rules
            but have edges with different OSM IDs

        Returns
        -------
        paths_to_simplify : list
        """

        # first identify all the nodes that are endpoints
        start_time = time.time()
        endpoints = set([node for node in G.nodes() if is_endpoint(G, node, strict=strict)])

        start_time = time.time()
        paths_to_simplify = []

        # for each endpoint node, look at each of its successor nodes
        for node in endpoints:
            for successor in G.successors(node):
                if successor not in endpoints:
                    # if the successor is not an endpoint, build a path from the
                    # endpoint node to the next endpoint node
                    try:
                        path = build_path(G, successor, endpoints, path=[node, successor])
                        paths_to_simplify.append(path)
                    except RuntimeError:
                        # recursion errors occur if some connected component is a
                        # self-contained ring in which all nodes are not end points.
                        # could also occur in extremely long street segments (eg, in
                        # rural areas) with too many nodes between true endpoints.
                        # handle it by just ignoring that component and letting its
                        # topology remain intact (this should be a rare occurrence)
                        # RuntimeError is what Python <3.5 will throw, Py3.5+ throws
                        # RecursionError but it is a subtype of RuntimeError so it
                        # still gets handled
                        pass

        return paths_to_simplify

    def is_endpoint(G, node, strict=True):
        """
        Return True if the node is a "real" endpoint of an edge in the network, \
        otherwise False. OSM data includes lots of nodes that exist only as points \
        to help streets bend around curves. An end point is a node that either: \
        1) is its own neighbor, ie, it self-loops. \
        2) or, has no incoming edges or no outgoing edges, ie, all its incident \
            edges point inward or all its incident edges point outward. \
        3) or, it does not have exactly two neighbors and degree of 2 or 4. \
        4) or, if strict mode is false, if its edges have different OSM IDs. \

        Parameters
        ----------
        G : networkx multidigraph

        node : int
            the node to examine
        strict : bool
            if False, allow nodes to be end points even if they fail all other rules \
            but have edges with different OSM IDs

        Returns
        -------
        bool
        """

        neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
        n = len(neighbors)
        d = G.degree(node)

        if node in neighbors:
            # if the node appears in its list of neighbors, it self-loops. this is
            # always an endpoint.
            return 'node in neighbours'

        # if node has no incoming edges or no outgoing edges, it must be an endpoint
        #elif G.out_degree(node)==0 or G.in_degree(node)==0:
            #return 'no in or out'

        elif not (n==2 and (d==2 or d==4)):
            # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
            # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
            # case it is a dead-end or an intersection of multiple streets or it has
            # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
            # or more than 4 degree (indicating a parallel edge) and thus is an
            # endpoint
            return 'condition 3'

        elif not strict:
            # non-strict mode
            osmids = []

            # add all the edge OSM IDs for incoming edges
            for u in G.predecessors(node):
                for key in G[u][node]:
                    osmids.append(G.edges[u, node, key]['osmid'])

            # add all the edge OSM IDs for outgoing edges
            for v in G.successors(node):
                for key in G[node][v]:
                    osmids.append(G.edges[node, v, key]['osmid'])

            # if there is more than 1 OSM ID in the list of edge OSM IDs then it is
            # an endpoint, if not, it isn't
            return len(set(osmids)) > 1

        else:
            # if none of the preceding rules returned true, then it is not an endpoint
            return False

    def build_path(G, node, endpoints, path):
        """
        Recursively build a path of nodes until you hit an endpoint node.

        :param G: networkx multidigraph
        :param int node: the current node to start from
        :param set endpoints: the set of all nodes in the graph that are endpoints
        :param list path: the list of nodes in order in the path so far
        :returns list: paths_to_simplify
        """

        # for each successor in the passed-in node
        for successor in G.successors(node):
            if successor not in path:
                # if this successor is already in the path, ignore it, otherwise add
                # it to the path
                path.append(successor)
                if successor not in endpoints:
                    # if this successor is not an endpoint, recursively call
                    # build_path until you find an endpoint
                    path = build_path(G, successor, endpoints, path)
                else:
                    # if this successor is an endpoint, we've completed the path,
                    # so return it
                    return path

        if (path[-1] not in endpoints) and (path[0] in G.successors(path[-1])):
            # if the end of the path is not actually an endpoint and the path's
            # first node is a successor of the path's final node, then this is
            # actually a self loop, so add path's first node to end of path to
            # close it
            path.append(path[0])

        return path

    ## MAIN PROCESS FOR CUSTOM SIMPLIFY ##

    G = G.copy()

    if type(G) != nx.classes.multidigraph.MultiDiGraph:
        G = ConvertToMultiDiGraph(G)
    initial_node_count = len(list(G.nodes()))
    initial_edge_count = len(list(G.edges()))
    all_nodes_to_remove = []
    all_edges_to_add = []

    # construct a list of all the paths that need to be simplified
    paths = get_paths_to_simplify(G, strict=strict)

    start_time = time.time()
    for path in paths:

        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        edge_attributes = {}
        for u, v in zip(path[:-1], path[1:]):

            # there shouldn't be multiple edges between interstitial nodes
            if not G.number_of_edges(u, v) == 1:
                pass
            # the only element in this list as long as above check is True
            # (MultiGraphs use keys (the 0 here), indexed with ints from 0 and
            # up)
            edge = G.edges[u, v, 0]
            for key in edge:
                if key in edge_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    edge_attributes[key].append(edge[key])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    edge_attributes[key] = [edge[key]]

        for key in edge_attributes:
            # don't touch the length attribute, we'll sum it at the end
            if key == 'Wkt':
                edge_attributes['Wkt'] = list(edge_attributes['Wkt'])
            elif key != 'length' and key != 'Wkt':      # if len(set(edge_attributes[key])) == 1 and not key == 'length':
                # if there's only 1 unique value in this attribute list,
                # consolidate it to the single value (the zero-th)
                edge_attributes[key] = edge_attributes[key][0]
            elif not key == 'length':
                # otherwise, if there are multiple values, keep one of each value
                edge_attributes[key] = list(set(edge_attributes[key]))

        # construct the geometry and sum the lengths of the segments
        edge_attributes['geometry'] = LineString([Point((G.nodes[node]['x'], G.nodes[node]['y'])) for node in path])
        edge_attributes['length'] = sum(edge_attributes['length'])

        # add the nodes and edges to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append({'origin':path[0],
                                 'destination':path[-1],
                                 'attr_dict':edge_attributes})

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge['origin'], edge['destination'], **edge['attr_dict'])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    msg = 'Simplified graph (from {:,} to {:,} nodes and from {:,} to {:,} edges) in {:,.2f} seconds'
    return G

def salt_long_lines(G, source, target, thresh = 5000, factor = 1, attr_list = None):
    """
    Adds in new nodes to edges greater than a given length

    :param G: a graph object
    :param source: crs object in format 'epsg:4326'
    :param target: crs object in format 'epsg:32638'
    :param thresh: distance in metres after which to break edges.
    :param factor: edge lengths can be returned in units other than metres by specifying a numerical multiplication factor. Factor behavior divides rather than multiplies.
    :param attr_dict: list of attributes to be saved onto new edges.
    """

    def cut(line, distance):
        # Cuts a line in two at a distance from its starting point
        if distance <= 0.0 or distance >= line.length:
            return [LineString(line)]

        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return [LineString(coords[:i+1]),LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance)
                return [LineString(coords[:i] + [(cp.x, cp.y)]),LineString([(cp.x, cp.y)] + coords[i:])]

    G2 = G.copy()
    edges = edge_gdf_from_graph(G2, geometry_tag = 'Wkt')
    edges_projected = edges.to_crs(target)
    nodes_projected = node_gdf_from_graph(G).to_crs(target).set_index('node_ID')

    # define transforms for exchanging between source and target projections

    #print(f"pyproj ver: {pyproj.__version__}")

    # pyproj < 2.1
    # project_WGS_UTM = partial(
    #             pyproj.transform,
    #             pyproj.Proj(init=source),
    #             pyproj.Proj(init=target))

    
    # project_UTM_WGS = partial(
    #             pyproj.transform,
    #             pyproj.Proj(init=target),
    #             pyproj.Proj(init=source))

    # pyproj >= 2.1.0
    # repeated transformations using the same inProj and outProj, using the Transformer object in pyproj 2+ is much faster
    wgs84 = pyproj.CRS(source)
    utm = pyproj.CRS(target)
    project_WGS_UTM = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    project_UTM_WGS = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

    long_edges, long_edge_IDs, unique_long_edges, new_nodes, new_edges = [], [], [], [], []

    #return 'done'

    # Identify long edges
    for idx, data in edges_projected.iterrows():

        u = data['stnode']
        v = data['endnode']

        # load geometry
        UTM_geom = data['Wkt']

        # test geometry length
        if UTM_geom.length > thresh:
            long_edges.append((u, v, data))
            long_edge_IDs.append((u,v))
            if (v, u) in long_edge_IDs:
                pass
            else:
                unique_long_edges.append((u, v, data))

    print('Identified %d unique edge(s) longer than %d. \nBeginning new node creation...' % (len(unique_long_edges), thresh))

    # iterate through one long edge for each bidirectional long edge pair

    j,o = 1, 0

    for u, v, data in unique_long_edges:

        # load geometry of long edge
        UTM_geom = data['Wkt']

        if UTM_geom.type == 'MultiLineString':
            UTM_geom = linemerge(UTM_geom)

        # flip u and v if Linestring running from v to u, coordinate-wise
        u_x_cond = round(UTM_geom.coords[0][0], 3) == round(nodes_projected.loc[u, 'geometry'].x, 3)
        u_y_cond = round(UTM_geom.coords[0][1], 3) == round(nodes_projected.loc[u, 'geometry'].y, 3)

        v_x_cond = round(UTM_geom.coords[0][0], 3) == round(nodes_projected.loc[v, 'geometry'].x, 3)
        v_y_cond = round(UTM_geom.coords[0][1], 3) == round(nodes_projected.loc[v, 'geometry'].y, 3)

        if u_x_cond and u_y_cond:
            pass
        elif v_x_cond and v_y_cond:
            u, v = v, u
        else:
            print('ERROR!')

        # calculate number of new nodes to add along length
        number_of_new_points = UTM_geom.length / thresh

        # for each new node
        for i in range(0, int(number_of_new_points+1)):

            ## GENERATE NEW NODES ##

            cur_dist = (thresh * (i+1))

            # generate new geometry along line
            new_point = UTM_geom.interpolate(cur_dist)

            new_point_WGS = transform(project_UTM_WGS, new_point)
            #print(f"new way new_point_WGS: {new_point_WGS}")

            node_data = {'geometry': new_point_WGS,
                        'x' : new_point_WGS.x,
                        'y': new_point_WGS.y}

            new_node_ID = str(u)+'_'+str(i+j)+'_'+str(o)

            # generate a new node as long as it isn't the final node
            if i < int(number_of_new_points):
                new_nodes.append((new_node_ID, node_data))

            ## GENERATE NEW EDGES ##
            # define geometry to be cutting (iterative)
            if i == 0:
                geom_to_split = UTM_geom

            else:
                geom_to_split = result[1]

            # cut geometry. result[0] is the section cut off, result[1] is remainder
            result = cut(geom_to_split, (thresh))

            #print(f"print result: {result[0]}")

            t_geom = transform(project_UTM_WGS, result[0])
            #print(f"new way t_geom: {t_geom}")

            edge_data = {'Wkt' : t_geom,
                        'length' : (int(result[0].length) / factor),
                        }

            if attr_list != None:
                for attr in attr_list:
                    if attr in data:
                        edge_data[attr] = data[attr]

            if i == 0:
                prev_node_ID = u

            if i == int(number_of_new_points):
                new_node_ID = v

            # append resulting edges to a list of new edges, bidirectional.
            new_edges.append((prev_node_ID,new_node_ID,edge_data))
            new_edges.append((new_node_ID,prev_node_ID,edge_data))

            o += 1

            prev_node_ID = new_node_ID

        j+=1

    # add new nodes and edges
    G2.add_nodes_from(new_nodes)
    G2.add_edges_from(new_edges)

    # remove the too-long edges
    for d in long_edges:
        G2.remove_edge(d[0],d[1])

    print('%d new edges added and %d removed to bring total edges to %d' % (len(new_edges),len(long_edges),G2.number_of_edges()))
    print('%d new nodes added to bring total nodes to %d' % (len(new_nodes),G2.number_of_nodes()))

    return G2

def pandana_snap(G, point_gdf, source_crs = 'epsg:4326', target_crs = 'epsg:4326', 
                    add_dist_to_node_col = True, time_it = False):
    """
    snaps points to a graph at very high speed
    :param G: a graph object, or the node geodataframe of a graph
    :param point_gdf: a geodataframe of points, in the same source crs as the geometry of the graph object
    :param source_crs: The crs for the input G and input point_gdf in format 'epsg:32638' 
    :param target_crs: The measure crs how distances between points are calculated. The returned point GeoDataFrame's CRS does not get modified. The crs object in format 'epsg:32638'
    :param add_dist_to_node_col: return distance to nearest node in the units of the target_crs
    :return: returns a GeoDataFrame that is the same as the input point_gdf but adds a column containing the id of the nearest node in the graph, and the distance if add_dist_to_node_col == True
    """
    import time

    if time_it == True:
      func_start = time.time()

    in_df = point_gdf.copy()

    # check if in_df has a geometry column, or else provide warning
    if not set(['geometry']).issubset(in_df.columns):
        raise Exception('input point_gdf should have a geometry column')

    if isinstance(G,nx.classes.multidigraph.MultiDiGraph) == True:
        node_gdf = node_gdf_from_graph(G)
    else:
        node_gdf = G

    if node_gdf.x.isnull().values.any() or node_gdf.y.isnull().values.any():
        raise Exception("some of the graph's x or y values contains null values, exiting")

    if add_dist_to_node_col is True:

        # only need to re-project if source is different than the target
        if source_crs != target_crs:

            in_df_proj = in_df.to_crs(f'{target_crs}')
            in_df_proj['x'] = in_df_proj.geometry.x
            in_df_proj['y'] = in_df_proj.geometry.y

            # print('print in_df')
            # print(in_df_proj)

            node_gdf_proj = node_gdf.to_crs(f'{target_crs}')
            node_gdf_proj['x'] = node_gdf_proj.geometry.x
            node_gdf_proj['y'] = node_gdf_proj.geometry.y

            G_tree = spatial.KDTree(node_gdf_proj[['x','y']].values)

            distances, indices = G_tree.query(in_df_proj[['x','y']].values)

            in_df['NN'] = list(node_gdf_proj['node_ID'].iloc[indices])
            in_df['NN_dist'] = distances

            #in_df = in_df.drop(['x','y','Proj_geometry'], axis = 1)

        else:

            try:
                in_df['x'] = in_df.geometry.x
                in_df['y'] = in_df.geometry.y
            except:
                in_df['x'] = in_df.geometry.apply(lambda geometry: geometry.x)
                in_df['y'] = in_df.geometry.apply(lambda geometry: geometry.y)

            G_tree = spatial.KDTree(node_gdf[['x','y']].values)
            distances, indices = G_tree.query(in_df[['x','y']].values)

            in_df['NN'] = list(node_gdf['node_ID'].iloc[indices])
            in_df['NN_dist'] = distances

    else:
        try:
            in_df['x'] = in_df.geometry.x
            in_df['y'] = in_df.geometry.y
        except:
            in_df['x'] = in_df.geometry.apply(lambda geometry: geometry.x)
            in_df['y'] = in_df.geometry.apply(lambda geometry: geometry.y)

        G_tree = spatial.KDTree(node_gdf[['x','y']].values)
        distances, indices = G_tree.query(in_df[['x','y']].values)

        in_df['NN'] = list(node_gdf['node_ID'].iloc[indices])

    if time_it == True:
        func_end = time.time()
        print('time elapsed for function')
        print(func_end - func_start)

    return in_df

def pandana_snap_c(G, point_gdf, source_crs = 'epsg:4326', target_crs = 'epsg:4326', 
                    add_dist_to_node_col = True, time_it = False):
    """
    snaps points to a graph at a faster speed than pandana_snap.
    :param G: a graph object, or the node geodataframe of a graph
    :param point_gdf: a geodataframe of points, in the same source crs as the geometry of the graph object
    :param source_crs: The crs for the input G and input point_gdf in format 'epsg:32638' 
    :param target_crs: The measure crs how distances between points are calculated. The returned point GeoDataFrame's CRS does not get modified. The crs object in format 'epsg:32638'
    :param add_dist_to_node_col: return distance to nearest node in the units of the target_crs
    :param time_it: return time to complete function
    :return: returns a GeoDataFrame that is the same as the input point_gdf but adds a column containing the id of the nearest node in the graph, and the distance if add_dist_to_node_col == True
    """
    import time

    if time_it == True:
      func_start = time.time()

    in_df = point_gdf.copy()

    # check if in_df has a geometry column, or else provide warning
    if not set(['geometry']).issubset(in_df.columns):
        raise Exception('input point_gdf should have a geometry column')

    if isinstance(G,nx.classes.multidigraph.MultiDiGraph) == True:
        node_gdf = node_gdf_from_graph(G)
    else:
        node_gdf = G

    if node_gdf.x.isnull().values.any() or node_gdf.y.isnull().values.any():
        raise Exception("some of the graph's x or y values contains null values, exiting")

    if add_dist_to_node_col is True:

        # only need to re-project if source is different than the target
        if source_crs != target_crs:

            in_df_proj = in_df.to_crs(f'{target_crs}')
            in_df_proj['x'] = in_df_proj.geometry.x
            in_df_proj['y'] = in_df_proj.geometry.y

            # print('print in_df')
            # print(in_df_proj)

            node_gdf_proj = node_gdf.to_crs(f'{target_crs}')
            node_gdf_proj['x'] = node_gdf_proj.geometry.x
            node_gdf_proj['y'] = node_gdf_proj.geometry.y

            G_tree = spatial.cKDTree(node_gdf_proj[['x','y']].values)

            distances, indices = G_tree.query(in_df_proj[['x','y']].values)

            in_df['NN'] = list(node_gdf_proj['node_ID'].iloc[indices])
            in_df['NN_dist'] = distances

        else:

            try:
                in_df['x'] = in_df.geometry.x
                in_df['y'] = in_df.geometry.y
            except:
                in_df['x'] = in_df.geometry.apply(lambda geometry: geometry.x)
                in_df['y'] = in_df.geometry.apply(lambda geometry: geometry.y)

            G_tree = spatial.cKDTree(node_gdf[['x','y']].values)
            distances, indices = G_tree.query(in_df[['x','y']].values)

            in_df['NN'] = list(node_gdf['node_ID'].iloc[indices])
            in_df['NN_dist'] = distances

    else:

        try:
            in_df['x'] = in_df.geometry.x
            in_df['y'] = in_df.geometry.y
        except:
            in_df['x'] = in_df.geometry.apply(lambda geometry: geometry.x)
            in_df['y'] = in_df.geometry.apply(lambda geometry: geometry.y)

        # .as_matrix() is now depreciated as of Pandas 1.0.0
        #G_tree = spatial.KDTree(node_gdf[['x','y']].as_matrix())
        G_tree = spatial.KDTree(node_gdf[['x','y']].values)

        #distances, indices = G_tree.query(in_df[['x','y']].as_matrix())
        distances, indices = G_tree.query(in_df[['x','y']].values)

        in_df['NN'] = list(node_gdf['node_ID'].iloc[indices])

    if time_it == True:
        func_end = time.time()
        print('time elapsed for function')
        print(func_end - func_start)

    return in_df
    
def pandana_snap_to_many(G, point_gdf, source_crs = 'epsg:4326', target_crs = 'epsg:4326', 
                    add_dist_to_node_col = True, time_it = False, k_nearest=5, origin_id = 'index'):
    """
    snaps points their k nearest neighbors in the graph. 
    :param G: a graph object
    :param point_gdf: a geodataframe of points, in the same source crs as the geometry of the graph object
    :param source_crs: The crs for the input G and input point_gdf in format 'epsg:32638' 
    :param target_crs: The desired crs returned point GeoDataFrame. The crs object in format 'epsg:32638'
    :param add_dist_to_node_col: return distance to nearest node in the units of the target_crs
    :param time_it: return time to complete function
    """
    import time

    if time_it == True:
      func_start = time.time()

    in_df = point_gdf.copy()

    # check if in_df has a geometry column, or else provide warning
    if not set(['geometry']).issubset(in_df.columns):
        raise Exception('input point_gdf should have a geometry column')

    node_gdf = node_gdf_from_graph(G)
    nn_map = {}

    if add_dist_to_node_col is True:

        # only need to re-project if source is different than the target
        if source_crs != target_crs:

            in_df_proj = in_df.to_crs(f'{target_crs}')
            in_df_proj['x'] = in_df_proj.geometry.x
            in_df_proj['y'] = in_df_proj.geometry.y

            # print('print in_df')
            # print(in_df_proj)

            node_gdf_proj = node_gdf.to_crs(f'{target_crs}')
            node_gdf_proj['x'] = node_gdf_proj.geometry.x
            node_gdf_proj['y'] = node_gdf_proj.geometry.y

            G_tree = spatial.cKDTree(node_gdf_proj[['x','y']].values)

            distances, indices = G_tree.query(in_df_proj[['x','y']].values, k = k_nearest)

            for origin, distance_list, index_list in zip(list(in_df[origin_id]), distances, indices):
                index_list_NN = list(node_gdf['node_ID'].iloc[index_list])
                nn_map[origin] = {'NN':list(index_list_NN), 'NN_dist':list(distance_list)}
            
        else:

            try:
                in_df['x'] = in_df.geometry.x
                in_df['y'] = in_df.geometry.y
            except:
                in_df['x'] = in_df.geometry.apply(lambda geometry: geometry.x)
                in_df['y'] = in_df.geometry.apply(lambda geometry: geometry.y)

            G_tree = spatial.cKDTree(node_gdf[['x','y']].values)
            distances, indices = G_tree.query(in_df[['x','y']].values, k = k_nearest)

            for origin, distance_list, index_list in zip(list(in_df[origin_id]), distances, indices):
                index_list_NN = list(node_gdf['node_ID'].iloc[index_list])
                nn_map[origin] = {'NN':list(index_list_NN), 'NN_dist':list(distance_list)}

    else:

        try:
            in_df['x'] = in_df.geometry.x
            in_df['y'] = in_df.geometry.y
        except:
            in_df['x'] = in_df.geometry.apply(lambda geometry: geometry.x)
            in_df['y'] = in_df.geometry.apply(lambda geometry: geometry.y)

        # .as_matrix() is now depreciated as of Pandas 1.0.0
        #G_tree = spatial.KDTree(node_gdf[['x','y']].as_matrix())
        G_tree = spatial.KDTree(node_gdf[['x','y']].values)

        #distances, indices = G_tree.query(in_df[['x','y']].as_matrix())
        distances, indices = G_tree.query(in_df[['x','y']].values, k = k_nearest)

        for origin, distance_list, index_list in zip(list(in_df[origin_id]), distances, indices):
            index_list_NN = list(node_gdf['node_ID'].iloc[index_list])
            nn_map[origin] = {'NN':list(index_list_NN)}

    if time_it == True:
        func_end = time.time()
        print('time elapsed for function')
        print(func_end - func_start)

    return nn_map

def pandana_snap_single_point(G, shapely_point, source_crs = 'epsg:4326', target_crs = 'epsg:4326'):
    """
    snaps a point to a graph at very high speed

    :param G: a graph object
    :param shapely_point: a shapely point (ex. Point(x, y)), in the same source crs as the geometry of the graph object
    :param source_crs: crs object in format 'epsg:32638'
    :param target_crs: crs object in format 'epsg:32638'
    :param add_dist_to_node_col: return distance in metres to nearest node
    """

    node_gdf = node_gdf_from_graph(G)

    G_tree = spatial.KDTree(node_gdf[['x','y']].values)
    distances, indices = G_tree.query([[shapely_point.x,shapely_point.y]])

    #print("print distances, indices")
    #print(distances, indices)

    return_list = list(node_gdf['node_ID'].iloc[indices])

    return return_list[0]

def pandana_snap_points(source_gdf, target_gdf, source_crs = 'epsg:4326', target_crs = 'epsg:4326', add_dist_to_node_col = True):
    """
    snaps points to another GeoDataFrame at very high speed

    :param source_gdf: a geodataframe of points
    :param target_gdf: a geodataframe of points, in the same source crs as the geometry of the source_gdfsg:32638'
    :param target_crs: crs object in format 'epsg:32638'
    :param add_dist_to_node_col: return distance in metres to nearest node
    :return: returns a GeoDataFrame that is the same as the input source_gdf but adds a column containing the id of the nearest node in the target_gdf, and the distance if add_dist_to_node_col == True
    """

    source_gdf = source_gdf.copy()
    target_gdf = target_gdf.copy()
    target_gdf['ID'] = target_gdf.index

    if add_dist_to_node_col is True:
        
        if source_crs != target_crs:
            target_gdf = target_gdf.to_crs(f'{target_crs}')
            source_gdf = source_gdf.to_crs(f'{target_crs}')
        
        target_gdf['x'] = target_gdf.geometry.x
        target_gdf['y'] = target_gdf.geometry.y

        source_gdf['x'] = source_gdf.geometry.x
        source_gdf['y'] = source_gdf.geometry.y

        G_tree = spatial.cKDTree(target_gdf[['x','y']].values)

        distances, indices = G_tree.query(source_gdf[['x','y']].values)

        source_gdf['idx'] = list(target_gdf['ID'].iloc[indices])

        source_gdf['idx_dist'] = distances

        source_gdf = source_gdf.drop(['x','y'], axis = 1)

    else:

        target_gdf['x'] = target_gdf.geometry.x
        target_gdf['y'] = target_gdf.geometry.y
        
        source_gdf['x'] = source_gdf.geometry.x
        source_gdf['y'] = source_gdf.geometry.y

        G_tree = spatial.cKDTree(target_gdf[['x','y']].values)

        distances, indices = G_tree.query(source_gdf[['x','y']].values)

        source_gdf['idx'] = list(target_gdf['ID'].iloc[indices])

    return source_gdf

def join_networks(base_net, new_net, measure_crs, thresh = 500):
    """
    joins two networks together within a binding threshold

    :param base_net: a base network object (nx.MultiDiGraph)
    :param new_net: the network to add on to the base (nx.MultiDiGraph)
    :param measure_crs: the crs number of the measurement (epsg code)
    :param thresh: binding threshold - unit of the crs - default 500m
    """
    G_copy = base_net.copy()
    join_nodes_df = pandana_snap(G_copy,
                         node_gdf_from_graph(new_net),
                         source_crs = 'epsg:4326',
                         target_crs = 'epsg:%s' % measure_crs,
                         add_dist_to_node_col = True)

    join_nodes_df = join_nodes_df.sort_values(by = 'NN_dist', ascending = True)
    join_nodes_df = join_nodes_df.loc[join_nodes_df.NN_dist < thresh]

    nodes_to_add, edges_to_add = [],[]

    for u, data in new_net.nodes(data = True):
        u = 'add_net_%s' % u
        nodes_to_add.append((u,data))

    for u,v, data in new_net.edges(data = True):
        u = 'add_net_%s' % u
        v = 'add_net_%s' % v
        edges_to_add.append((u,v,data))

    gdf_base = node_gdf_from_graph(base_net)
    gdf_base = gdf_base.set_index('node_ID')

    for index, row in join_nodes_df.iterrows():
        u = 'add_net_%s' % row.node_ID
        v = row.NN
        data = {}
        data['length'] = row.NN_dist / 1000
        data['infra_type'] = 'net_glue'
        data['Wkt'] = LineString([row.geometry, gdf_base.geometry.loc[v]])
        edges_to_add.append((u, v, data))
        edges_to_add.append((v, u, data))

    G_copy.add_nodes_from(nodes_to_add)
    G_copy.add_edges_from(edges_to_add)

    G_copy = nx.convert_node_labels_to_integers(G_copy)

    return G_copy

def clip(G, bound, source_crs = 'epsg:4326', target_crs = 'epsg:4326', geom_col = 'geometry', largest_G = True):
    """
    Removes any edges that fall beyond a polygon, and shortens any other edges that do so

    :param G: a graph object.
    :param bound: a shapely polygon object
    :param source_crs: crs object in format 'epsg:4326'
    :param target_crs: crs object in format 'epsg:4326'
    :param geom_col: label name for geometry object
    :param largest_G: if True, takes largest remaining subgraph of G as G
    """

    from shapely.geometry import MultiPolygon, Polygon

    edges_to_add, nodes_to_add = [],[]
    edges_to_remove, nodes_to_remove = [],[]

    if type(bound) == MultiPolygon or type(bound) == Polygon:
        pass
    else:
        raise ValueError('Bound input must be a Shapely Polygon or MultiPolygon object!')

    if type(G) != nx.classes.multidigraph.MultiDiGraph:
        raise ValueError('Graph object must be of type networkx.classes.multidigraph.MultiDiGraph!')

    # pyproj < 2.1
    # project_WGS_UTM = partial(
    #     pyproj.transform,
    #     pyproj.Proj(init=source_crs),
    #     pyproj.Proj(init=target_crs))

    # pyproj >= 2.1.0
    wgs84 = pyproj.CRS(source_crs)
    utm = pyproj.CRS(target_crs)
    project_WGS_UTM = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

    G_copy = G.copy()
    print('pre_clip | nodes: %s | edges: %s' % (G_copy.number_of_nodes(), G_copy.number_of_edges()))

    existing_legitimate_point_geometries = {}
    for u, data in G_copy.nodes(data = True):
        geo_point = Point(round(data['x'],10),round(data['y'],10))
        if bound.contains(geo_point):
            existing_legitimate_point_geometries[u] = geo_point
        else:
            nodes_to_remove.append(u)

    iterator = 0
    done_edges = []

    for u, v, data in G_copy.edges(data = True):

        done_edges.append((v,u))

        if (u,v) in done_edges:
            pass

        else:
            # define basics from data dictionary
            try:
                infra_type = data['infra_type']
            except:
                infra_type = data['highway']
            #extract the geometry of the geom_col, if there is no explicit geometry, load the wkt
            try:
                geom = data[geom_col]
            except:
                geom = loads(data['Wkt'])

            # road fully within country - do nothing
            if bound.contains(geom) == True:
                pass

            # road fully outside country - remove entirely
            elif bound.intersects(geom) == False:

                edges_to_remove.append((u, v))
                edges_to_remove.append((v, u))
                nodes_to_remove.append(u)
                nodes_to_remove.append(v)

            # road partially in, partially out
            else:

                # start by removing existing edges
                edges_to_remove.append((u, v))
                edges_to_remove.append((v, u))

                # identify the new line sections inside the boundary
                new_geom = bound.intersection(geom)
                if type(new_geom) == MultiLineString:
                    new_geom = linemerge(new_geom)

                # If there is only one:
                if type(new_geom) == LineString:

                    new_nodes, new_edges, new_node_dict_entries, iterator = new_edge_generator(new_geom,infra_type,iterator,existing_legitimate_point_geometries,geom_col,project_WGS_UTM)
                    existing_legitimate_point_geometries.update(new_node_dict_entries)
                    nodes_to_add.append(new_nodes)
                    edges_to_add.append(new_edges)

                elif type(new_geom) == MultiLineString:

                    for n in new_geom:
                        new_nodes, new_edges, new_node_dict_entries, iterator = new_edge_generator(n,infra_type,iterator,existing_legitimate_point_geometries,geom_col, project_WGS_UTM)
                        existing_legitimate_point_geometries.update(new_node_dict_entries)
                        nodes_to_add.append(new_nodes)
                        edges_to_add.append(new_edges)

    # Remove bad geometries
    G_copy.remove_nodes_from(nodes_to_remove)
    G_copy.remove_edges_from(edges_to_remove)

    # Add new geometries
    nodes_to_add = [item for sublist in nodes_to_add for item in sublist]
    edges_to_add = [item for sublist in edges_to_add for item in sublist]
    G_copy.add_nodes_from(nodes_to_add)
    G_copy.add_edges_from(edges_to_add)

    # Re-label nodes
    G_copy = nx.convert_node_labels_to_integers(G_copy)
    print('post_clip | nodes: %s | edges: %s' % (G_copy.number_of_nodes(), G_copy.number_of_edges()))

    # Select only largest remaining graph
    if largest_G == True:
        # compatible with NetworkX 2.4
        list_of_subgraphs = list(G_copy.subgraph(c).copy() for c in nx.strongly_connected_components(G_copy))
        max_graph = None
        max_edges = 0
        for i in list_of_subgraphs:
            if i.number_of_edges() > max_edges:
                max_edges = i.number_of_edges()
                max_graph = i
        # set your graph equal to the largest sub-graph
        G_copy = max_graph

    return G_copy

def new_edge_generator(passed_geom, infra_type, iterator, existing_legitimate_point_geometries, geom_col, project_WGS_UTM):
    """
    Generates new edge and node geometries based on a passed geometry. WARNING: This is a child process of clip(), and shouldn't be run on its own

    :param passed_geom: a shapely Linestring object
    :param infra_type: the road / highway class of the passed geometry
    :param iterator: helps count the new node IDs to keep unique nodes
    :param existing_legitimate_point_geometries: a dictionary of points already created / valid in [u:geom] format
    :param project_WGS_UTM: projection object to transform passed geometries
    :param geom_col: label name for geometry object
    """

    edges_to_add = []
    nodes_to_add = []

    # new start and end points will be start and end of line
    u_geo = passed_geom.coords[0]
    v_geo = passed_geom.coords[-1]
    u_geom, v_geom = Point(round(u_geo[0],10),round(u_geo[1],10)), Point(round(v_geo[0],10),round(v_geo[1],10))

    # check to see if geometry already exists. If yes, assign u and v node IDs accordingly
    # else, make a new u and v ID
    if u_geom in existing_legitimate_point_geometries.values():
        u = list(existing_legitimate_point_geometries.keys())[list(existing_legitimate_point_geometries.values()).index(u_geom)]

    else:
        u = 'new_node_%s' % iterator
        node_data = {}
        node_data['x'] = u_geom.x
        node_data['y'] = u_geom.y
        nodes_to_add.append((u,node_data))
        iterator += 1

    if v_geom in existing_legitimate_point_geometries.values():
        v = list(existing_legitimate_point_geometries.keys())[list(existing_legitimate_point_geometries.values()).index(v_geom)]

    else:
        v = 'new_node_%s' % iterator
        node_data = {}
        node_data['x'] = v_geom.x
        node_data['y'] = v_geom.y
        nodes_to_add.append((v,node_data))
        iterator += 1

    # update the data dicionary for the new geometry
    UTM_geom = transform(project_WGS_UTM, passed_geom)
    edge_data = {}
    edge_data[geom_col] = passed_geom
    edge_data['length'] = UTM_geom.length / 1000
    edge_data['infra_type'] = infra_type

    # assign new edges to network
    edges_to_add.append((u, v, edge_data))
    edges_to_add.append((v, u, edge_data))

    # new node dict entries - add newly created geometries to library of valid nodes
    new_node_dict_entries = []

    for u, data in nodes_to_add:
        new_node_dict_entries.append((u,Point(round(data['x'],10),round(data['y'],10))))

    return nodes_to_add, edges_to_add, new_node_dict_entries, iterator


def project_gdf(gdf, to_crs=None, to_latlong=False):
    """
    Taken from OSMNX

    Project a GeoDataFrame from its current CRS to another. If to_crs is None, project to the UTM CRS for the UTM zone in which the
    GeoDataFrame's centroid lies. Otherwise project to the CRS defined by
    to_crs. The simple UTM zone calculation in this function works well for
    most latitudes, but may not work for some extreme northern locations like
    Svalbard or far northern Norway.

    :param gdf: geopandas.GeoDataFrame the GeoDataFrame to be projected
    :param to_crs: string or pyproj.CRS if None, project to UTM zone in which gdf's centroid lies, otherwise project to this CRS
    :param to_latlong: bool if True, project to settings.default_crs and ignore to_crs
    :return: the projected GeoDataFrame
    """
    if gdf.crs is None or len(gdf) < 1:
        raise ValueError("GeoDataFrame must have a valid CRS and cannot be empty")

    # if to_latlong is True, project the gdf to latlong
    if to_latlong:
        gdf_proj = gdf.to_crs(settings.default_crs)
        #utils.log(f"Projected GeoDataFrame to {settings.default_crs}")

    # else if to_crs was passed-in, project gdf to this CRS
    elif to_crs is not None:
        gdf_proj = gdf.to_crs(to_crs)
        #utils.log(f"Projected GeoDataFrame to {to_crs}")

    # otherwise, automatically project the gdf to UTM
    else:
        if CRS.from_user_input(gdf.crs).is_projected:
            raise ValueError("Geometry must be unprojected to calculate UTM zone")

        # calculate longitude of centroid of union of all geometries in gdf
        avg_lng = gdf["geometry"].unary_union.centroid.x

        # calculate UTM zone from avg longitude to define CRS to project to
        utm_zone = math.floor((avg_lng + 180) / 6) + 1
        utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

        # project the GeoDataFrame to the UTM CRS
        gdf_proj = gdf.to_crs(utm_crs)
        #utils.log(f"Projected GeoDataFrame to {gdf_proj.crs}")

    return gdf_proj


def gn_project_graph(G, to_crs=None):
    """
    Taken from OSMNX. Project graph from its current CRS to another.
    If to_crs is None, project the graph to the UTM CRS for the UTM zone in
    which the graph's centroid lies. Otherwise, project the graph to the CRS
    defined by to_crs.

    :param G: networkx.MultiDiGraph the graph to be projected
    :param to_crs: string or pyproj.CRS if None, project graph to UTM zone in which graph centroid lies, otherwise project graph to this CRS
    :return: networkx.MultiDiGraph the projected graph
    """
    # STEP 1: PROJECT THE NODES
    gdf_nodes = ox.utils_graph.graph_to_gdfs(G, edges=False)

    # create new lat/lng columns to preserve lat/lng for later reference if
    # cols do not already exist (ie, don't overwrite in later re-projections)
    # if "lon" not in gdf_nodes.columns or "lat" not in gdf_nodes.columns:
    #     gdf_nodes["lon"] = gdf_nodes["x"]
    #     gdf_nodes["lat"] = gdf_nodes["y"]

    # project the nodes GeoDataFrame and extract the projected x/y values
    gdf_nodes_proj = project_gdf(gdf_nodes, to_crs=to_crs)
    gdf_nodes_proj["x"] = gdf_nodes_proj["geometry"].x
    gdf_nodes_proj["y"] = gdf_nodes_proj["geometry"].y
    gdf_nodes_proj = gdf_nodes_proj.drop(columns=["geometry"])

    # STEP 2: PROJECT THE EDGES
    gdf_edges_proj = ox.utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=False).drop(
            columns=["geometry"]
        )

    # STEP 3: REBUILD GRAPH
    # turn projected node/edge gdfs into a graph and update its CRS attribute
    G_proj = ox.utils_graph.graph_from_gdfs(gdf_nodes_proj, gdf_edges_proj, G.graph)
    #G_proj.graph["crs"] = gdf_nodes_proj.crs

    #utils.log(f"Projected graph with {len(G)} nodes and {len(G.edges)} edges")
    return G_proj


def reproject_graph(input_net, source_crs, target_crs):
    """
    to-do: delete, is not working

    Converts the node coordinates of a graph. Assumes that there are straight lines between the start and end nodes.

    :param input_net: a base network object (nx.MultiDiGraph)
    :param source_crs: The projection of the input_net (epsg code)
    :param target_crs: The projection input_net will be converted to (epsg code)
    """

    # pyproj < 2.1
    # project_WGS_UTM = partial(
    #             pyproj.transform,
    #             pyproj.Proj(init=source_crs),
    #             pyproj.Proj(init=target_crs))

    # pyproj >= 2.1.0
    wgs84 = pyproj.CRS(source_crs)
    utm = pyproj.CRS(target_crs)
    project_WGS_UTM = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

    i = list(input_net.nodes(data = True))

    for j in i:
        #print(j[1])
        #print(j[1]['x'])
        #print(transform(project_WGS_UTM,j[1]['geom']))
        j[1]['x'] = transform(project_WGS_UTM,j[1]['geom']).x
        j[1]['y'] = transform(project_WGS_UTM,j[1]['geom']).y
        #j[1]['geom'] = transform(project_WGS_UTM,j[1]['geom'])


    return input_net

def euclidean_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)

    :param lat1: lat1
    :param lon1: lon1
    :param lat2: lat2
    :param lon2: lon2

    """

    from math import radians, cos, sin, asin, sqrt

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

def utm_of_graph(G):
     
    # STEP 1: PROJECT THE NODES
    gdf_nodes = node_gdf_from_graph(G)
    
    # calculate longitude of centroid of union of all geometries in gdf
    avg_lng = gdf_nodes["geometry"].unary_union.centroid.x

    # calculate UTM zone from avg longitude to define CRS to project to
    utm_zone = math.floor((avg_lng + 180) / 6) + 1
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    return utm_crs

def advanced_snap(G, pois, u_tag = 'stnode', v_tag = 'endnode', node_key_col='osmid', edge_key_col='osmid', poi_key_col=None, road_col = 'highway', oneway_tag = 'oneway', path=None, threshold=500, knn=5, measure_crs='epsg:3857', factor = 1, verbose = False):
    """
    Connect and integrate a set of POIs into an existing road network.

    Given a road network in the form of two GeoDataFrames: nodes and edges,
    link each POI to the nearest edge (road segment) based on its projection
    point (PP) and generate a new integrated road network including the POIs,
    the projected points, and the connection edge.
    
    Credit for original code: Yuwen Chang, 2020-08-16

    1. Make sure all three input GeoDataFrames have defined crs attribute. Try something like `gdf.crs` or `gdf.crs = 'epsg:4326'`. They will then be converted into epsg:3857 or specified measure_crs for processing.

    :param pois (GeoDataFrame): a gdf of POI (geom: Point)
    :param nodes (GeoDataFrame): a gdf of road network nodes (geom: Point)
    :param edges (GeoDataFrame): a gdf of road network edges (geom: LineString)
    :param node_key_col (str): The node tag id in the returned graph
    :param edge_key_col (str): The edge tag id in the returned graph
    :param poi_key_col (str): a unique key column of pois should be provided, e.g., 'index', 'osmid', 'poi_number', etc. Currently, this will be renamed into 'osmid' in the output. [NOTE] For use in pandana, you may want to ensure this column is numeric-only to avoid processing errors. Preferably use unique integers (int or str) only, and be aware not to intersect with the node key, 'osmid' if you use OSM data, in the nodes gdf.
    :param poi_key_col (str): The tag to be used for oneway edges
    :param path (str): directory path to use for saving optional shapefiles (nodes and edges). Outputs will NOT be saved if this arg is not specified.
    :param threshold (int): the max length of a POI connection edge, POIs withconnection edge beyond this length will be removed. The unit is in meters as crs epsg is set to 3857 by default during processing.
    :param knn (int): k nearest neighbors to query for the nearest edge. Consider increasing this number up to 10 if the connection output is slightly unreasonable. But higher knn number will slow down the process.
    :param measure_crs (int):  preferred EPSG in meter units. Suggested to use the correct UTM projection.
    :param factor: allows you to scale up / down unit of returned new_footway_edges if other than meters. Set to 1000 if length in km.
    :return: G (graph): the original gdf with POIs and PPs appended and with connection edges appended and existing edges updated (if PPs are present)pois_meter (GeoDataFrame): gdf of the POIs along with extra columns, such as the associated nearest lines and PPs new_footway_edges (GeoDataFrame): gdf of the new footway edges that connect the POIs to the orginal graph
    """

    import rtree
    import itertools
    from shapely.ops import snap, split
    
    pd.options.mode.chained_assignment = None

    # check if POIs are not MultiPolygon
    if pois.geom_type.str.contains('MultiPoint').sum() > 0:
        raise ValueError("POIs must not be MultiPoint")

    # raise warning
    if (pois[poi_key_col].dtype != 'int64' and pois[poi_key_col].dtype != 'int'):
        print("POI keys are not ints, are you sure this is okay?")

    ## STAGE 0: initialization
    
    nodes = node_gdf_from_graph(G)
    nodes = nodes.rename(columns={'node_ID': node_key_col})

    # try:
    #     # convert node_key_col to int if needed
    #     nodes[node_key_col] = nodes[node_key_col].astype('int64')
    # except:
    #     print('error, node_key_col needs to be an int or convertible to an int')

    edges = edge_gdf_from_graph(G, oneway_tag = oneway_tag, single_edge=True)
    
    graph_crs = edges.crs

    start = time.time()

    # 0-2: configurations
    # set poi arguments
    node_highway_pp = 'projected_pap'  # POI Access Point
    node_highway_poi = 'poi'
    edge_highway = 'projected_footway'
    osmid_prefix = 9990000000

    # convert CRS
    pois_meter = pois.to_crs(measure_crs)
    nodes_meter = nodes.to_crs(measure_crs)
    edges_meter = edges.to_crs(measure_crs)

    # 0-1: helper functions
    
    # find nearest edge
    def find_kne(point, lines, near_idx):
        # getting the distances between the point and the lines
        dists = np.array(list(map(lambda l: l.distance(point), lines)))
        kne_pos = dists.argsort()[0]
        #kne = lines.iloc[[kne_pos]]
        

        #debugging
        #return lines, kne_pos

        #kne = lines[kne_pos]
        kne = lines.iloc[kne_pos]

        kne_idx = near_idx[kne_pos]

        

        #kne_idx = kne.index[0]
        #return kne_idx, kne.values[0]
        #return kne_pos, kne

        # return the index of the nearest edge, and the geometry of the nearest edge
        return kne_idx, kne

    def get_pp(point, line):
        """Get the projected point (pp) of 'point' on 'line'."""
        
        # project new Point to be interpolated
        pp = line.interpolate(line.project(point))  # PP as a Point

        # reduce precision
        #can't reduct it here because the split_function needs the point exactly on the line
        #pp = loads(dumps(pp, rounding_precision=3))

        return pp

    def split_line(line, pps):
        """Split 'line' by all intersecting 'pps' (as multipoint).

        Returns:
            new_lines (list): a list of all line segments after the split
        """
        # IMPORTANT FIX for ensuring intersection between splitters and the line
        # but no need for updating edges_meter manually because the old lines will be
        # replaced anyway
        # we want the tolerance to be really small, I changed it to a bigger tolerance of .5 meters and it caused 
        # the end of the line to snap to the PP therefore creating a gap
        line = snap(line, pps, 1e-4)  # slow?

        try:
            # with Shapely ver 2, Geometry objects are not iterable, so you need to use the geoms property
            new_lines = list(split(line, pps).geoms)  # split into segments
            return new_lines
        except TypeError as e:
            print('Error when splitting line: {}\n{}\n{}\n'.format(e, line, pps))
            return []


    def update_nodes(nodes, new_points, ptype, measure_crs='epsg:3857'):
        """Update nodes with a list (pp) or a GeoDataFrame (poi) of new_points.
        
        Args:
            ptype: type of Point list to append, 'pp' or 'poi'
        """
        nonlocal osmid_prefix

        # create gdf of new nodes (projected PAPs)
        if ptype == 'pp':
            new_nodes = gpd.GeoDataFrame(new_points, columns=['geometry'], crs=measure_crs)
            new_nodes[road_col] = node_highway_pp
            n = len(new_nodes)
            new_nodes[node_key_col] = [int(osmid_prefix + i) for i in range(n)]

        # create gdf of new nodes (original POIs)
        elif ptype == 'poi':
            new_nodes = new_points[['geometry', poi_key_col]]
            new_nodes.columns = ['geometry', node_key_col]
            new_nodes[road_col] = node_highway_poi

        else:
            print("Unknown ptype when updating nodes.")

        # merge new nodes (it is safe to ignore the index for nodes)
        gdfs = [nodes, new_nodes]
        #nodespd = pd.concat(gdfs, ignore_index=True, sort=False)
        nodes = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True, sort=False),
                                 crs=gdfs[0].crs)
        return nodes, new_nodes  # all nodes, newly added nodes only

    def update_edges(edges, new_lines, replace=True, nodes_meter=None, pois_meter=None):
        """
        Update edge info by adding new_lines; or,
        replace existing ones with new_lines (n-split segments).

        Args:
            replace: treat new_lines (flat list) as newly added edges if False,
                     else replace existing edges with new_lines (often a nested list)
        
        Note:
            kne_idx refers to 'fid in Rtree'/'label'/'loc', not positional iloc
        """
        
        # for interpolation (split by pp): replicate old line
        if replace:
            # create a flattened gdf with all line segs and corresponding kne_idx
            kne_idxs = list(line_pps_dict.keys())
            #print("print kne_idxs")
            #print(kne_idxs)
            # number of times each line is split
            lens = [len(item) for item in new_lines]
            #print("print lens")
            #print(lens)
            new_lines_gdf = gpd.GeoDataFrame(
                {'kne_idx': np.repeat(kne_idxs, lens),
                 'geometry': list(itertools.chain.from_iterable(new_lines))}, crs=measure_crs)
            # merge to inherit the data of the replaced line
            cols = list(edges.columns)
            cols.remove('geometry')  # don't include the old geometry
            new_edges = new_lines_gdf.merge(edges[cols], how='left', left_on='kne_idx', right_index=True)
            new_edges.drop('kne_idx', axis=1, inplace=True)

            #print('before')
            #print(new_edges['geometry'])

            # round nodes
            new_edges['geometry'] = new_edges.apply(lambda x: loads(dumps(x['geometry'], rounding_precision=3)), axis=1)
            
            print('after')
            print(new_edges['geometry'])

            new_lines = new_edges['geometry']  # now a flatten list
            
        # for connection (to external poi): append new lines
        else:
            new_edges = gpd.GeoDataFrame(pois[[poi_key_col]], geometry=new_lines, columns=[poi_key_col, 'geometry'], crs=measure_crs)
            # round nodes
            new_edges['geometry'] = new_edges.apply(lambda x: loads(dumps(x['geometry'], rounding_precision=3)), axis=1)
            new_edges[oneway_tag] = True
            new_edges[road_col] = edge_highway

        # https://stackoverflow.com/questions/61955960/shapely-linestring-length-units
        # update features (a bit slow)
        # length is only calculated and added to new lines
        new_edges['length'] = [l.length for l in new_lines]
        if factor > 1:
            new_edges['length'] = [l.length / factor for l in new_lines]
        # try to apply below to just new lines?
        if replace:
            new_edges[u_tag] = new_edges['geometry'].map(
                lambda x: nodes_id_dict.get(list(x.coords)[0], None))
        else:
            new_edges[u_tag] = new_edges[poi_key_col]

        new_edges[v_tag] = new_edges['geometry'].map(
            lambda x: nodes_id_dict.get(list(x.coords)[-1], None))

        print("debugging")
        # debugging 
        for x in new_edges['geometry']:
            node_id_match = nodes_id_dict.get(list(x.coords)[0], None)
            if node_id_match is None:
                print(f" node_id_match is None, coords are: {list(x.coords)[0]}")


        # try:
        #     # convert node_key_col to int if needed
        #     new_edges[u_tag] = new_edges[u_tag].astype('int64')
        # except:
        #     print('error, to nodes of new edges need to be an int or convertible to an int')

        # try:
        #     # convert node_key_col to int if needed
        #     new_edges[v_tag] = new_edges[v_tag].astype('int64')
        # except:
        #     print('error, from nodes of new edges need to be an int or convertible to an int')

        new_edges[edge_key_col] = ['_'.join(list(map(str, s))) for s in zip(new_edges[v_tag], new_edges[u_tag])]

        # remember to reindex to prevent duplication when concat
        start = edges.index[-1] + 1
        stop = start + len(new_edges)
        new_edges.index = range(start, stop)

        # for interpolation: remove existing edges
        if replace:
            edges = edges.drop(kne_idxs, axis=0)
        # for connection: filter invalid links
        else:
            unvalid_pos = np.where(new_edges['length'] > threshold)[0]
            # do not add new edges if they are longer than the threshold or if the length equals 0, if the length equals 0 that means the poi was overlaying an edge itself, therefore no extra edge needs to be created
            #unvalid_pos = np.where((new_edges['length'] > threshold) | (new_edges['length'] == 0))[0]
            unvalid_new_edges = new_edges.iloc[unvalid_pos]
            #print("print unvalid lines over threshold")
            #print(unvalid_new_edges)

            print(f"node count before: {nodes_meter.count()[0]}")
            nodes_meter = nodes_meter[~nodes_meter[node_key_col].isin(unvalid_new_edges.stnode)]
            print(f"node count after: {nodes_meter.count()[0]}")

            print(f"pois_meter count before: {pois_meter.count()[0]}")
            pois_meter = pois_meter[~pois_meter[poi_key_col].isin(unvalid_new_edges.stnode)]
            print(f"pois_meter count after: {pois_meter.count()[0]}")

            #valid_pos = np.where(new_edges['length'] <= threshold)[0]
            valid_pos = np.where((new_edges['length'] <= threshold) & (new_edges['length'] > 0))[0]
            n = len(new_edges)
            n_fault = n - len(valid_pos)
            f_pct = n_fault / n * 100
            print("Remove edge projections greater than threshold: {}/{} ({:.2f}%)".format(n_fault, n, f_pct))
            new_edges = new_edges.iloc[valid_pos]  # use 'iloc' here

        dfs = [edges, new_edges]
        edges = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=False, sort=False), crs=dfs[0].crs)

        if nodes_meter is not None:
            return edges, new_edges, nodes_meter, pois_meter
        else:
            # all edges, newly added edges only
            return edges, new_edges

    

    print("print edges_meter")
    #print(edges_meter)

    # build rtree
    print("Building rtree...")
    Rtree = rtree.index.Index()
    # items() now replaces iteritems() for a GeoSeries in GeoPandas
    [Rtree.insert(fid, geom.bounds) for fid, geom in edges_meter['geometry'].items()]

    if verbose == True:
        print("finished Building rtree")
        print('seconds elapsed: ' + str(time.time() - start))

    ## STAGE 1: interpolation
    # 1-1: update external nodes (pois)

    # print("print nodes_meter 2385764797 before")
    # print(nodes_meter.loc[nodes_meter.node_ID == 2385764797])
    
    print("updating external nodes (pois)")
    nodes_meter, _ = update_nodes(nodes_meter, pois_meter, ptype='poi', measure_crs=measure_crs)
    
    if verbose == True:
        print("finished updating external nodes (pois)")
        print('seconds elapsed: ' + str(time.time() - start))

    # print("print nodes_meter 2385764797 in between")
    # print(nodes_meter.loc[nodes_meter.node_ID == 2385764797])

    # 1-2: update internal nodes (interpolated pps)
    # locate nearest edge (kne) and projected point (pp)
    print("Projecting POIs to the network...2")
    #pois_meter['near_idx'] = [list(Rtree.nearest(point.bounds, knn))
                              #for point in pois_meter['geometry']]  # slow

    
                              
    #pois_meter['near_lines'] = [edges_meter['geometry'][near_idx]
                                #for near_idx in pois_meter['near_idx']]  # very slow


    def nearest_edge(row):
        near_idx = list(Rtree.nearest(row['geometry'].bounds, knn))
        near_lines = edges_meter['geometry'][near_idx]
        return near_idx, near_lines     
        
    #debug
    #return pois_meter, edges_meter

    # https://stackoverflow.com/questions/33802940/python-pandas-meaning-of-asterisk-sign-in-expression
    #pois_meter['near_idx'], pois_meter['near_lines'] = zip(*pois_meter.apply(nearest_edge, axis=1))

    pois_meter['near_idx'], pois_meter['near_lines'] = list(map(list, zip(*pois_meter.apply(nearest_edge, axis=1))))


    if verbose == True:
        print("finished pois_meter['near_idx'] and pois_meter['near_lines']")
        print('seconds elapsed: ' + str(time.time() - start))

    
                                
    pois_meter['kne_idx'], knes = zip(
        *[find_kne(point, near_lines, near_idx) for point, near_lines, near_idx in
          zip(pois_meter['geometry'], pois_meter['near_lines'], pois_meter['near_idx'])])  # slow

    if verbose == True:
        print("finished pois_meter['kne_idx']")
        print('seconds elapsed: ' + str(time.time() - start))

    # each POI point gets assigned a projected point
    print("assigning a projected point to each POI")
    pois_meter['pp'] = [get_pp(point, kne) for point, kne in zip(pois_meter['geometry'], knes)]


    if verbose == True:
        print("finished assigning a projected point to each POI")
        print('seconds elapsed: ' + str(time.time() - start))

    pp_column = pois_meter[['pp']]

    #print("print pp_column")
    #print(pp_column)

    # This below is an advanced method to take care of an edge case where a projected point is found near an endpoint

    #pp_column['coords'] = pp_column['pp'].map(lambda x: x.coords[0])
    # The coords of the projected point are rounded and stored in the 'coords' column while the original geometry is retained. The original geometry
    # will be needed for the split_line function as it needs high precision
    pp_column['coords'] = pp_column['pp'].map(lambda x: dumps(x, rounding_precision=3))

    # Get rid of any potential duplicates
    pp_column.drop_duplicates(inplace=True, subset="coords")

    # discard pp that have the same coordinate of an existing node
    #nodes_meter['coords'] = nodes_meter['geometry'].map(lambda x: x.coords[0])

    # reduce precision for the existing nodes in the graph and store them in the 'coords' column
    nodes_meter['coords'] = nodes_meter.apply(lambda x: dumps(x['geometry'], rounding_precision=3), axis=1)
    
    pp_column = pp_column.merge(nodes_meter['coords'], on='coords', how='left', indicator=True)
    pp_column = pp_column.query('_merge == "left_only"')

    # now the projected points that are very close to an existing endpoint in the graph will not be added
    # this will enable new projected edges to connected to existing endpoints in the graph instead of the newly created projected points
    pp_column = pp_column['pp']

    # update nodes
    print("Updating internal nodes...")
    nodes_meter, _new_nodes = update_nodes(nodes_meter, list(pp_column), ptype='pp', measure_crs=measure_crs)

    if verbose == True:
        print("finished Updating internal nodes")
        print('seconds elapsed: ' + str(time.time() - start))

    print("print _new_nodes")
    #print(_new_nodes)

    # doesn't work because dataframes don't match because there can be the same projected point for multiple pois
    #pois_meter["pp_id"] = _new_nodes[node_key_col]

    #print("nodes_meter")
    #print(nodes_meter)

    #return nodes_meter, _new_nodes

    
   
    nodes_coord = nodes_meter['geometry'].map(lambda x: x.coords[0])
    
    #print("print nodes_coord")
    #print(nodes_coord)
    
    #nodes_id_dict = dict(zip(nodes_coord, nodes_meter[node_key_col].astype('int64')))
    nodes_id_dict = dict(zip(nodes_coord, nodes_meter[node_key_col]))

    nodes_id_dict2 = {}
    for (x,y),v in nodes_id_dict.items():
        nodes_id_dict2[round(x,3),round(y,3)] = v

    nodes_id_dict = nodes_id_dict2

    #return nodes_id_dict

    # nodes_id_dict = {}
    # try:
    #     nodes_id_dict = dict(zip(nodes_coord, nodes_meter[node_key_col].astype('int64')))
    # except:
    #     print('error, nodes_id_dict nodes need to be an int or convertible to an int')


    # 1-3: update internal edges (split line segments)
    print("Updating internal edges...")
    # split
    # A nearest edge may have more than one projected point on it
    line_pps_dict = {k: MultiPoint(list(v)) for k, v in pois_meter.groupby(['kne_idx'])['pp']}

    #return nodes_id_dict, line_pps_dict, nodes_meter

    if verbose == True:
        print("finished creating line_pps_dict")
        print('seconds elapsed: ' + str(time.time() - start))


    print("creating new_lines")
    # new_lines becomes a list of lists
    # need to make sure that new line geometries's coordinate order match the stnode and endnode order
    new_lines = [split_line(edges_meter['geometry'][idx], pps) for idx, pps in line_pps_dict.items()]  # bit slow


    if verbose == True:
        print("finished creating new_lines")
        print('seconds elapsed: ' + str(time.time() - start))

    #return nodes_id_dict, new_lines, line_pps_dict, edges_meter, nodes_meter

    # print("edges_meter before") 
    # print(edges_meter.loc[edges_meter.endnode == 3874047473])

    
    # replacing existing lines with split lines
    print("Updating update_edges")
    edges_meter, _ = update_edges(edges_meter, new_lines, replace=True)
    #new_edges = update_edges(edges_meter, new_lines, replace=True)
    #return(new_edges)

    #return edges_meter, _

    if verbose == True:
        print("finished Updating update_edges")
        print('seconds elapsed: ' + str(time.time() - start))

    # print("edges_meter after")
    # print(edges_meter.loc[edges_meter.endnode == 3874047473])
    
    ## STAGE 2: connection
    # 2-1: update external edges (projected footways connected to pois)
    # establish new_edges
    print("Updating external links...")
    #pps_gdf = nodes_meter[nodes_meter['highway'] == node_highway_pp]
    #new_lines = [LineString([p1, p2]) for p1, p2 in zip(pois_meter['geometry'], pps_gdf['geometry'])]
    new_lines = [LineString([p1, p2]) for p1, p2 in zip(pois_meter['geometry'], pois_meter['pp'])]
    edges_meter, new_footway_edges, nodes_meter, pois_meter = update_edges(edges_meter, new_lines, replace=False, nodes_meter=nodes_meter, pois_meter=pois_meter)

    

    if verbose == True:
        print("finished Updating external links")
        print('seconds elapsed: ' + str(time.time() - start))

    # print("print nodes_meter")
    # print(nodes_meter)
    # print("print edges_meter")
    # print(edges_meter)

    ## STAGE 3: output
    # convert CRS
    nodes = nodes_meter.to_crs(epsg=4326)
    edges = edges_meter.to_crs(epsg=4326)

    # print("print nodes")
    # print(nodes)
    # print("print edges")
    # print(edges)
      
    # preprocess for pandana
    nodes.index = nodes[node_key_col]  # IMPORTANT
    nodes.index.name = None
    nodes['x'] = [p.x for p in nodes['geometry']]
    nodes['y'] = [p.y for p in nodes['geometry']]

    # edges.reset_index(drop=True, inplace=True)
    edges['length'] = edges['length'].astype(float)

    # report issues
    # - examine key duplication
    if len(nodes_meter) != len(nodes_id_dict):
        print("NOTE: duplication in node coordinates keys")
        print("Nodes count:", len(nodes_meter))
        print("Node coordinates key count:", len(nodes_id_dict))
    # - examine missing nodes
    print("Missing 'from' nodes:", edges[u_tag].isnull().sum())
    print("Missing 'to' nodes:", edges[v_tag].isnull().sum())
   
    # convert back to input graph CRS
    nodes = nodes.to_crs(graph_crs)
    edges = edges.to_crs(graph_crs)
    pois_meter = pois_meter.to_crs(graph_crs)

    new_footway_edges = new_footway_edges.to_crs(graph_crs)
       
    #print("print edges")
    #print(edges)
    
    #print("print nodes")
    #print(nodes)

    # Makes bi-directional graph from edges 
    print("making a new graph from edges and nodes")

    # not sure why, but there may be an edge case where en edge has the same to and from node and a length of 0
    edges = edges.loc[edges['length']>0]

    #return nodes, edges

    # now the edges_and_nodes_gdf_to_graph function has the ability to add reverse edges from a single-way GDF using the add_missing_reflected_edges flag. 
    # This is much faster than using the add_missing_reflected_edges after a graph is already created
    G = edges_and_nodes_gdf_to_graph(nodes, edges, node_tag = node_key_col, u_tag = u_tag, v_tag = v_tag, geometry_tag = 'geometry', discard_node_col=['coords'], add_missing_reflected_edges=True, oneway_tag=oneway_tag)
    #G = add_missing_reflected_edges(G, one_way_tag=oneway_tag)

    # set graph crs
    G.crs = graph_crs

    # save and return shapefile optional
    if path:
        nodes = nodes.drop(['coords'], axis=1)
        nodes.to_file(path+'/nodes.shp')
        if 'Wkt' in edges.columns:
            edges = edges.drop(['Wkt'], axis=1)
        edges.to_file(path+'/edges.shp')

    return G, pois_meter, new_footway_edges  # modified graph, snapped POIs, new edges


def add_intersection_delay(G, intersection_delay=7, time_col = 'time', highway_col='highway', filter=['projected_footway','motorway']):
    """
    Find node intersections. For all intersection nodes, if directed edge is going into the intersection then add delay to the edge.
    If the highest rank road at an intersection intersects a lower rank road, then the highest rank road does not get delayed. This assumes the highest rank road has the right-of-way.

    :param G: a base network object (nx.MultiDiGraph)
    :param intersection_delay: The number of seconds to delay travel time at intersections
    :filter: The filter is a list of highway values where the type of highway does not get an intersection delay.
    :returns: a base network object (nx.MultiDiGraph)
    """

    highway_rank = {
                'motorway': 1,
                'motorway_link': 1,
                'trunk': 1,
                'trunk_link': 1,
                'primary': 2,
                'primary_link': 2,
                'secondary': 3,
                'secondary_link':3,
                'tertiary': 4,
                'tertiary_link': 4,
                'unclassified': 5,
                'residential': 5,
                'track': 5
                }

    G_copy = G.copy()

    node_intersection_list = []

    for node in G.nodes:
        #print(G_reflected_time.degree(node))
        # if degree is greater than 2, then it is an intersection
        if G.degree(node) > 2:
            node_intersection_list.append(node)

    for intersection in node_intersection_list:
            
        pred_node_dict = {}
        for pred_node in G.predecessors(intersection):
            for edge in G[pred_node][intersection]:
                #print(pred_node, intersection)
                new_key = G_copy[pred_node][intersection][edge].get(highway_col)
                # it's possible that the highway can have more than one classification in a list
                if isinstance(new_key, list):
                    new_key = new_key[0]
                pred_node_dict[pred_node] = highway_rank.get(new_key)
                
        # update all 'None' values to 5
        pred_node_dict = {k:(5 if v==None else v) for k, v in pred_node_dict.items() }
        pred_node_dict = dict(sorted(pred_node_dict.items(), key=lambda item: item[1], reverse=False))
        #print(pred_node_dict)
        
        first_element_value = pred_node_dict[next(iter(pred_node_dict))]
        res = Counter(pred_node_dict.values())
        if res[first_element_value] <= 2:
            #print('skip')
            # remove all elements with same value
            pred_node_dict = {key:val for key, val in pred_node_dict.items() if val != first_element_value}
        else:
            pred_node_dict = pred_node_dict
        
        #print(f"print pred_node_dict again: {pred_node_dict}")
        for pred_node,value in pred_node_dict.items():
        #for pred_node in G.predecessors(intersection):
            #print(pred_node)
            #print(intersection)
            #print(G[pred_node][intersection])
            for edge in G[pred_node][intersection]:
                if G_copy[pred_node][intersection][edge].get(highway_col) not in filter:
                    G_copy[pred_node][intersection][edge][time_col] = G[pred_node][intersection][edge][time_col] + intersection_delay

    return G_copy
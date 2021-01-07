#the cleaning network part
import os, sys, time, importlib

import networkx as nx
import osmnx as ox

from shapely.ops import unary_union
from shapely.wkt import loads
from shapely.geometry import LineString, MultiLineString, Point

def eliminate_small_networks(G, method="Largest", min_size=100):
    """ Analyze G object to eliminate floating and small networks
    
    :param G:
        networkx object to be shrunk
    :param method:
        How to shrink network: Largest means to extract ONLY the largest network.
        Small means to remove all networks with edges smaller than param min_size    
    :param min_size:
        minimum network size to include in the network                   
    """
    if method == "Largest":
        list_of_subgraphs = list(nx.strongly_connected_component_subgraphs(G))
        max_graph = None
        max_edges = 0
        for i in list_of_subgraphs:
            if i.number_of_edges() > max_edges:
                max_edges = i.number_of_edges()
                max_graph = i
    elif method == "Small":
        raise(ValueError("Small not yet implemented"))
    else:
        raise(ValueError(f"{method} not ann accepted method. Did you mean Largest or Small?"))
    return(max_graph)


def CleanNetwork(G, wpath = '', country='', 
                UTM={'init': 'epsg:3857'}, WGS = {'init': 'epsg:4326'}, 
                junctdist = 50, verbose = False, geom_col = 'Wkt'):   
    """ Topologically simplifies an input graph object by collapsing junctions and removing interstital nodes
        :param G:
            Networkx graph object containing nodes and edges.
        :param wpath:
            the write path - a drive directory for inputs and output
        :param country:
            this parameter allows for the sequential processing of multiple countries
        :param UTM:
            {'init': 'epsg:3857'} the epsg code of the projection, in metres, to apply the junctdist
        :param junctdist:
            distance within which to collapse neighboring nodes. simplifies junctions. 
            Set to 0.1 if not simplification desired. 50m good for national (primary / secondary) networks
        :param verbose: 
            if True, saves down intermediate stages for dissection
        : returns :
            Networkx multidigraph
    """
    # Squeezes clusters of nodes down to a single node if they are within the snapping tolerance
    a = gn.simplify_junctions(G, UTM, WGS, junctdist)

    # ensures all streets are two-way
    a = gn.add_missing_reflected_edges(a)
    
    #save progress
    if verbose is True: 
        gn.save(a, 'a', wpath)
    
    # Finds and deletes interstital nodes based on node degree
    b = gn.custom_simplify(a)
    
    # rectify geometry
    for u, v, data in b.edges(data = True):
        if type(data[geom_col]) == list:
                data[geom_col] = gn.unbundle_geometry(data[geom_col])
    
    # save progress
    if verbose is True: 
        gn.save(b, 'b', wpath)
    
    # For some reason CustomSimplify doesn't return a MultiDiGraph. Fix that here
    c = gn.convert_to_MultiDiGraph(b)

    # This is the most controversial function - removes duplicated edges. This takes care of two-lane but separate highways, BUT
    # destroys internal loops within roads. Can be run with or without this line
    c = gn.remove_duplicate_edges(c)

    # Run this again after removing duplicated edges
    c = gn.custom_simplify(c)

    # Ensure all remaining edges are duplicated (two-way streets)
    c = gn.add_missing_reflected_edges(c)
    
    # save final
    if verbose:
        gn.save(c, '%s_processed' % country, wpath)
    
    print('Edge reduction: %s to %s (%d percent)' % (G.number_of_edges(), 
                                               c.number_of_edges(), 
                                               ((G.number_of_edges() - c.number_of_edges())/G.number_of_edges()*100)))
    return c

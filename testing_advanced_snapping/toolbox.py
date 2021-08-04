# Yuwen Chang
# 2020-08-16

import numpy as np
import pandas as pd
import geopandas as gpd

import rtree
import itertools

from shapely.geometry import MultiPoint, LineString
from shapely.ops import snap, split

# Get reference to GOSTNets
import sys
sys.path.append(r'C:\repos\GOSTnets')
import GOSTnets as gn

pd.options.mode.chained_assignment = None


def connect_poi(G, pois, u_tag = 'to', v_tag = 'from', node_key_col='osmid', poi_key_col=None, path=None, threshold=500, knn=5, measure_crs='epsg:3857'):
    """Connect and integrate a set of POIs into an existing road network.

    Given a road network in the form of two GeoDataFrames: nodes and edges,
    link each POI to the nearest edge (road segment) based on its projection
    point (PP) and generate a new integrated road network including the POIs,
    the projected points, and the connection edge.

    Args:
        pois (GeoDataFrame): a gdf of POI (geom: Point)
        nodes (GeoDataFrame): a gdf of road network nodes (geom: Point)
        edges (GeoDataFrame): a gdf of road network edges (geom: LineString)
        key_col (str): a unique key column of pois should be provided,
                       e.g., 'index', 'osmid', 'poi_number', etc.
                       Currently, this will be renamed into 'osmid' in the output.
                       [NOTE] For use in pandana, you may want to ensure this
                              column is numeric-only to avoid processing errors.
                              Preferably use unique integers (int or str) only,
                              and be aware not to intersect with the node key,
                              'osmid' if you use OSM data, in the nodes gdf.
        path (str): directory path to use for saving files (nodes and edges).
                      Outputs will NOT be saved if this arg is not specified.
        threshold (int): the max length of a POI connection edge, POIs with
                         connection edge beyond this length will be removed.
                         The unit is in meters as crs epsg is set to 3857 by
                         default during processing.
        knn (int): k nearest neighbors to query for the nearest edge.
                   Consider increasing this number up to 10 if the connection
                   output is slightly unreasonable. But higher knn number will
                   slow down the process.
        measure_crs (int): preferred EPSG in meter units. Suggested 3857 or 3395.

    Returns:
        nodes (GeoDataFrame): the original gdf with POIs and PPs appended
        edges (GeoDataFrame): the original gdf with connection edges appended
                              and existing edges updated (if PPs are present)

    Note:
        1. Make sure all three input GeoDataFrames have defined crs attribute.
           Try something like `gdf.crs` or `gdf.crs = 'epsg:4326'`.
           They will then be converted into epsg:3857 or specified measure_crs for processing.
    """

    ## STAGE 0: initialization
    
    nodes = gn.node_gdf_from_graph(G)
    edges = gn.edge_gdf_from_graph(G)
    
    graph_crs = edges.crs
    
    # 0-1: helper functions
    
    # find nearest edge
    def find_kne(point, lines):
        dists = np.array(list(map(lambda l: l.distance(point), lines)))
        kne_pos = dists.argsort()[0]
        kne = lines.iloc[[kne_pos]]
        kne_idx = kne.index[0]
        return kne_idx, kne.values[0]

    def get_pp(point, line):
        """Get the projected point (pp) of 'point' on 'line'."""
        # project new Point to be interpolated
        pp = line.interpolate(line.project(point))  # PP as a Point
        return pp

    def split_line(line, pps):
        """Split 'line' by all intersecting 'pps' (as multipoint).

        Returns:
            new_lines (list): a list of all line segments after the split
        """
        # IMPORTANT FIX for ensuring intersection between splitters and the line
        # but no need for updating edges_meter manually because the old lines will be
        # replaced anyway
        line = snap(line, pps, 1e-8)  # slow?

        try:
            new_lines = list(split(line, pps))  # split into segments
            return new_lines
        except TypeError as e:
            print('Error when splitting line: {}\n{}\n{}\n'.format(e, line, pps))
            return []

    def update_nodes(nodes, new_points, ptype, measure_crs='epsg:3857'):
        """Update nodes with a list (pp) or a GeoDataFrame (poi) of new_points.
        
        Args:
            ptype: type of Point list to append, 'pp' or 'poi'
        """
        # create gdf of new nodes (projected PAPs)
        if ptype == 'pp':
            new_nodes = gpd.GeoDataFrame(new_points, columns=['geometry'], crs=measure_crs)
            n = len(new_nodes)
            new_nodes['highway'] = node_highway_pp
            new_nodes[node_key_col] = [int(osmid_prefix + i) for i in range(n)]

        # create gdf of new nodes (original POIs)
        elif ptype == 'poi':
            new_nodes = new_points[['geometry', poi_key_col]]
            new_nodes.columns = ['geometry', node_key_col]
            new_nodes['highway'] = node_highway_poi
            #new_nodes[node_key_col] = new_nodes[node_key_col].astype(int)
            new_nodes[node_key_col] = new_nodes[node_key_col]

        else:
            print("Unknown ptype when updating nodes.")

        # merge new nodes (it is safe to ignore the index for nodes)
        gdfs = [nodes, new_nodes]
        nodespd = pd.concat(gdfs, ignore_index=True, sort=False)
        nodes = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True, sort=False),
                                 crs=gdfs[0].crs)
        return nodes, new_nodes  # all nodes, newly added nodes only

    def update_edges(edges, new_lines, replace):
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
            lens = [len(item) for item in new_lines]
            new_lines_gdf = gpd.GeoDataFrame(
                {'kne_idx': np.repeat(kne_idxs, lens),
                 'geometry': list(itertools.chain.from_iterable(new_lines))})
            # merge to inherit the data of the replaced line
            cols = list(edges.columns)
            cols.remove('geometry')  # don't include the old geometry
            new_edges = new_lines_gdf.merge(edges[cols], how='left', left_on='kne_idx', right_index=True)
            new_edges.drop('kne_idx', axis=1, inplace=True)
            new_lines = new_edges['geometry']  # now a flatten list
        # for connection (to external poi): append new lines
        else:
            new_edges = gpd.GeoDataFrame(pois[[poi_key_col]], geometry=new_lines, columns=[poi_key_col, 'geometry'])
            new_edges['oneway'] = False
            new_edges['highway'] = edge_highway

        # https://stackoverflow.com/questions/61955960/shapely-linestring-length-units
        # update features (a bit slow)
        # length is only calculated and added to new lines
        new_edges['length'] = [l.length for l in new_lines]
        
        new_edges[u_tag] = new_edges['geometry'].map(
            lambda x: nodes_id_dict.get(list(x.coords)[0], None))
        new_edges[v_tag] = new_edges['geometry'].map(
            lambda x: nodes_id_dict.get(list(x.coords)[-1], None))
        new_edges[node_key_col] = ['_'.join(list(map(str, s))) for s in zip(new_edges[v_tag], new_edges[u_tag])]

        # remember to reindex to prevent duplication when concat
        start = edges.index[-1] + 1
        stop = start + len(new_edges)
        new_edges.index = range(start, stop)

        # for interpolation: remove existing edges
        if replace:
            edges = edges.drop(kne_idxs, axis=0)
        # for connection: filter invalid links
        else:
            valid_pos = np.where(new_edges['length'] <= threshold)[0]
            n = len(new_edges)
            n_fault = n - len(valid_pos)
            f_pct = n_fault / n * 100
            print("Remove faulty projections: {}/{} ({:.2f}%)".format(n_fault, n, f_pct))
            new_edges = new_edges.iloc[valid_pos]  # use 'iloc' here

        # merge new edges
        dfs = [edges, new_edges]
        edges = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=False, sort=False), crs=dfs[0].crs)

        # all edges, newly added edges only
        return edges, new_edges

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

    # build rtree
    print("Building rtree...")
    Rtree = rtree.index.Index()
    [Rtree.insert(fid, geom.bounds) for fid, geom in edges_meter['geometry'].iteritems()]

    ## STAGE 1: interpolation
    # 1-1: update external nodes (pois)
    nodes_meter, _ = update_nodes(nodes_meter, pois_meter, ptype='poi', measure_crs=measure_crs)
    # 1-2: update internal nodes (interpolated pps)
    # locate nearest edge (kne) and projected point (pp)
    print("Projecting POIs to the network...")
    pois_meter['near_idx'] = [list(Rtree.nearest(point.bounds, knn))
                              for point in pois_meter['geometry']]  # slow
                              
    pois_meter['near_lines'] = [edges_meter['geometry'][near_idx]
                                for near_idx in pois_meter['near_idx']]  # very slow
                                
    pois_meter['kne_idx'], knes = zip(
        *[find_kne(point, near_lines) for point, near_lines in
          zip(pois_meter['geometry'], pois_meter['near_lines'])])  # slow

    pois_meter['pp'] = [get_pp(point, kne) for point, kne in zip(pois_meter['geometry'], knes)]
    # update nodes
    print("Updating internal nodes...")
    nodes_meter, _new_nodes = update_nodes(nodes_meter, list(pois_meter['pp']), ptype='pp', measure_crs=measure_crs)
       
    pois_meter[node_key_col] = _new_nodes[node_key_col]
    
    nodes_coord = nodes_meter['geometry'].map(lambda x: x.coords[0])
    
    #print("print nodes_coord")
    #print(nodes_coord)
    
    #nodes_id_dict = dict(zip(nodes_coord, nodes_meter[node_key_col].astype(int)))
    nodes_id_dict = dict(zip(nodes_coord, nodes_meter[node_key_col]))
    
    #print("print nodes_id_dict")
    #print(nodes_id_dict)
    
    # 1-3: update internal edges (split line segments)
    print("Updating internal edges...")
    # split
    # A nearest edge may have more than one projected point on it
    line_pps_dict = {k: MultiPoint(list(v)) for k, v in pois_meter.groupby(['kne_idx'])['pp']}
    # testing
    #print("line_pps_dict")
    #print(line_pps_dict)
    # new_lines becomes a list of lists
    new_lines = [split_line(edges_meter['geometry'][idx], pps) for idx, pps in line_pps_dict.items()]  # bit slow
    # testing
    #print("new_lines")
    #print(new_lines)
    edges_meter, _ = update_edges(edges_meter, new_lines, replace=True)

    ## STAGE 2: connection
    # 2-1: update external edges (projected footways connected to pois)
    # establish new_edges
    print("Updating external links...")
    pps_gdf = nodes_meter[nodes_meter['highway'] == node_highway_pp]
    new_lines = [LineString([p1, p2]) for p1, p2 in zip(pois_meter['geometry'], pps_gdf['geometry'])]
    edges_meter, _ = update_edges(edges_meter, new_lines, replace=False)

    ## STAGE 3: output
    # convert CRS
    nodes = nodes_meter.to_crs(epsg=4326)
    edges = edges_meter.to_crs(epsg=4326)
    
    # preprocess for pandana
    nodes.index = nodes[node_key_col]  # IMPORTANT
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
    print("Missing 'from' nodes:", len(edges[edges[u_tag] == None]))
    print("Missing 'to' nodes:", len(edges[edges[v_tag] == None]))
    
    # convert back to input graph CRS
    nodes = nodes.to_crs(graph_crs)
    edges = edges.to_crs(graph_crs)
    pois_meter = pois_meter.to_crs(graph_crs)

    # save and return shapefile optional
    if path:
        nodes.to_file(path+'/nodes.shp')
        edges.to_file(path+'/edges.shp')
        
    #print("print edges")
    #print(edges)
    
    #print("print nodes")
    #print(nodes)

    G = gn.edges_and_nodes_gdf_to_graph(nodes, edges, node_tag = node_key_col, u_tag = u_tag, v_tag = v_tag, geometry_tag = 'geometry')

    #return nodes, edges
    return G, pois_meter  # modified graph, snapped POIs

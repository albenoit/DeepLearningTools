import numpy as np
from cdlib import algorithms
from cdlib import TemporalClustering
import statistics
import networkx as nx
from itertools import combinations
from deeplearningtools.helpers.federated.clustering_evaluation import ClusteringEvaluator
from deeplearningtools.helpers.distance_metrics.utils import default_distances_to_similarities, get_clientinfo
from deeplearningtools.helpers.distance_metrics.metrics import ClientMatrixMetric, multimetric_client_distance_matrices

import matplotlib

matplotlib.use('Agg') # forbids display, all figures must be saved as image files

class Graph_builder:

    def __init__(self, sim_metric:ClientMatrixMetric, additional_metrics:set[ClientMatrixMetric]=None, hparams:dict=None):
        self.temporal_clustering = TemporalClustering()
        self.graph = nx.MultiGraph()
        self.cluster_evaluation = {}
        self.subgraphs = {}
        self.selected_subgraph = None
        self.base_communities = None

        self.select_sim_metric = sim_metric

        self.sim_metrics = additional_metrics or set()
        self.sim_metrics.add(self.select_sim_metric)

        #hparams
        self.hparams = hparams or dict()

    def get_edges(self):
        return self.graph.edges

    def get_nodes(self):
        return self.graph.nodes
    
    def get_sorted_nodes(self):
        def key_fn(node):
            return int(node)
        return sorted(list(self.graph.nodes), key=key_fn)

    def get_nodes_model_average(self, nodes_list:list=None, weights:list=None):
        target_nodes=nodes_list
        weighted_average=[]
        if nodes_list is None:
            target_nodes=self.graph.nodes
        #print("target nodes to average", target_nodes)
        # Process centroids
        nb_layers=len(target_nodes[0]['model'])
        for layer in range(nb_layers):
            weighted_average.append(np.average([target_nodes[id]['model'][layer] for id, node in enumerate(target_nodes)], axis=0, weights=weights))
        return weighted_average

    def select_subgraph(self, selected_edges: str):
        subgraph = self.subgraphs[selected_edges]
        self.selected_subgraph = subgraph

    def get_subgraph(self, selected_edges: str):
        if selected_edges in self.subgraphs:
            return self.subgraphs[selected_edges]
        return None

    def get_base_clusters(self):
        if self.base_communities is None:
            return []
        return self.base_communities.communities
    
    def get_evaluations(self):
        return self.cluster_evaluation

    def get_base_communities(self):
        return self.base_communities

    def get_nodes_property(self, property: str):
        return nx.get_node_attributes(self.graph, property)
    
    def get_client_from_cid(self, cid):
        """ Return the node id corresponding to the given client id
            :param cid: the client id to search for
            :returns: the node id corresponding to the given client id or None if not found
        """
        cids = nx.get_node_attributes(self.graph, "cid")
        return next((key for key, val in cids.items() if val == cid), None)

    ######################################   SOME UTILS   ###############################################

    def _update_similarities(self, last_round):
        similarities = []

        nodes = list(self.graph.nodes)
        node_indexes = {v:i for i,v in enumerate(nodes)}
        clients = [get_clientinfo(self.graph.nodes[node]) for node in nodes]

        distance_matrices = multimetric_client_distance_matrices(clients, self.sim_metrics, preprocessing_max_variants=3)
        similarity_matrix = distance_matrices[self.select_sim_metric.name]
        # if not self.select_sim_metric.is_similarity_score():
        #     similarity_matrix = self.distances_to_similarities(similarity_matrix)
        similarity_matrix = self.distances_to_similarities(similarity_matrix)

        # Update edge distances
        # do not go through this loop if we work with centroids (bias metrics unknown)
        for node_a, node_b in combinations(nodes, 2):
            ia = node_indexes[node_a]
            ib = node_indexes[node_b]

            distances = {k:m[ia][ib] for k,m in distance_matrices.items()}
            similarity = similarity_matrix[ia][ib]
            similarities.append(similarity)

            self.graph.add_edge(node_a, node_b, key='distances', last_round=last_round, distances=distances)
            self.graph.add_edge(node_a, node_b, key='similarity',
                                                            last_round=last_round,
                                                            update=True,
                                                            similarity=similarity)

        deciles=0
        median=0
        if len(similarities)>1:
            median = statistics.median(similarities)
            deciles = np.percentile(similarities, np.arange(0, 100, 10))
        #print('_update_similarities:: Deciles:', deciles, ' median:', median)
        return deciles, median

    def update_network(self, incoming_nodes, rnd, prune=False, scale=0.0):
        # update nodes
        print("\n\n\n\n\n\n-----------------------")
        print('update_network:number of incoming_nodes: ', len(incoming_nodes))
        for selected_node in incoming_nodes:
            if selected_node.id not in self.graph.nodes:
                #add node to the graph
                self.graph.add_node(selected_node.id, last_round=rnd, model=selected_node.data, cid=selected_node.cid)
            else:
                print('updating node ', selected_node.id)
                attr= {"last_round":rnd, "model": selected_node.data, "cid": selected_node.cid}

                node = self.graph.nodes[selected_node.id]
                if 'model' in node:
                    attr['gradient'] = [a-m for a,m in zip(attr['model'], node['model'])] # FIXME: Would be better with numpy array

                nx.set_node_attributes(self.graph, {selected_node.id: attr})

        deciles, median = self._update_similarities(rnd)     
        # filter edges zero distances as well as the ones below a pruning threshold
        threshold = 0
        if prune:
            threshold=deciles[scale]
        def edge_filter(edge):
            return edge['similarity'] <= threshold
        # remove edges
        # -> first scan edges to be removed
        edges_to_remove=[]
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            #print('check edge distance : ', edge, ' threshold : ', threshold)
            if key != 'similarity': continue # edge key
            if edge_filter(data):
                #print('-> removing edge ', edge[0], edge[1])
                edges_to_remove.append((u, v))
        # -> then remove them
        for edge in edges_to_remove:
            self.graph.remove_edge(edge[0], edge[1])

    def create_subgraph(self, selected_edges: str, rnd:int):
        subgraph = nx.Graph()
        # retrieve node data for this subgraph
        for node in self.graph.nodes():
            node_rnd = nx.get_node_attributes(self.graph, "last_round")[node]
            node_model = nx.get_node_attributes(self.graph, "model")[node]
            subgraph.add_node(node, last_round=node_rnd, model=node_model)

        labels = {}
        # add new edges
        for u, v, data in self.graph.edges(data=True):
            if selected_edges in data.keys(): # FIXME: works but is prone to error if a different keyed edge has selected_edges as a key in their data
                link_value = data[selected_edges] # Keep full precision, display precision managed later
                subgraph.add_edge(u, v, weight=link_value, last_round=rnd)
                labels[(u, v)] = '{:.2e}'.format(link_value)
        self.subgraphs[selected_edges] = subgraph
   
    # create cluster for the whole graph, not only the subgraph
    def apply_clustering(self, name: str, tabular_model=False, clusterAlgorithm:str=None, clusteringParams:dict=None):
        """ Compute the clustering of the subgraph

            :param name: the name of the subgraph
            :param tabular_model: the model format (sequential layers or keras model)
            :param clusterAlgorithm: Name of the clustering algorithm to use. This will take priority over hyperparameters.
            :param clusteringParams: Additional clustering parameters used by algorithms. These will take priority over hyperparameters.

            :returns: a graph describing identified clusters with their members and average model centroid
        """
        def get_cluster_param(key:str, default=None):
            if key == 'clusterAlgorithm': return clusterAlgorithm or self.hparams.get('clusterAlgorithm', default)
            if clusteringParams and key in clusteringParams: return clusteringParams[key]
            return self.hparams.get(key, default)

        subgraph = self.subgraphs[name]
        edge_weights = [e[2]['weight'] for e in subgraph.edges(data=True)]
        selected_algorithm = get_cluster_param('clusterAlgorithm', 'louvain')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~ Applying client clustering using algorithms :', selected_algorithm)
        # https://cdlib.readthedocs.io/en/0.2.0/reference/cd_algorithms/node_clustering.html

        match selected_algorithm:
            case 'louvain':
                # Small resolution = less cluster (opposite of theory)
                r = get_cluster_param('louvainResolution', 1)
                print('~~ -> using louvainResolution :', r)
                coms = algorithms.louvain(subgraph, resolution=r, randomize=False)
            case 'leiden':
                print("leiden")
                coms = algorithms.leiden(subgraph, weights=edge_weights)
            case 'surprise':
                print("surprise")
                coms = algorithms.surprise_communities(subgraph, weights=edge_weights)
            # case 'greedyModularity':
            #     coms = algorithms.greedy_modularity(subgraph, weight=edge_weights) # FIXME: weight attribute not matching its doc, not working
            case 'rbPots':
                # Higher res = more clusters (r>0)
                print("rbPots")
                r = get_cluster_param('rbPotsResolution', 1)
                print('~~ -> using rbPotsResolution :', r)
                coms = algorithms.rb_pots(subgraph, weights=edge_weights, resolution_parameter=r)
            case 'rberPots':
                print("rberPots")
                # Higher res = more clusters (r>0)
                r = get_cluster_param('rberPotsResolution', 1)
                coms = algorithms.rber_pots(subgraph, weights=edge_weights, resolution_parameter=r)

        self.base_communities = coms
        evaluator = ClusteringEvaluator(subgraph, coms, self.select_sim_metric)
        self.cluster_evaluation["size"] = evaluator.size()
        self.cluster_evaluation["avg_distance"] = evaluator.avg_distance() # doesnt seem to take the similarity value as a distance. avg is always 1 :-(
        self.cluster_evaluation["avg_embeddedness"] = evaluator.avg_embeddedness()
        self.cluster_evaluation["avg_intracluster_similarity"] = evaluator.avg_intracluster_similarity()
        self.cluster_evaluation["silhouette_score"] = evaluator.silhouette_score()
        clusters = coms.communities
        print("apply_clustering::clusters : ", clusters)
        # Process centroids
        new_cluster_graph=nx.Graph()
        for cluster_id, cluster_clients in enumerate(clusters):
            target_nodes=[self.graph.nodes[node] for node in cluster_clients]
            cluster_model_mean=self.get_nodes_model_average(nodes_list=target_nodes)
            #cluster_model_mean=np.mean(client_models, keepdims=True)#axis=0)
            new_cluster_graph.add_node(cluster_id, model=cluster_model_mean, members=cluster_clients)
        return new_cluster_graph

    def jaccard_score(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        u = len(set1.union(set2))
        i = len(set1.intersection(set2))
        return i / u

    # TODO sounds nice
    def track_communities(self, subgraph, rnd, name):
        """
        detect communities at current round and add to history
        arguments:
            subgraph, the graph to cluster
            rnd, the round index
        returns:
            nothing
        """
        # A. single step clustering relying on weighted edges, cluster overlap, (from https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algorithms.html)
        # danmf, dcs, dpclus, edmot, graph_entropy, ipca, lswl, lswl_plus, wCommunity
        round_communities = algorithms.ipca(subgraph)
        self.temporal_clustering.add_clustering(round_communities, rnd)
        print(self.temporal_clustering.community_matching(self.jaccard_score, False))
        # B. temporal clustering: tiles based on constrain propagation
        # round_communities = algorithms.louvain(subgraph)  # here any CDlib algorithm can be applied
        
        #dynamic graph:
        # #1: instant matching (should be applied offline, after the training process)
        #jaccard = lambda x, y:  len(set(x) & set(y)) / len(set(x) | set(y))
        #matches = self.temporal_clustering.community_matching(jaccard, two_sided=True)
        # #2: temporal matching  https://cdlib.readthedocs.io/en/latest/reference/cd_algorithms/algs/cdlib.algorithms.tiles.html#cdlib.algorithms.tiles

    def normalize(self, values, bounds):
        return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]
        
    def distance_normalization(self, dist, mini, maxi):
        return self.normalize([dist], {'actual':{'lower':mini,'upper':maxi},
                                                'desired':{'lower':1,'upper':10}})[0]

    def distances_to_similarities(self, distances:np.ndarray):
        return default_distances_to_similarities(distances)

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import matplotlib.figure
import numpy as np
import pandas as pd

import networkx as nx
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # forbids display, all figures must be saved as image files

# @BUGVERSION
# Si ca bug, j'ai du modifier les imports de Weights, parameters_to_weights, weights_to_parameters
# et les remplacer par parameters_to_ndarrays, Parameters, ndarrays_to_parameters
# car la version de flower a changé
from flwr.common import (FitIns, FitRes, EvaluateIns, EvaluateRes, Parameters,
                         Scalar, parameters_to_ndarrays,
                         ndarrays_to_parameters)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import deeplearningtools

from deeplearningtools.helpers.distance_metrics.utils import get_clientinfo
from deeplearningtools.helpers.distance_metrics.metrics import METRIC_SET, DEFAULT_METRIC, LayerMatrixMetric, ClientMetric, \
    write_pltinfo, build_layer_pltinfo, build_client_pltinfo
from deeplearningtools.helpers.distance_metrics.preprocessing import preprocess_models

from deeplearningtools.helpers.federated.graphcfl_visualisation import GraphCFL_Visualisation, LOGS_PATH
from deeplearningtools.helpers.federated.graphcfl_graph_builder import Graph_builder
from flwr.server.strategy.aggregate import weighted_loss_avg
import random
from math import comb
import pickle
from deeplearningtools.helpers.distance_metrics.utils import default_distances_to_similarities
#
# CLIENT FEDERATED LEARNING
#
class ClientParam:
    """ a Basic class to store client parameters"""
    def __init__(self, id: str, cid: str, data: List[np.ndarray], proxy: ClientProxy=None, fit_res: FitRes=None) -> None:
        self.id = id #reported in the eperiment settings file
         # client cid reported to flower server and considered exclusively on the configure_fid method
         # In simulation mode, cid=id=usersettings.hparams['procID'] while in real scenario, cid is an encoded value
        self.cid = cid
        self.data = data
        self.proxy = proxy
        self.fit_res = fit_res

##
# Hérite de la classe FedAvg (qui est une stratégie d'agrégation) pour clusteriser au moment de l'agrégation
# Une instance de cette classe est passée au server et est utilisée comme stratégie d'agrégation
##

class GraphCFL_strategy(FedAvg):
    """Configurable FlexCFL strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[Parameters], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        target_num_clusters=3,
        update_strategy: str ='delta_w',
        layer_similarity:bool=False,
        save_clients:bool=False,
        **kwargs
    ) -> None:
        super().__init__()

        #try to load experiment settings file (will raise on error since some hyperparameters are expected)
        self.working_directory=os.getcwd()
        self.usersettings, _= deeplearningtools.tools.experiment_settings.loadExperimentsSettings(filename=deeplearningtools.experiments_manager.SETTINGSFILE_COPY_NAME, 
                                                              call_from_session_folder=True)
        #print('Could read experiment settings')
        self.hparams=self.usersettings.hparams

        # Model saving
        self.save_clients = save_clients

        # Distance/similarity metric
        sim_name = kwargs.get('simMetric') or self.hparams.get('simMetric')
        self.sim_metric:ClientMetric = METRIC_SET.get_metric(sim_name) or DEFAULT_METRIC
        assert isinstance(self.sim_metric, ClientMetric), \
            (f"The selected distance metric need to be an instance of {ClientMetric.__name__} "
            'to enable "interscope" evaluation (client-cluster, client-global, cluster-global).')

        self.layer_similarity = layer_similarity

        self.multi_metric = kwargs.get('multi_metric')
        if self.multi_metric: 
            self.multi_metric = METRIC_SET.get_metrics(self.multi_metric)

        self.graphVisualisation = GraphCFL_Visualisation(self.sim_metric)
        self.graph_client = Graph_builder(self.sim_metric, self.multi_metric, hparams=self.hparams)
        self.graph_cluster = Graph_builder(self.sim_metric, self.multi_metric, hparams=self.hparams)

        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        
        # is None by default when using RL
        self.initial_parameters = initial_parameters
        
        parameters = None
        if initial_parameters != None:
            parameters = parameters_to_ndarrays(initial_parameters) #will store the global average model parameters
        self.global_average_model_params = parameters
        self.target_num_clusters = target_num_clusters
        self.update_strategy = update_strategy
        
        self.selected_clients = []
        self.cluster_evaluation = {}
        self.node_size = {}
        self.pruning_key = None
        self.subgraph_name = self.sim_metric.name # FIXME: Fix duplicate variables, name == selected_edges == self.subgraph_name

        # by default, the global model average is provided to the fitted clients
        # -> experiment settings file with hparam['fitconfig'] can override this method to apply a more complex fit config
        if 'fitConfig' in self.usersettings.hparams.keys():
            print('##############################################')
            print('configure_fit provides a specific model to each client according to hparam:', self.usersettings.hparams['fitConfig'])
            if self.usersettings.hparams['fitConfig']=='1nn':
                self.apply_fit_config = self.apply_base_fit_config_nearest_cluster
            elif self.usersettings.hparams['fitConfig']=='wnn':
                self.apply_fit_config = self.apply_base_fit_config_weighted_nearest_cluster
            else:
                print('##############################################')
                print('configure_fit provides global model to each client, as for classical FedAvg')
                self.apply_fit_config = self.apply_base_fit_config_global
        else:
            print('##############################################')
            print('configure_fit provides global model to each client, as for classical FedAvg')
            self.apply_fit_config = self.apply_base_fit_config_global

        # Make a new folder to store the logs
        os.makedirs(LOGS_PATH)
            
    def sample_clients(self, client_manager: ClientManager):
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        selected_clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        print('Configure_fit, sampled clients:', [client.cid for client in selected_clients])
        return selected_clients

    def apply_base_fit_config_global(self, server_round: int, clients: List[ClientProxy], config: Dict[str, Scalar]):
        """ apply a basic fit config to each selected client :
        -> send the global average model parameters to each new client
        Arguments:
        server_round: the server round index
        clients: the list of selected clients
        config: the configuration to be sent to each client
        Returns:
        a list of tuples (client, fit_ins) to be used by the server
        """
        print('apply_basse_fit_config_global : Fitting new clients with global model')
        client_fit_configs=[]
        for client in clients:
            fit_ins = FitIns(ndarrays_to_parameters(self.global_average_model_params), config)
            client_fit_configs.append((client, fit_ins))
        return client_fit_configs
    
    def apply_base_fit_config_nearest_cluster(self, server_round: int, clients: List[ClientProxy], config: Dict[str, Scalar]):
        """ apply a basic fit config to each selected client :
        -> send the global average model parameters to each new client
        -> OR send the closest known cluster model to each known client
        Arguments:
        server_round: the server round index
        clients: the list of selected clients
        config: the configuration to be sent to each client
        Returns:
        a list of tuples (client, fit_ins) to be used by the server
        """
        print('apply_base_fit_config_nearest_cluster : Fitting new clients with closest known cluster or global model')
        client_fit_configs=[]
        for client in clients:
            # get the client from the client id
            client_id=self.graph_client.get_client_from_cid(client.cid)
            if self.multi_metric is None and server_round >= 2 and client_id is not None:
                closest_clusters = self.graph_client.graph.nodes[client_id]['closest_clusters']
                closest_cluster=int(closest_clusters.iloc[0]['id'])
                print('Fitting again known client ', client_id, ' initiliazed with closest cluster model', closest_cluster)
                config['cluster_id'] = closest_cluster
                parameters=self.graph_cluster.graph.nodes[closest_cluster]['model']
            else:
                print('Fitting new client ', client.cid, ', initialized with global model')
                parameters = self.global_average_model_params
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)

            client_fit_configs.append((client, fit_ins))
        return client_fit_configs
    
    def apply_base_fit_config_weighted_nearest_cluster(self, server_round: int, clients: List[ClientProxy], config: Dict[str, Scalar]):
        """ apply a basic fit config to each selected client :
        -> send the global average model parameters to each new client
        -> OR send the weightedsum of the closest known cluster models to each known client
        Arguments:
        server_round: the server round index
        clients: the list of selected clients
        config: the configuration to be sent to each client
        Returns:
        a list of tuples (client, fit_ins) to be used by the server
        """
        print('apply_base_fit_config_weighted_nearest_cluster : Fitting new clients with weighted closest known cluster or global model')
        client_fit_configs=[]
        for client in clients:
            # get the client from the client id
            client_id=self.graph_client.get_client_from_cid(client.cid)
            if self.multi_metric is None and server_round > 2 and client_id is not None:
                closest_clusters = self.graph_client.graph.nodes[client_id]['closest_clusters']
                closest_cluster_ids=closest_clusters['id'].to_numpy()
                closest_clusters_models=[{'model':self.graph_cluster.graph.nodes[cluster]['model']} for cluster in closest_cluster_ids]
                """# weighting based on the inverse distance to the closest clusters
                closest_clusters_models_distances_inv=1.0/closest_clusters['dist'].to_numpy()
                cluster_weights=closest_clusters_models_distances_inv/np.sum(closest_clusters_models_distances_inv)
                print('**********************************************\nclosest_clusters[dist]', closest_clusters['dist'].to_numpy())
                print('Fitting nearest clusters ', closest_cluster_ids, ' at distances ', closest_clusters['dist'].to_numpy())
                print('RAW soft weighting:', cluster_weights)
                """
                # TODO rather use semi soft assignement to the closest clusters [Liu 2011]
                # also have a look at :https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec14_handout.pdf
                # and https://www.cs.cmu.edu/~02251/recitations/recitation_soft_clustering.pdf
                beta=1.0
                closest_clusters_models_distances_inv=np.exp(-beta*closest_clusters['dist'].to_numpy())
                # compute weights for the weighted sum of the closest clusters
                cluster_weights=closest_clusters_models_distances_inv/np.sum(closest_clusters_models_distances_inv)
                #print('semi soft weighting:', cluster_weights)
                config['cluster_id'] = closest_cluster_ids
                parameters=self.graph_cluster.get_nodes_model_average(closest_clusters_models, weights=cluster_weights)
            else:
                print('Fitting new client ', client.cid, ', initializing with global model')
                parameters = self.global_average_model_params

            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)

            client_fit_configs.append((client, fit_ins))
        return client_fit_configs
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training. Send the personalised model to each selected client"""

        # Sample clients
        clients= self.sample_clients(client_manager)

        # Prepare client/config pairs
        config = {'server_round':server_round, "cluster_id": "None"}
        
        
        # Apply the fit config to each client
        # -> self.apply_fit_config is set at class initialization
        # -> choose the appropriate method to apply the fit config self.apply_fit_config_xxx or build your own
        client_fit_configs = self.apply_fit_config(server_round, clients, config)

        # Return client/config pairs
        return client_fit_configs

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        # evaluation of the global model
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        # evluation of centroids

        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def update_client_graph(self, selected_clients: List[ClientParam], rnd:int, name: str)-> None:
        """
            update the whole client graph with the selected clients values of this round
            Arguments:
            selected_clients: the list of selected clients that have been updated
            rnd: the round index (helps report on last updated nodes)
            name: the name of the subgraph to be created/updated
            Returns nothing, only update self.graph_clients
        """
        self.graph_client.update_network(selected_clients, rnd)
        self.graph_client.create_subgraph(name, rnd)
        self.graph_client.select_subgraph(selected_edges=name)

    def update_cluster_graph(self, new_clients_clustering:nx.Graph, rnd:int, selected_edges: str) -> None:
        """
        Updates the cluster multigraph
        Arguments:
        rnd: the round index (helps report on last updated nodes)
        selected_edges: the edges name to be used to build a simplified graph (not multigraph)
        Returns nothing, only update self.graph_clients
        """
        #TODO: track_communities between previous cluster graph and the incoming one
        #self.graph_cluster.track_communities(new_clients_clustering, rnd, self.subgraph_name)

        # Measure the matching between the previous communities and the new ones
        """
        TODO
        evaluation.adjusted_rand_index(self.graph_cluster.get_base_communities(),
                                       new_clients_clustering)
        """
        ############# create a new graph
        self.graph_cluster = Graph_builder(self.sim_metric, self.multi_metric, hparams=self.hparams)        
        # force incoming graph to the self.graph_cluster.graph attribute
        self.graph_cluster.graph=nx.MultiGraph(new_clients_clustering)
        # add edges as the distance between cluster average models
        self.graph_cluster._update_similarities(rnd)
        # add round attribute to the nodes
        for cluster in self.graph_cluster.get_nodes():
            # add distance average measure and last_round property to each node
            attr= {cluster: {"last_round":rnd}}
            nx.set_node_attributes(self.graph_cluster.graph, attr)
        
        self.graph_cluster.create_subgraph(selected_edges, rnd)        

    #######################################################################################
    #######################################################################################
     
    def normalize(self, values, bounds):
        return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]
        
    def distance_normalization(self, dist, mini, maxi):
        return self.normalize([dist], {'actual':{'lower':mini,'upper':maxi},
                                                'desired':{'lower':1,'upper':10}})[0]

    def update_client_nearest_clusters(self, kneighbors:int=3):
        """ search for the k nearest clusters for each client
        and update the client graph accordingly
        Arguments:
        kneighbors: the number of closest clusters to keep
        Returns nothing, only update self.graph_clients
        """
        # updates edges bet
        for client in self.graph_client.get_nodes():
            client_info = get_clientinfo(self.graph_client.graph.nodes[client])
            cluster_distances = {'id': [], 'dist': []}
            #print('update_client_nearest_clusters, client:', client)
            if len(self.graph_cluster.get_nodes())>0:
                #print('Looking for nearest clusters for client : ', client.id)
                #print('Centroids : ', self.graph_cluster.graph.nodes)
                for cluster in self.graph_cluster.get_nodes():
                    #compute the distance between the client and the centroid of the cluster
                    cluster_info = get_clientinfo(self.graph_cluster.graph.nodes[cluster])
                    distance = self.sim_metric.client_distance(client_info, cluster_info)
                    cluster_distances['id'].append(cluster)
                    cluster_distances['dist'].append(distance)
                # sort the distances and keep the k closest ones
                sorted=pd.DataFrame.from_dict(cluster_distances).sort_values("dist")
                closest_clusters = sorted.head(kneighbors) 
                nx.set_node_attributes(self.graph_client.graph, {client: {"closest_clusters": closest_clusters}})
                #print('Closest clusters : ', closest_clusters)
            else:
                raise ValueError("_update_cluster_client_distances: No centroid to compare with")
            
        #Finally, measure the average distance of cluster members to the closest cluster centroid
        #-> each client within the client graph has its nearest clusters table to pick into
        for cluster in self.graph_cluster.get_nodes():
            dist_sum=0
            for member in self.graph_cluster.graph.nodes[cluster]['members']:
                dist_sum+=self.graph_client.graph.nodes[member]['closest_clusters'].iloc[0]['dist']
            # add distance average measure and last_round property to each node
            attr= {cluster: {'closestClusterAvgDist':dist_sum/len(self.graph_cluster.graph.nodes[cluster]['members'])}}
            nx.set_node_attributes(self.graph_cluster.graph, attr)
                

    def plot_client_cluster_graphs(self, rnd:int, selected_edges:str, plot_intra_cluster_edges=True)-> None:
        """ plot client and cluster graphs
            high level function to plot both client and cluster graphs
            Arguments:
            rnd: the round index (helps report on last updated nodes)
            selected_edges: the edges name to be used to build a simplified graph (not multigraph) and conduct both plotting and community/clustering management
            plot_intra_cluster_edges: if True, plot edges within the same cluster, if False, plot only inter-cluster edges

        """
        # FIXME: Change selected_edges use cases
        # Used to get similarity value between nodes + Used to title graphs

        try:
            # plot client graph
            self.plot_client_graph(rnd, selected_edges, plot_intra_cluster_edges)
            # plot cluster graph
            self.plot_cluster_graph(rnd, selected_edges)
            # save the graphs to images
            self.save_png_graphs(rnd, selected_edges=self.sim_metric.name)
            
            self.save_interclient_client_similarity_matrices(rnd)
            if self.layer_similarity:
                self.save_interclient_layer_similarity_matrices(rnd)
        except Exception as e:
            print('Exception in plot_client_cluster_graphs: could not plot graph ', e)
        
    def plot_client_graph(self, rnd:int, selected_edges:str, plot_intra_cluster_edges=True)-> None:
        """
        Updates the client multigraph by updating edges of known already declared/involved clients
        Arguments:
        rnd: the round index (helps report on last updated nodes)
        update_all: if False, only update edges with respect to provided client, if True, update all known edges (compute metric between newly updated and maybe nodes not involved for a long time)
        selected_edges: the edges name to be used to build a simplified graph (not multigraph) and conduct both plotting and community/clustering management
        plot_intra_cluster_edges: if True, plot edges within the same cluster, if False, plot only inter-cluster edges
        Returns nothing, only update self.graph_clients
        """

        ############# focus on a subgraph
        plotted_subgraph = self.graph_client.get_subgraph(selected_edges=selected_edges)
        communities=self.graph_client.get_base_clusters()
        if not(plot_intra_cluster_edges):
            #get graph for each community
            for community in communities:
                if len(community) >0:
                    community_graph = self.graph_client.graph.subgraph(community)
                    #remove the edges of this subgraph from plotted_subgraph
                    plotted_subgraph.remove_edges_from(community_graph.edges)

        ############# update the visualisation of the network
        self.graphVisualisation.update_client_graph(plotted_subgraph, rnd, self.graph_client.graph, communities, default_distances_to_similarities)
        
        ############# finally plot subgraph
        self.graphVisualisation.save_gml_graph(plotted_subgraph, communities, rnd, "clients", subplt=211)
        #self.graphVisualisation.save_graph("clients")

    def plot_cluster_graph(self, rnd:int, selected_edges:str)-> None:
        
        plotted_subgraph = self.graph_cluster.get_subgraph(selected_edges=selected_edges)

        cluster_evaluation = self.graph_client.get_evaluations()
        print('cluster_evaluation[avg_distance]', cluster_evaluation["avg_distance"])

        ############# update the size of the node
        # let's say each node is in it's own cluster
        # update the graph visualisation for the cluster graph
        clusters_list = [[cluster] for cluster in plotted_subgraph.nodes]
        closestClusterAvgDist=[self.graph_cluster.graph.nodes[cluster]['closestClusterAvgDist'] for cluster in self.graph_cluster.graph.nodes]
        self.graphVisualisation.update_cluster_graph(plotted_subgraph, rnd, self.graph_cluster.graph, closestClusterAvgDist)
        # retrieve node colors from the client graph
        node_colors = self.graphVisualisation.get_node_colors()
        # apply cluster colors from the client graph to the cluster graph
        self.graphVisualisation.update_colors(clusters_list, node_colors)

        ############# finally plot subgraph
        """clusters_visu = []
        for node in all_clusters:
            clusters_visu.append([node.id])
        """

        self.graphVisualisation.save_gml_graph(plotted_subgraph, clusters_list, rnd, "clusters", subplt=212)

    def save_png_graphs(self, rnd:int, selected_edges: str, name = "clusters", title=None):
        self.graphVisualisation.save_png_graph(selected_edges, rnd, name, title)

    def save_interclient_pltinfo(self, pltinfo:dict, metric_name:str, rnd:int, filename:str):
        path = os.path.join(os.getcwd(), "interclient_similarity", metric_name, f"round_{rnd}", "pltinfo", f"{filename}.pltinfo.pickle")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_pltinfo(pltinfo, path)

    def save_clientinfo(self, client_info:dict, rnd:int, subdir:str, name:str):
        save_path = os.path.join(os.getcwd(), 'saved_models', f'round_{rnd}', subdir, f"{name}.pickle")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pickle.dump(client_info, open(save_path, "wb"))
        
    def save_interclient_layer_similarity_matrices(self, rnd:int):
        nodes = self.graph_client.get_sorted_nodes()
        models = [self.graph_client.graph.nodes[node]['model'] for node in nodes] # Index-0 node = Index-0 model
        len_combi = comb(len(nodes), 2)

        for metric in self.similarity_metrics_iterator():
            if not isinstance(metric, LayerMatrixMetric): continue
            print(f"> [{metric.name}] Evaluating and saving {len_combi} layer similarity matrices (round {rnd})...")

            preprocessed_models = preprocess_models(models, metric.preprocessing_steps)

            for ia, ib in combinations(range(len(nodes)), 2):
                node_a = nodes[ia]
                node_b = nodes[ib]
                model1 = preprocessed_models[ia]
                model2 = preprocessed_models[ib]

                dst_matrix = metric._layer_distance_matrix(model1, model2)
                pltinfo = build_layer_pltinfo(dst_matrix, metric.name, f"{node_a}", f"{node_b}", rnd)
                self.save_interclient_pltinfo(pltinfo, metric.name, rnd, f'{node_a}-{node_b}')
        
        print(f"> Saved layer similarity matrices.")

    def save_interclient_client_similarity_matrices(self, rnd:int):
        nodes = self.graph_client.get_sorted_nodes()
        l_nodes = len(nodes)

        for metric in self.similarity_metrics_iterator():
            print(f"> [{metric.name}] Saving {l_nodes}x{l_nodes} interclient similarity matrix (round {rnd})...")
            dst_matrix = np.full((l_nodes, l_nodes), np.nan)
            for i in range(l_nodes):
                for j in range(i+1, l_nodes):
                    node_a, node_b = nodes[i], nodes[j]

                    dst_matrix[i,j] = dst_matrix[j,i] = \
                        self.graph_client.graph.edges[node_a, node_b, 'distances']['distances'][metric.name]

            pltinfo = build_client_pltinfo(dst_matrix, metric.name, list(map(str, nodes)), rnd)
            self.save_interclient_pltinfo(pltinfo, metric.name, rnd, "client_similarity")

        print("> Saved client similarity matrix.")

    def save_client_cluster_info(self, rnd:int):
        print("> Saving client and cluster information in PICKLE files...")

        def save_graph_models(graph:Graph_builder, subdir:str):
            for node in graph.get_nodes():
                client_info = get_clientinfo(graph.graph.nodes[node])
                client_info['model'] = np.asarray(client_info['model'], dtype=object)

                if 'gradient' in client_info:
                    client_info['gradient'] = np.asarray(client_info['gradient'], dtype=object)
                
                self.save_clientinfo(client_info, rnd, subdir, node)

        save_graph_models(self.graph_client, 'clients')
        save_graph_models(self.graph_cluster, 'clusters')

        # Global model
        global_model = np.asarray(self.global_average_model_params, dtype=object)
        global_info = {'model': global_model}
        self.save_clientinfo(global_info, rnd, "", "global_model")

        print("> Saved client and cluster information.")

    def similarity_metrics_iterator(self):
        yield self.sim_metric
        if self.multi_metric:
            for metric in self.multi_metric:
                if self.sim_metric != metric:
                    yield metric

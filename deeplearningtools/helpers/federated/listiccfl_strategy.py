from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (FitRes, Parameters,
                         Scalar, parameters_to_ndarrays,
                         ndarrays_to_parameters)
from flwr.server.client_proxy import ClientProxy

#base class to be used by custom strategies that will provide some generic tools
from deeplearningtools.helpers.federated.graphcfl_strategy import GraphCFL_strategy, ClientParam
from deeplearningtools.helpers.federated.clustering_evaluation import ClusteringEvaluator
import json
import os

# TODO
# -> détection de communauté
# ---> biais comme facteur de vérification de la contribution
# ---> qualité de l'apprentissage

# -> réduction biais
# ---> pondération des clients au sein des clusters
# ---> pondération des clusters sur agrégation modèle serveur



##
# Inherits from GraphCFL_strategy that relies on the standard FedAvg strategy and adds some graph and logging tools
##
class ListicCFL_strategy(GraphCFL_strategy):
    """Configurable customised FlexCFL strategy implementation."""

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
        target_num_clusters=5,
        update_strategy: str ='delta_w',
        k_nearest_clusters=3,
        layer_similarity:bool=False,
        save_clients:bool=False,
        **kwargs
    ) -> None:
        super().__init__(fraction_fit,
                         fraction_evaluate,
                         min_fit_clients,
                         min_evaluate_clients,
                         min_available_clients,
                         evaluate_fn,
                         on_fit_config_fn,
                         on_evaluate_config_fn,
                         accept_failures,
                         initial_parameters,
                         target_num_clusters,
                         update_strategy,
                         layer_similarity,
                         save_clients,
                         **kwargs)
        
        self.k_nearest_clusters=k_nearest_clusters

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], List[Parameters]]:
        """Aggregate fit results using weighted average."""

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        self.selected_clients = [
            ClientParam(
                        id=fit_res.metrics['client_id'],
                        cid=client_proxy.cid,
                        data=parameters_to_ndarrays(fit_res.parameters),
                        proxy=client_proxy,
                        fit_res=fit_res,
            )
            for client_proxy, fit_res in results
        ]

        ### Change the clustering strategy here
        self.update_client_graph(self.selected_clients, rnd, 'similarity')
        new_cluster_graph = self.graph_client.apply_clustering(name='similarity')
        print('new_cluster_graph: ', new_cluster_graph)

        ### TODO: Customize the clustering strategy here

        #-> here, naively apply the resulting client clustering to the server graph
        self.update_cluster_graph(new_clients_clustering=new_cluster_graph, rnd=rnd, selected_edges='similarity')

        # Finally get client-> closest cluster(s) distances
        # -> useful for plotting and configure_fit
        self.update_client_nearest_clusters(kneighbors=self.k_nearest_clusters)

        ########### Global model
        self.global_average_model_params=self.graph_cluster.get_nodes_model_average() 

        ########### update graph edges and plot
        plot_intra_cluster_edges=True
        if len(self.graph_client.get_nodes())>20:
            plot_intra_cluster_edges=False
        self.plot_client_cluster_graphs(rnd=rnd, selected_edges='similarity', plot_intra_cluster_edges=plot_intra_cluster_edges)

        # Return the global aggregated model for global information
        #-> each client and cluster models are kept in self.graph_client and self.graph_cluster

        if self.save_clients: self.save_client_cluster_info(rnd=rnd)

        return ndarrays_to_parameters(self.global_average_model_params), {}
    
    # @debug custome evaluation of the servers' model
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {"all_data_evaluation":True})

        if eval_res is None:
            return None
        loss, metrics = eval_res

        ############# evaluation of clusters' centroids on their "mirror" dataset
        #e.g., if a cluster is {1,2,3}, we should evaluate it on a test dataset with these labels only
        #try:
        eval_res_clusters = []
        if self.graph_client.get_base_clusters() != None: # and self.usersettings.hparams["minFit"] == self.usersettings.hparams["minCl"]:
            communities = self.graph_client.get_base_clusters()
            centroids=self.graph_cluster.graph.nodes
            # evluation of centroids: welcome to very long runtimes
            for i, centroid in enumerate(centroids):
                eval_res_clusters.append(self.evaluate_fn(server_round, self.graph_cluster.graph.nodes[i]['model'], {"clients": communities[i], "all_data_evaluation": False}))

        # include cluster evaluation in the metrics along with global model evaluation on the same clusters data
        # produce a json file with this structure:
        """
        {cluster_models: {server_round: {cluster_id: metrics}}} (cluster n is evaluated with cluster model n)
        {global_model: {server_round: {cluster_id: metrics}}}
        """
        
        metrics["cluster_models"] = {}
        metrics["cluster_models"][str(server_round)] = {}

        metrics["global_model"] = {}
        metrics["global_model"][str(server_round)] = {}
        for i, eval_cluster in enumerate(eval_res_clusters):
            print("\n\n\n\n\n", eval_cluster, "\n\n\n\n\n")
            metrics["cluster_models"][str(server_round)][i] = eval_cluster[1]
            communities = self.graph_client.get_base_clusters()
            metrics["global_model"][str(server_round)][i] = self.evaluate_fn(server_round, parameters_ndarrays, {"clients": communities[i], "all_data_evaluation": False})

        #except Exception as e:
        #    print("[listiccfl_strategy][evaluate()] Error when trying to evaluate cluster models!")
        #    print("Error : ", e)

        client_subgraph = self.graph_client.get_subgraph(selected_edges='similarity')
        cluster_subgraph = self.graph_cluster.get_subgraph(selected_edges='similarity')
        if client_subgraph is not None and cluster_subgraph is not None:
            communities = self.graph_client.get_base_communities()
            client_evaluator = ClusteringEvaluator(client_subgraph, communities, self.sim_metric)
            #cluster_coms = self.graph_cluster.get_base_communities()
            #cluster_evaluator = ClusteringEvaluator(cluster_subgraph, cluster_coms)
            metrics["clustering"] = {}
            intra = client_evaluator.avg_intracluster_similarity()
            inter = client_evaluator.avg_internode_similarity()
            metrics["clustering"]["avg_intracluster_similarity"] = intra
            metrics["clustering"]["avg_internode_similarity"] = inter
            metrics["clustering"]["intra_inter_similarity_ratio"] = client_evaluator.intra_inter_similarity_ratio(inter, intra)
            metrics["clustering"]["avg_embeddedness"] = client_evaluator.avg_embeddedness()
            metrics["clustering"]["silhouette_score"] = client_evaluator.silhouette_score()
            metrics["clustering"]["cifar100_purity"] = client_evaluator.cifar100_purity()
            metrics["clustering"]["cifar10_adjusted_rand_index"] = client_evaluator.cifar10_adjusted_rand_index()
            metrics["clustering"]["mnist_adjusted_rand_index"] = client_evaluator.mnist_adjusted_rand_index()
            try:
                # will not work if used outside must or jz
                metrics["clustering"]["cifar10_purity"] = client_evaluator.cifar10_purity()
            except:
                pass
            metrics["clustering"]["avg_distance"] = client_evaluator.avg_distance()
            metrics["clustering"]["size"] = client_evaluator.size()
        else:
            metrics["clustering"] = {}
            metrics["clustering"]["avg_intracluster_similarity"] = []
            metrics["clustering"]["intra_inter_similarity_ratio"] = []
            metrics["clustering"]["cifar10_adjusted_rand_index"] = []
            metrics["clustering"]["mnist_adjusted_rand_index"] = []
            metrics["clustering"]["avg_internode_similarity"] = []
            metrics["clustering"]["avg_embeddedness"] = []
            metrics["clustering"]["silhouette_score"] = []
            metrics["clustering"]["cifar100_purity"] = []
            metrics["clustering"]["cifar10_purity"] = []
            metrics["clustering"]["avg_distance"] = []
            metrics["clustering"]["size"] = []

        log_path = "metrics"
        filename = "metrics_" + str(server_round) + ".json"
        path = os.path.join(os.getcwd(),log_path)
        print('metrics: ', metrics)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(os.getcwd(),log_path, filename), 'w') as outfile:
            outfile.write(json.dumps(metrics, indent = 4))

        return loss, metrics

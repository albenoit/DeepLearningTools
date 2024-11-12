from .methods import cosine_distance, lp, procrustes_disparity, \
    deep_relative_trust

from .preprocessing import PreprocessingStep, preprocess_clients,\
      preprocess_models_solo, preprocess_clients_solo, preprocess_models,\
      PreprocessingTree, NoBiasStep, WeightBiasFusion2DStep

import numpy as np
import pickle
import copy

#region Metric base classes
def comparator_matrix(list1:list, list2:list, comparator_fn) -> np.ndarray:
    assert len(list1) == len(list2)
    l = len(list1)
    matrix = np.full((l, l), np.nan)
    for i in range(l):
        for j in range(i+1,l):
            matrix[i,j] = matrix[j,i] = comparator_fn(list1[i], list2[j])
    return matrix

def self_comparator_matrix(elements:list, comparator_fn) -> np.ndarray:
    return comparator_matrix(elements, elements, comparator_fn)

class ClientMatrixMetric:
    """
    Distance metric that can generate a client distance matrix.
    """
    def __init__(self, name:str) -> None:
        self.name = name
        self.preprocessing_steps = []

    def _client_distance_matrix(self, clients:list[dict]) -> np.ndarray: raise NotImplementedError()

    def client_distance_matrix(self, clients:list[dict], ignored_steps:list[PreprocessingStep]=None) -> np.ndarray:
        clients = preprocess_clients(clients, self.preprocessing_steps, ignored_steps)
        return self._client_distance_matrix(clients)


class ClientMetric(ClientMatrixMetric):
    """
    Distance metric that can evaluate the distance between two clients.

    As such, it can also compute :
    - Client distance matrix
    """
    def _client_distance(self, client1:dict, client2:dict) -> float: raise NotImplementedError()

    def _client_distance_matrix(self, clients: list[dict]) -> np.ndarray:
        return self_comparator_matrix(clients, self._client_distance)
    
    def client_distance(self, client1:dict, client2:dict, ignored_steps:list[PreprocessingStep]=None) -> float:
        client1, client2 = preprocess_clients_solo([client1, client2], self.preprocessing_steps, ignored_steps)
        return self._client_distance(client1, client2)


class ModelMatrixMetric(ClientMatrixMetric):
    """
    Distance metric that can generate a model distance matrix.

    As such, it can also compute :
    - Client distance matrix
    """
    def _model_distance_matrix(self, models:list[list[np.ndarray]]) -> np.ndarray: raise NotImplementedError()

    def _client_distance_matrix(self, clients: list[dict]) -> np.ndarray:
        return self._model_distance_matrix([c['model'] for c in clients])
    
    def model_distance_matrix(self, models:list[list[np.ndarray]], ignored_steps:list[PreprocessingStep]=None) -> np.ndarray:
        models = preprocess_models(models, self.preprocessing_steps, ignored_steps)
        return self._model_distance_matrix(models)


class ModelMetric(ModelMatrixMetric, ClientMetric):
    """
    Distance metric that can evaluate the distance between two models.

    As such, it can also compute :
    - Model distance matrix
    - Client distance
    - Client distance matrix
    """
    def _model_distance(self, model1:list[np.ndarray], model2:list[np.ndarray]) -> float: raise NotImplementedError()

    def _client_distance(self, client1: dict, client2: dict) -> float:
        return self._model_distance(client1['model'], client2['model'])
    def _model_distance_matrix(self, models: list[list[np.ndarray]]) -> np.ndarray:
        return self_comparator_matrix(models, self._model_distance)
    
    def model_distance(self, model1:list[np.ndarray], model2:list[np.ndarray], ignored_steps:list[PreprocessingStep]=None) -> float:
        model1, model2 = preprocess_models_solo([model1, model2], self.preprocessing_steps, ignored_steps)
        return self._model_distance(model1, model2)


class LayerMatrixMetric(ModelMetric):
    """
    Distance metric that can generate a distance matrix from two sets of model layers.

    As such, it can also compute :
    - Model distance
    - Model distance matrix
    - Client distance
    - Client distance matrix
    """
    AGGREGATIONS = {
        'sum': lambda matrix: np.nansum(matrix),
        'mean': lambda matrix: np.nanmean(matrix),
        'mean_squared': lambda matrix: np.nanmean(np.power(matrix, 2)),
        'mean_sqroot': lambda matrix: np.nanmean(np.sqrt(matrix)),
        'product': lambda matrix: np.nanprod(matrix),
    }

    def _layer_distance_matrix(self, model1:list[np.ndarray], model2:list[np.ndarray]) -> np.ndarray: raise NotImplementedError()
    def _aggregate_matrix(self, layer_dst_matrix:np.ndarray) -> float: raise NotImplementedError()

    def _model_distance(self, model1: list[np.ndarray], model2: list[np.ndarray]) -> float:
        return self._aggregate_matrix(self._layer_distance_matrix(model1, model2))
    
    def layer_distance_matrix(self, model1:list[np.ndarray], model2:list[np.ndarray], ignored_steps:list[PreprocessingStep]=None) -> np.ndarray:
        model1, model2 = preprocess_models_solo([model1, model2], self.preprocessing_steps, ignored_steps)
        return self._layer_distance_matrix(model1, model2)


class LayerMetric(LayerMatrixMetric):
    """
    Distance metric that can evaluate the distance between two layers.

    As such, it can also compute :
    - Layer distance matrix
    - Model distance
    - Model distance matrix
    - Client distance
    - Client distance matrix
    """
    def __init__(self, name: str, pairwise:bool=False) -> None:
        super().__init__(name)
        self.pairwise = pairwise

    def _layer_distance(self, layer1:np.ndarray, layer2:np.ndarray) -> float: raise NotImplementedError()
    def _layer_distance_matrix(self, model1: list[np.ndarray], model2: list[np.ndarray]) -> np.ndarray:
        if not self.pairwise:
            return comparator_matrix(model1, model2, self._layer_distance)
        
        dst_vector = [self._layer_distance(l1, l2) for l1, l2 in zip(model1, model2)]
        matrix = np.full((len(dst_vector), len(dst_vector)), np.nan)
        np.fill_diagonal(matrix, dst_vector)
        return matrix

def multimetric_client_distance_matrices(clients:list[dict], metrics:list[ClientMatrixMetric], preprocessing_max_variants:int=0, use_ray:bool=False) -> dict[str,np.ndarray]:
    """
    Computes a Client Distance Matrix for each metric, and returns them in a dictionary, with metric names as keys.

    ### Preprocessing
    See `PreprocessingTree`.
    
    Metrics have preprocessing steps, to prepare clients for comparaison. Some metrics can have common leading preprocessing steps.
    As such, the function evaluates the preprocessing chains (sequences of steps) with the most metrics, and uses them to preprocess clients.
    This prevents certain steps to be recalculated for each metric, at the cost of storing preprocessed copies of the clients.
    You can specify the maximum number of chains to keep with `preprocessing_max_variants`.
    
    ### Params
    :param clients: The clients to compare.
    :param metrics: The metrics to use.
    :param preprocessing_max_variants: Maximum number of preprocessed copies of the clients to store in memory. Set `-1` to maximally optimize.
    :param use_ray: Enables Ray parallelism, with each metric computation in a separate task.
    """
    if preprocessing_max_variants <= 0:
        jobs = [(m, clients, None) for m in metrics]
    else:
        tree_optimizer = PreprocessingTree(metrics, preprocessing_max_variants)
        jobs = tree_optimizer.get_preprocessed_jobs(clients) # It could be parallelised with Breadth-First Search

    res = {}
    def process_job(i:int):
        metric, clients, ignored_steps = jobs[i]
        res[metric.name] = metric.client_distance_matrix(clients, ignored_steps=ignored_steps)

    if use_ray:
        import ray
        ray.init(ignore_reinit_error=True)
        remote_fn = ray.remote(process_job)
        ray.get([remote_fn.remote(i) for i in range(len(jobs))])
    else:
        for i in range(len(jobs)): process_job(i)

    return res
#endregion
    
class MetricSet:
    """
    Collection of `ClientMatrixMetric` instances.
    """
    def __init__(self, *metrics:ClientMatrixMetric):
        self.__metrics:dict[str, ClientMatrixMetric] = {}
        for metric in metrics: self.add_metric(metric)

    def add_metric(self, metric:ClientMatrixMetric):
        self.__metrics[metric.name] = metric

    def add_metrics(self, metrics:list[ClientMatrixMetric]):
        for m in metrics: self.add_metric(m)

    def get_metric(self, name:str) -> ClientMatrixMetric | None:
        """
        Returns the metric with the given name.

        ### Metric aggregation
        Any aggregation key as a prefix will create a layer-and-model metric from the layer-only metric.

        Examples : `product-l1_dst`, `mean-l2_dst`

        ### Metric prefix flags
        You can add flags to any prefix by separating them by an '&'.
        - `diag` [aggregation prefixes] : Computes model similarity using the diagonal of the layer similarity matrix only.

        Examples :
        - `sum-l2_dst` : Sum of all layer L2 distances of the layer similarity matrix.
        - `sum&diag-l2_dst` : Sum of pairwise layer L2 distances.

        ### No bias comparison
        You can specify the prefix `nb` to remove bias layers from MODEL and CLIENT distance computations.

        Examples :
        - `nb-trusted_dst`: `trusted_dst` without bias layers
        - `nb-mean-cosine_dst`: `mean-cosine_dst` without bias layers
        - [BAD] `mean-nb-cosine_dst`: `cosine_dst` without bias layers, and meaned. `cosine_dst` is a layer-wise metric only, so `nb` won't work.
        """
        metric = self.__metrics.get(name)
        if metric: return metric
        elif not name: return None

        parts = name.split("-", maxsplit=1)
        if len(parts) < 2: return None

        head, subname = parts
        headname, *headflags = head.split("&")
        submetric = self.get_metric(subname) # Recursive

        if headname == 'nb':
            metric = copy.deepcopy(submetric)
            metric.name = name
            metric.preprocessing_steps.insert(0, NoBiasStep())

        elif headname == 'fb':
            metric = copy.deepcopy(submetric)
            metric.name = name
            metric.preprocessing_steps.insert(0, WeightBiasFusion2DStep())

        elif isinstance(submetric, LayerMetric) and headname in LayerMetric.AGGREGATIONS:
            metric = copy.deepcopy(submetric)
            metric.name = name
            if 'diag' in headflags: metric.pairwise = True
            metric._aggregate_matrix = LayerMetric.AGGREGATIONS[headname]
        
        else: return None

        self.add_metric(metric)
        return metric
    
    def get_metrics(self, names:list[str]) -> set[ClientMatrixMetric]:
        s = set(self.get_metric(n) for n in names)
        if None in s: s.remove(None)
        return s
    
    def __getitem__(self, key:str) -> ClientMatrixMetric:
        return self.__metrics[key]
    
    def remove_metric(self, name:str) -> bool:
        if name in self.__metrics:
            del self.__metrics[name]
            return True
        return False
    
    def get_all_metrics(self) -> list[ClientMatrixMetric]:
        return list(self.__metrics)
    
    def __str__(self) -> str:
        return f"[MetricSet] {len(self)} metrics\n" + '\n'.join(['- ' + str(m) for m in self.__metrics.values()])
    
    def __len__(self) -> int:
        return len(self.__metrics)

#region Metric instantiations
METRIC_SET = MetricSet()

def init_metricset():
    cosine_dst = LayerMetric("cosine_dst", pairwise=True)
    cosine_dst._layer_distance = cosine_distance

    model_cosine_dst = ModelMetric("model_cosine_dst")
    model_cosine_dst._model_distance = cosine_distance

    lp_dsts = []
    for p in [0.1, 0.5, 1, 2, 3, 10]:
        lp_dst = LayerMetric(f"l{p}_dst", pairwise=True)
        lp_dst._layer_distance = lambda l1, l2, p_arg=p: lp(p_arg, l1, l2)
        lp_dsts.append(lp_dst)

        model_lp_dst = ModelMetric(f"model_l{p}_dst")
        model_lp_dst._model_distance = lp_dst._layer_distance # Same function, different argument types
        lp_dsts.append(model_lp_dst)

    procrustes_dst = LayerMetric("procrustes_dst")
    procrustes_dst._layer_distance = procrustes_disparity
    procrustes_dst.preprocessing_steps.append(WeightBiasFusion2DStep()) # 2D layers : Fusion/Remove bias layers

    trusted_dst = ModelMetric("trusted_dst")
    trusted_dst._model_distance = lambda m1, m2: deep_relative_trust(m1, m2, return_drt_product=True)[0]
    
    log_trusted_dst = ModelMetric("log_trusted_dst")
    log_trusted_dst._model_distance = lambda m1, m2: np.log(trusted_dst._model_distance(m1, m2))

    METRIC_SET.add_metrics([cosine_dst, *lp_dsts, procrustes_dst, trusted_dst, log_trusted_dst, model_cosine_dst])

init_metricset()

DEFAULT_METRIC = METRIC_SET.get_metric("trusted_dst")
#endregion

#region Pltinfo I.O.
def build_layer_pltinfo(dst_matrix:np.ndarray, metric_name:str, model_a:str, model_b:str, round:int=-1) -> dict:
    return {
        'type': 'layer',
        'matrix': dst_matrix,
        'model_a': model_a,
        'model_b': model_b,
        'round': round,
        'metric_name': metric_name
    }

def build_model_pltinfo(dst_matrix:np.ndarray, metric_name:str, model_names:list[str], round:int=-1) -> dict:
    return {
        'type': 'model',
        'matrix': dst_matrix,
        'model_names': model_names,
        'round': round,
        'metric_name': metric_name
    }

def build_client_pltinfo(dst_matrix:np.ndarray, metric_name:str, client_names:list[str], round:int=-1) -> dict:
    return {
        'type': 'client',
        'matrix': dst_matrix,
        'client_names': client_names,
        'round': round,
        'metric_name': metric_name
    }

def read_pltinfo(filepath:str) -> dict:
    return pickle.load(open(filepath, "rb"))

def write_pltinfo(pltinfo:dict, filepath:str):
    pickle.dump(pltinfo, open(filepath, "wb"))

class UnknownPltinfoTypeException(Exception):
    """
    Raised when a PLTINFO object has an unknown type that cannot be handled.
    """
    def __init__(self, ptype:str) -> None:
        """
        Raised when a PLTINFO object has an unknown type that cannot be handled.

        :param ptype: The type of the PLTINFO object.
        """
        super().__init__(f"Unknown PLTINFO type : {ptype}")

#endregion

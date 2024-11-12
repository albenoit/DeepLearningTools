import numpy as np

#region Base classes
class PreprocessingStep:
    def preprocess_client(self, client:dict) -> dict: raise NotImplementedError()
    def preprocess_clients(self, clients:list[dict]) -> list[dict]: return list(map(self.preprocess_client, clients))
    def __eq__(self, value: object) -> bool:
        return type(self) == type(value)
    def __hash__(self) -> int:
        return hash(type(self))
    def __str__(self) -> str:
        return type(self).__name__


class ModelPreprocessingStep(PreprocessingStep):
    def preprocess_model(self, model:list[np.ndarray]) -> list[np.ndarray]: raise NotImplementedError()
    def preprocess_models(self, models:list[list[np.ndarray]]) -> list[list[np.ndarray]]: return list(map(self.preprocess_model, models))
    def preprocess_client(self, client: dict) -> dict:
        new_client = client.copy()
        new_client['model'] = self.preprocess_model(new_client['model'])
        if 'gradient' in new_client:
            new_client['gradient'] = self.preprocess_model(new_client['gradient'])
        return new_client
    

class PreprocessingTree:
    """
    Joins common metric preprocessing steps to reduce preprocessing time, at the cost of storing copies of given clients.
    
    Class made to support potential heavy metric preprocessing in the future.
    """
    def __init__(self, metrics:list, max_nodes:int=-1):
        """
        :param metrics: The metrics to use as a basis for the graph.
        :param max_nodes: The maximum number of variants to compute and store. Set `-1` to maximally optimize.
        """
        self.__metrics = metrics
        self.__graph = self.__new_node()
        autodetect = max_nodes < 0

        # Build graph of common preprocessing paths
        for metric in metrics:
            node = self.__graph
            for step in metric.preprocessing_steps:
                node['count'] += 1
                if step not in node['next']:
                    node['next'][step] = self.__new_node(back=node, step=step)
                node = node['next'][step]

            node['count'] += 1

        # Find maximum count nodes
        self.__variants = []
        stack = [self.__graph]
        while stack:
            node = stack.pop()
            candidate = len(node['next']) != 1 or node['count'] > next(iter(node['next'].values()))['count']

            if candidate and (not autodetect or node['count'] > 1):
                self.__variants.append(node)
                if not autodetect and len(self.__variants) > max_nodes+1:
                    self.__variants.remove(min(self.__variants, key=lambda r: r['count']))

            stack.extend(node['next'].values())

        for node in self.__variants: node['variant'] = True

    def __new_node(self, back=None, step=None):
        return {'step':step, 'count':0, 'back':back, 'next':{}, 'variant':False}
    
    def get_preprocessed_jobs(self, clients:list[dict]) -> list[tuple]:
        # Preprocess clients
        self.__graph['clients'] = clients
        final_steps = {}
        for var_root in sorted(self.__variants, key=lambda r: r['count'], reverse=True):
            steps = []
            clients = var_root.get('clients')
            node = var_root

            while clients is None:
                steps.append(node['step'])
                node = node['back']
                clients = node.get('clients')
            steps = steps[::-1]
            final_steps[id(var_root)] = steps

            for step in steps: clients = step.preprocess_clients(clients)
            var_root['clients'] = clients

        # Build metric:clients dictionary
        res = []
        for metric in self.__metrics:
            root = self.__graph
            for step in metric.preprocessing_steps:
                if (_next:=root['next'].get(step)) is None or not _next['variant']: break
                root = _next
            res.append((metric, root['clients'], final_steps[id(root)]))
        return res
    
    def __str__(self) -> str:
        lines = []
        def str_node(node, start_str=""):
            lines.append(f"{start_str}{str(node['step'])} {node['count']}{(' (Variant)' if node['variant'] else '')}")
            for subnode in node['next'].values():
                str_node(subnode, start_str+"|--")
        str_node(self.__graph)
        return '\n'.join(lines)
        
#endregion

#region Preprocessing utils
def filter_ignored_steps(preprocessing_steps:list[PreprocessingStep], ignored_steps:list[PreprocessingStep]|None) -> list[PreprocessingStep]:
    if ignored_steps is None: return preprocessing_steps
    i = 0
    for step1, step2 in zip(preprocessing_steps, ignored_steps):
        if step1 != step2: break
        i += 1
    return preprocessing_steps[i:]

def preprocess_clients(clients:list[dict], preprocessing_steps:list[PreprocessingStep], ignored_steps:list[PreprocessingStep]=None) -> list[dict]:
    for step in filter_ignored_steps(preprocessing_steps, ignored_steps): clients = step.preprocess_clients(clients)
    return clients

def preprocess_clients_solo(clients:list[dict], preprocessing_steps:list[PreprocessingStep], ignored_steps:list[PreprocessingStep]=None) -> list[dict]:
    for step in filter_ignored_steps(preprocessing_steps, ignored_steps): clients = [step.preprocess_client(c) for c in clients]
    return clients

def preprocess_models(models:list[dict], preprocessing_steps:list[PreprocessingStep], ignored_steps:list[PreprocessingStep]=None) -> list[dict]:
    for step in filter_ignored_steps(preprocessing_steps, ignored_steps):
        if not isinstance(step, ModelPreprocessingStep): continue
        models = step.preprocess_models(models)
    return models

def preprocess_models_solo(models:list[dict], preprocessing_steps:list[PreprocessingStep], ignored_steps:list[PreprocessingStep]=None) -> list[dict]:
    for step in filter_ignored_steps(preprocessing_steps, ignored_steps):
        if not isinstance(step, ModelPreprocessingStep): continue
        models = [step.preprocess_model(m) for m in models]
    return models
#endregion

#region Preprocessing steps classes
def _next_layer_is_bias(model:list[np.ndarray], i:int):
    return i+1 < len(model) and len(model[i+1].shape) == 1 and model[i].shape[-1] == model[i+1].shape[0]

class NoBiasStep(ModelPreprocessingStep):
    """
    Creates a shallow copy of the model, with weight layers only.
    """
    def preprocess_model(self, model: list[np.ndarray]) -> list[np.ndarray]:
        weight_indices = []
        i = 0
        while i < len(model):
            weight_indices.append(i)
            if _next_layer_is_bias(model, i): i+=1
            i+=1
        return [model[w] for w in weight_indices]
    

class WeightBiasFusion2DStep(ModelPreprocessingStep):
    """
    Creates a deep copy of the model, flattening weight layers to 2D, and concatenating weight (2D) and bias (1D) layers.
    """
    def preprocess_model(self, model: list[np.ndarray]) -> list[np.ndarray]:
        new_model = []
        i = 0
        while i < len(model):
            layer = model[i].copy()
            if len(layer.shape) > 2: layer = np.resize(layer, (np.prod(layer.shape[:-1]), layer.shape[-1]))
            if _next_layer_is_bias(model, i):
                layer = np.insert(layer, 0, model[i+1], axis=0)
                i+=1
            new_model.append(layer)
            i+=1
        return new_model
    
    
class Flatten2DStep(ModelPreprocessingStep):
    """
    Creates a deep copy of the model, flattening weight layers to 2D, and removing 1D layers (including bias layers).
    """
    def preprocess_model(self, model: list[np.ndarray]) -> list[np.ndarray]:
        new_model = []
        for layer in model:
            if len(layer.shape) > 2: layer = np.resize(layer, (np.prod(layer.shape[:-1]), layer.shape[-1]))
            elif len(layer.shape) == 2: layer = layer.copy()
            else: continue
            new_model.append(layer)
        return new_model
#endregion

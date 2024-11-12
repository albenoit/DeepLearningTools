from .preprocessing import *
from .metrics import ClientMetric

def test_preprocessingtree_variants():
    def create_metric(i):
        metric = ClientMetric(str(i))
        metric._client_distance = lambda c,d: 1
        return metric
    
    def create_model():
        model = []
        for _ in range(3):
            model.append(np.random.random((10,8)))
            model.append(np.random.random((8)))
        return model
    
    clients = [{'model': create_model()} for _ in range(5)]
    metric1 = create_metric(1)
    metric1.preprocessing_steps += [NoBiasStep(), WeightBiasFusion2DStep()]

    metric2 = create_metric(2)
    metric2.preprocessing_steps += [NoBiasStep(), Flatten2DStep()]

    metric3 = create_metric(3)
    metric3.preprocessing_steps += [NoBiasStep(), WeightBiasFusion2DStep(), Flatten2DStep()]

    metric4 = create_metric(4)

    metrics = [metric1, metric2, metric3, metric4]
    tree = PreprocessingTree(metrics, -1)
    print(tree)

if __name__ == "__main__":
    test_preprocessingtree_variants()
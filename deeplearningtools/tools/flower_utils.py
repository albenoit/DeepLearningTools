### codes inspired from repository the Flwer repository: https://github.com/adap/flower 
"""Contains utility functions for CNN FL on MNIST."""


from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from flwr.common import Metrics
from flwr.server.history import History

def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    print('Reporting on the available metrics', metric_dict.keys())
    for key in metric_dict.keys():

        rounds, values = zip(*metric_dict[key])

        if isinstance(values[0], dict):
            print('Flower_utils.plot_metric_from_history, values is dict, skipping')
            continue
        fig = plt.figure()
        axis = fig.add_subplot(111)
        plt.plot(np.asarray(rounds), np.asarray(values), label=key +r" other federated learning rounds")

        plt.title(f"{metric_type.capitalize()} Validation - MNIST")
        plt.xlabel("Rounds")
        plt.ylabel(key.capitalize())
        plt.legend(loc="lower right")

        # Set the apect ratio to 1.0
        xleft, xright = axis.get_xlim()
        ybottom, ytop = axis.get_ylim()
        axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)

        plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_{key}_metrics{suffix}.png"))
        plt.close()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}


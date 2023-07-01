
# This is a template for a custom strategy.
# basically, you can customise the following functions:
# - configure_fit: customise the next round configuration (client selection, their parameters, ...)
# - aggregate_fit: customise the model aggregate function
# - evaluate: customise the (centralised) model evaluation function
# please look at the official documentation for more details: https://flower.dev/docs/quickstart_federated_learning.html#custom-strategies

# please follow the following naming convention for the strategy filename and class name: 
# <strategy_name>_strategy.py with lower case for the file name and <Strategy_Name>_strategy (upper case) for the class name
# place <strategy_name>_strategy.py in folder deeplearningtools/helpers/federated/
# Then, when time comes to make use of it, in the experiment settings file, in the hparams dictionnaty, just add the following key-value pair:
# 'federated': 'Template_strategy', # the value follows calss name convention, i.e. <Strategy_Name>_strategy

import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple
from flwr.common import Scalar, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common import (FitIns, FitRes, Parameters,
                         Scalar, parameters_to_ndarrays, Parameters,
                         ndarrays_to_parameters)

class Template_strategy(FedAvg):
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
        num_clusters=3,
        update_strategy: str ='delta_w'
    ) -> None:
        ''' maybe add some parameters here '''
        super().__init__()

        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """maybe customise the next round configuration (client selection, their parameters, ...))"""
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """ maybe customise the model aggregate function"""
        return super().aggregate_fit(rnd, results, failures)
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """ maybe customise the (centralised) model evaluation function."""
        return super().evaluate(server_round, parameters)

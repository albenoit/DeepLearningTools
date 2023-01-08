# federated erver should be started from the main script:  start_federated_server.py
# main parameter server, should be started first

import flwr as fl
fl.server.start_server(config={"num_rounds": 30})


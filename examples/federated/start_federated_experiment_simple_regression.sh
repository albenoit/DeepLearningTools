#!/bin/bash

# supposing that the data is ready (localiztion ok, maybe a kafka server is started)
# run all the commands to conduct a single experiment
echo "WARNING, PLEASE OPEN THIS SCRIPT AND ADJUST PATH TO THE APPROPRIATE CONTAINER..."
#echo "starting federated parameter server..." 
nohup singularity run install/tf2_addons.2.9.1.sif start_federated_server.py --usersettings examples/federated/mysettings_curve_fitting.py &
sleep 5

for id in {-5..0}
do
    nohup singularity run  install/tf2_addons.2.9.1.sif experiments_manager.py --procID $id  --usersettings examples/federated/mysettings_curve_fitting.py &
done
echo "some first clients are working..." 
sleep 300

echo "adding new clients working on other data intervals..." 
for id in {0..5}
do
    nohup singularity run  install/tf2_addons.2.9.1.sif experiments_manager.py --procID $id  --usersettings examples/federated/mysettings_curve_fitting.py &
done
echo "production model optims are running..." 

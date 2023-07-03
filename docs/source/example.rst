.. highlight:: shell

Example and demonstration
===========================


# Please find here a list of commands that allows to start the demos
# -> These demos suppose that your system has all the tools installed to run them
# A discussed in the README.md the most convenient approach is to rely on a container using Singularity or Apptainer
# and using the receipe provided in the install folder.
# in the following, one expects the following container to exist: /path/to/containers/tf2_addons.2.11.0.sif

# NOTES:
# -> once run, all demos produce logs in specific timestamped folders in the 'experiments' folder
# -> have a look at the logs reported on the Tensorboard tool using command
apptainer exec /path/to/containers/tf2_addons.2.11.0.sif tensorboard --logdir experiments/YOURTARGETSUBFOLDER
 
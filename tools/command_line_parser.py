#basic command line parser that helps control the experiments_manager.py script
import argparse
def get_commands():
    parser = argparse.ArgumentParser(description='Deep learning experiments manager')
    parser.add_argument("-m","--model_dir", default=None,
                        help="Output directory for model and training stats.")
    parser.add_argument("-d","--debug", action='store_true',
                        help="set to activate debug mode")
    parser.add_argument("-p","--predict", action='store_true',
                        help="Switch to prediction mode")
    parser.add_argument("-l","--predict_stream", default=0, type=int,
                        help="set the number of successive predictions, infinite loop if <0")
    parser.add_argument("-s","--start_server", action='store_true',
                        help="start the tensorflow server on the machine to run predictions")
    parser.add_argument("-psi","--singularity_tf_server_container_path", default='',
                        help="start the tensorflow server on a singularity container to run predictions")
    parser.add_argument("-u","--usersettings",
                        help="filename of the settings file that defines an experiment")
    parser.add_argument("-r","--restart_interrupted", action='store_true',
                        help="Set to restart an interrupted session, model_dir option should be set")
    parser.add_argument("-pid","--procID", default=None,
                        help="Specifiy here an ID to identify the process (useful for federated training sessions)")
    parser.add_argument("-dist","--distributed", action='store_true',
                        help="activate this option to make use of sidtributed computing (approach depends on the expereiments settings specifications")

    parser.add_argument("-c","--commands", action='store_true',
                        help="show command examples")

    return parser


def get_default_args():

    parser=get_commands()
    return parser.parse_args([])
    
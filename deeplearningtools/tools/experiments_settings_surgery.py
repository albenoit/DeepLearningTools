# ========================================
# FileName: experiment_settings_surgery.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A function to inserts additional hyperparameters into the configuration file
# for DeepLearningTools.
# =========================================

import numpy as np
import os
import re

ADDITIONNAL_PARAMS_FINAL_LINE='#NEXT FOLLOWS THE ORIGINAL SETTINGS FILE\n'

def insert_additionnal_hparams(settings_file, hparams):
  """
  Inserts additional hyperparameters into a settings file and returns the path of the updated file.
  
  :param settings_file: The path of the original settings file.
  :param hparams: A dictionary containing the additional hyperparameters to be inserted.

  :return: The path of the updated settings file.
  """
  #basic stop condition, if no additionnal hparams, then return the input script filename
  if hparams is None:
    return settings_file
  # read the original experiments settings file in READ mode
  original_settings_file=open(settings_file,"r")
  print('reading file', original_settings_file)
  updated_settings_data=original_settings_file.readlines() #returns the LIST of lines

  original_settings_file.close()
  #Here, we prepend the string we want to on first line
  additionnal_hparams="#NEXT FOLLOWS SOME ADDITIONNAL HYPERPARAMETERS UPDATES, AUTOMATICALY INSERTED, MAY UPDATE PREVIOUSLY DEFINED VALUES\n"+'hparams_addons='+str(hparams)+'\n'
  #make the link between those added hparams and the expected hand written hparams
  hparams_line_id=0

  # FIXME : UGLY file processing follows to insert additionnal hyperparameters...
  def find_line_after_initial_hparams_dict(allLines):
    line_after_hparams_dict=0
    found_hparams_dict=False
    #first find the hparams opening declaration
    hparams_start_line=0
    for lineID, line in enumerate(allLines):
      if re.match(r"^hparams[\s]{0,}=[\s]{0,}{",line):
        hparams_start_line=lineID
        found_hparams_dict=True
        break
    #print('hparams_start_line='+str(hparams_start_line))
    #finally find the hparams declaration closing '}'
    for lineID, line in enumerate(allLines[hparams_start_line:]):
      if '}' in line:
        line_after_hparams_dict=hparams_start_line+lineID+1
        break
    #raw_input('hparams_stop_line='+str(line_after_hparams_dict))

    return found_hparams_dict, line_after_hparams_dict
  def find_last_hparams_addons_line(allLines):
    """"
    Tries to find ADDITIONNAL_PARAMS_FINAL_LINE within the provided lines

    :return: the line offset that corresponds to the last found occurence
    """
    nb_lines=len(allLines)-1
    for lineID, line in enumerate(reversed(allLines)):
      if line == ADDITIONNAL_PARAMS_FINAL_LINE:
        break
    if lineID==nb_lines:
      return 0
    return nb_lines-lineID +1
  
  hparams_line_insert=0
  has_hparams, hand_written_hparam_line=find_line_after_initial_hparams_dict(updated_settings_data[hparams_line_insert:])
  
  # also search for already added additional parameters (willing to add some more after them)
  ADDITIONNAL_PARAMS_FINAL_LINE
  hparams_line_insert+=hand_written_hparam_line
  if has_hparams:
    hparams_line_insert+=find_last_hparams_addons_line(updated_settings_data[hparams_line_insert:])
    additionnal_hparams+='hparams.update(hparams_addons)'
  else:
    additionnal_hparams+='hparams=hparams_addons'
  additionnal_hparams+='\n'+ADDITIONNAL_PARAMS_FINAL_LINE
  #Prepending string
  updated_settings_data.insert(hparams_line_insert,additionnal_hparams)
  # write the updated settings file to a tmp folder
  randomID=np.random.randint(0,1e6)
  updated_settings_filename=os.path.basename(settings_file)+'.withAddedHparams.'+str(randomID)+'.py'
  tmpdir='tmp/'
  #write temporary settings file
  try:
    print('Creating folder :', tmpdir)
    os.makedirs(tmpdir)
  except Exception as e:
    print('Could not create ',tmpdir, 'already exists:', os.path.exists(tmpdir))
  updated_settings_filename=os.path.join(tmpdir, updated_settings_filename)
  print('updated_settings_filename',updated_settings_filename)
  try:
    new_settings_file=open(updated_settings_filename,'w')
  except Exception as e:
    raise Exception('Something went wrong when writing temporary settings file...',e)
  new_settings_file.writelines(updated_settings_data)
  new_settings_file.close()
  settings_file=updated_settings_filename
  print('Created a temporary updated experiment settings file here : ' +settings_file)
  return settings_file

### testing purpose
#script='experiment_settings.py'
#insert_additionnal_hparams(script, {'test':1})

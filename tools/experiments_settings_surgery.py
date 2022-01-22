import numpy as np
import os

def reload_external_hparams_from_file(sessionFolder):
  hparams_addons=None
  #then try to load optionnal external hyperparameters in the session folder
  print("sessionFolder",sessionFolder)
  filename_optionnal_hparams=os.path.join(sessionFolder, settingsFile_addons_saveName)
  print('**Looking for additionnal hyperparameters in optionnal file '+filename_optionnal_hparams)
  if os.path.exists(filename_optionnal_hparams):
    print('  -> Found external hyperparameters file')
    hparams_addons=imp.load_source('settings', filename_optionnal_hparams).hparams_addons
    print('  -> loaded additionnal_hparams : '+str(hparams_addons))
  return hparams_addons

def insert_additionnal_hparams(settings_file, hparams):

  #basic stop condition, if no additionnal hparams, then return the input script filename
  if hparams is None:
    return settings_file
  # read the original experiments settings file in READ mode
  original_settings_file=open(settings_file,"r")
  print('reading file', original_settings_file)
  updated_settings_data=original_settings_file.readlines() #returns the LIST of lines

  original_settings_file.close()
  #Here, we prepend the string we want to on first line
  additionnal_hparams="#IN THE NEXT FEW FOLLOWING LINES SOME ADDITIONNAL HYPERPARAMETERS HAVE BEEN AUTOMATICALY INSERTED\n"+'hparams_addons='+str(hparams)+'\n'
  #make the link between those added hparams and the expected hand written hparams
  hparams_line_id=0

  # FIXME : UGLY file processing follows to insert additionnal hyperparameters...
  import re
  def get_next_line_after_last_python_future_line(allLines):
    last_future_lineID=0
    for lineID, line in enumerate(allLines):
      if re.match("^from[\s]__future__",line):
        last_future_lineID=lineID
    if last_future_lineID>0:
      last_future_lineID+=1#propose the next line to enable safe text insertion
    #print('last_future_lineID='+str(last_future_lineID))
    return last_future_lineID

  def find_line_after_initial_hparams_dict(allLines):
    line_after_hparams_dict=0
    found_hparams_dict=False
    #first find the hparams opening declaration
    hparams_start_line=0
    for lineID, line in enumerate(allLines):
      if re.match("^hparams[\s]{0,}=[\s]{0,}{",line):
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

  hparams_line_insert=0#get_next_line_after_last_python_future_line(updated_settings_data)
  has_hparams, hand_written_hparam_line=find_line_after_initial_hparams_dict(updated_settings_data[hparams_line_insert:])
  hparams_line_insert+=hand_written_hparam_line
  if has_hparams:
    additionnal_hparams+='hparams.update(hparams_addons)'
  else:
    additionnal_hparams+='hparams=hparams_addons'
  additionnal_hparams+='\n#NEXT FOLLOWS THE ORIGINAL SETTINGS FILE\n\n'
  #Prepending string
  updated_settings_data.insert(hparams_line_insert,additionnal_hparams)
  # write the updated settings file to a tmp folder
  randomID=np.random.randint(0,1e6)
  updated_settings_filename=os.path.basename(settings_file)+'.withAddedHparams.'+str(randomID)+'.py'
  tmpdir='tmp/'
  #write temporary settings file
  if not(os.path.exists(tmpdir)):
    os.makedirs(tmpdir)
  updated_settings_filename=os.path.join(tmpdir, updated_settings_filename)
  print('updated_settings_filename',updated_settings_filename)
  try:
    new_settings_file=open(updated_settings_filename,'w')
  except Exception as e:
    print('Something went wrong when writing temporary settings file...',e)
  new_settings_file.writelines(updated_settings_data)
  new_settings_file.close()
  settings_file=updated_settings_filename
  print('Created a temporary updated experiment settings file here : ' +settings_file)
  return settings_file

### testing purpose
#script='experiment_settings.py'
#insert_additionnal_hparams(script, {'test':1})

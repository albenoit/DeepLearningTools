import os

def create_session_folder(sessionFolderBase, usersettings):
    sessionFolder = sessionFolderBase
    sessionFolder_addon=''
    for key, value in usersettings.hparams.items():
        sessionFolder_addon+='_'+key+str(value)
    #insert sessionname addons in the original one
    sessionFolder+=sessionFolder_addon

    #truncate if path exceeds sytem max length
    max_length = os.pathconf('/', 'PC_NAME_MAX')
    if len(sessionFolder)>max_length:
        sessionFolder=sessionFolder[:max_length]
    #print('Found hparams: '+str(usersettings.hparams))
    return sessionFolder

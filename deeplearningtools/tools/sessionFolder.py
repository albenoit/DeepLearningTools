

def create_session_folder(sessionFolderBase, usersettings):
    sessionFolder = sessionFolderBase
    sessionFolder_addon=''
    for key, value in usersettings.hparams.items():
        sessionFolder_addon+='_'+key+str(value)
    #insert sessionname addons in the original one
    sessionFolder+=sessionFolder_addon
    #print('Found hparams: '+str(usersettings.hparams))
    return sessionFolder

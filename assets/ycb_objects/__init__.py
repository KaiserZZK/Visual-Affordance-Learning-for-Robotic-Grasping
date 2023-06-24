import os

def getDataPath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir

def getURDFPath(name):
    urdf_path = os.path.join(getDataPath(),name,'model.urdf')
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(urdf_path)
    return urdf_path

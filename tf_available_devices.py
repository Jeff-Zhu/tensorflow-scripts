import warnings

warnings.filterwarnings('ignore',category=FutureWarning)

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_devices = device_lib.list_local_devices()
    print("All local devices: ", local_devices)
    return [x.name for x in local_devices if x.device_type == 'GPU']

get_available_gpus()

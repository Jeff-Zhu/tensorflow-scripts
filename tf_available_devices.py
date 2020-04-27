import warnings

warnings.filterwarnings('ignore',category=FutureWarning)

from tensorflow.python.client import device_lib
import tensorflow as tf

def get_available_gpus():
    local_devices = device_lib.list_local_devices()
    print("All local devices: ", local_devices)

    print(" tf.test.is_built_with_cuda() = ", tf.test.is_built_with_cuda() )
    print(" tf.test.is_gpu_available() = ", tf.test.is_gpu_available())
    return [x.name for x in local_devices if x.device_type == 'GPU']

get_available_gpus()

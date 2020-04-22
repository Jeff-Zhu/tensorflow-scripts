import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

def main(saved_model_dir):

  model = tf.saved_model.load(tags=[tf.compat.v1.saved_model.tag_constants.SERVING], export_dir=saved_model_dir)
  concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  concrete_func.inputs[0].set_shape([1, 224, 224, 3])
  converter = tf.lite.TFLiteConverterV2.from_concrete_functions([concrete_func])

  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_quant_model = converter.convert()

if __name__== "__main__":

  main(sys.argv[1])

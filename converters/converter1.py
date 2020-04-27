import sys
import pathlib
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

def main(saved_model_dir):
  converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  # converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
  converter.default_ranges_stats = (0, 255)
  input_arrays = converter.get_input_arrays()
  converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean_value, std_dev
  tflite_model = converter.convert()

  tflite_model_file = pathlib.Path(
       saved_model_dir + "/model_quantized.tflite"
       )
  tflite_model_file.write_bytes(tflite_model)

  ## converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  ## converter.optimizations = [tf.lite.Optimize.DEFAULT]
  ## converter.representative_dataset = representative_dataset_gen
  ## tflite_quant_model = converter.convert()

if __name__== "__main__":

  main(sys.argv[1])

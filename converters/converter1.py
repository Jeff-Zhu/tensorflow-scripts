import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

def main(saved_model_dir):
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset_gen
  tflite_quant_model = converter.convert()

if __name__== "__main__":

  main(sys.argv[1])

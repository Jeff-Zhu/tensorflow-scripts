#!/usr/bin/env python
#
# zjxy63@gmail.com
# ==============================================================================
"""This convenient script quantizes a TensorFlow Lite mode.

Example usage:

python tflig_quantize.py foo.tflite foo_quantized.tflit
"""

import sys

import warnings

warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf


def quantize_mode( saved_model_dir, quantized_tflite_output, optimizations):
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.optimizations = optimizations
  tflite_quant_model = converter.convert()
  # Save the quantized model to disk
  open(quantized_tflite_output, "wb").write(tflite_model)
  


def main(argv):
  try:
    saved_model_dir = argv[1]
    quantized_tflite_output = argv[2]
    if (len(argv) > 3):
      opt_str = argv[3].lower()
      if (opt_str.find('size')):
        optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
      elif (opt_str.find('latency')):
        optimizations = [tf.lite.Optimize.LATENCY]
      else:
        optimizations = [tf.lite.Optimize.DEFAULT]
    else:
      # DEFAULT, OPTIMIZE_FOR_LATENCY, OPTIMIZE_FOR_SIZE
      optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

    
  except IndexError:
    print("Usage: %s <saved model dir> <output quantized tflite [optiomizations]>" % (argv[0]))
  else:
    quantize_mode(saved_model_dir, quantized_tflite_output, optimizations)


if __name__ == "__main__":
  main(sys.argv)

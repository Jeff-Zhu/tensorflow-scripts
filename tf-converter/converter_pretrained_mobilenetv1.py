import os
import sys
import pathlib

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as npimport 

def main(saved_model_dir):

  if saved_model_dir:
  model = tf.keras.applications.MobileNet(weights="imagenet", input_shape=(224, 224, 3), alpha=0.25)
  # here we pretrained model no need use SaveModel 
  # here we will pass model directly to TFLiteConverter
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
 
  #if you want to save the TF Lite model use below steps or else skip
  tflite_model_file = pathlib.Path('/tmp/pretrainedmodel.tflite')
  tflite_model_file.write_bytes(tflite_model)

  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  tflite_quant_model = converter.convert()
  tflite_quant_model_file = pathlib.Path('/tmp/pretrainedmodel_quantized.tflite')
  tflite_quant_model_file.write_bytes(tflite_quant_model)

if __name__== "__main__":

  main(sys.argv[1])

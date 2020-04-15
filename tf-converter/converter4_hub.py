import sys
import pathlib

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as npimport 

def main(saved_model_dir):

  model = tf.keras.applications.MobileNet(weights="imagenet", input_shape=(224, 224, 3), alpha=0.25)

  IMAGE_SHAPE = (224, 224)
  num_classes = 1000
  model = tf.keras.Sequential([
      hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4", input_shape=IMAGE_SHAPE+(3,), trainable=False),  # Can be True, see below.
      tf.keras.layers.Dense(num_classes, activation='softmax')
  ])
  model.build([None, 224, 224, 3])  # Batch input shape.
  ## inputs = tf.keras.Input(shape=(224, 224, 3))
  ## outputs = model(inputs)
  ## model = tf.keras.Model(inputs=inputs, outputs=outputs)
  # here we pretrained model no need use SaveModel 
  # here we will pass model directly to TFLiteConverter
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
 
  #if you want to save the TF Lite model use below steps or else skip
  tflite_model_file = pathlib.Path('/tmp/pretrainedmodel.tflite')
  tflite_model_file.write_bytes(tflite_model)

  # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_DEFAULT,  .OPTIMIZE_FOR_SIZE]
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
  tflite_quant_model = converter.convert()
  tflite_quant_model_file = pathlib.Path('/tmp/pretrainedmodel_quantized.tflite')
  tflite_quant_model_file.write_bytes(tflite_quant_model)

if __name__== "__main__":

  main(sys.argv[1])

import os
import sys
import pathlib
from pathlib import Path
from typing import Any, Union

from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf
import numpy as npimport

import xavier_mobilenet_yolo2_backbone as xmyolo2


def main(saved_model_dir, model_name_prefix):

    input_shape = (128, 128, 3)
    alpha = 0.25
    model = xmyolo2.MobileNetYolo2Backbone(input_shape=input_shape, alpha=alpha)

    print("MobileNetYolo2Backbone ", input_shape, "alpha = ", alpha, " Summary:")
    print(model.summary())

    keras_model_file: Union[Path, Any] = pathlib.Path(
        saved_model_dir + "/" + model_name_prefix + ".h5"
    )
    print("Saving keras model to ", keras_model_file)
    model.save(keras_model_file)

    # here we pretrained model no need use SaveModel
    # here we will pass model directly to TFLiteConverter
    ## TF 2.0
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # TF 1.14
    # converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_file)
    tflite_model = converter.convert()

    # if you want to save the TF Lite model use below steps or else skip
    tflite_model_file = pathlib.Path(
        saved_model_dir + "/" + model_name_prefix + ".tflite"
    )
    tflite_model_file.write_bytes(tflite_model)


    # Create V1 converter for full integer optimization
    filepath = '/tmp/saved_model'
    # TF 2.1
    tf.keras.models.save_model(model, filepath, save_format='tf')

    ### checkpoint_path = "/tmp/checkpoint/cp-{epoch:04d}.ckpt"
    ### checkpoint_dir = os.path.dirname(checkpoint_path)
    ### 
    ### # Save the weights using the `checkpoint_path` format
    ### model.save_weights(checkpoint_path.format(epoch=0))
    

    ## # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_DEFAULT,  .OPTIMIZE_FOR_SIZE]
    ## # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    ## converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    ## converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
    ## tflite_quant_model = converter.convert()
    ## tflite_quant_model_file = pathlib.Path(
    ##     saved_model_dir + "/" + model_name_prefix + "_quantized.tflite"
    ## )
    ## tflite_quant_model_file.write_bytes(tflite_quant_model)
    ## 
    ## # print("TFLite quantized MobileNet 224, 224, 3), alpha=0.25 Summary:")
    ## # print(tflite_model.summary())


if __name__ == "__main__":

    if len(sys.argv) > 2:
        model_prefix = sys.argv[2]
    else:
        model_prefix = "xmnet_yolo2"

    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = "."

    main(saved_model_dir=model_dir, model_name_prefix=model_prefix)

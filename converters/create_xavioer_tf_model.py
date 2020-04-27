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
    # TF 2.1
    tf.keras.models.save_model(model, saved_model_dir, save_format="tf")

    # Create V1 converter from *.pb in saved model dir
    q8_converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # q8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    q8_converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
    q8_converter.default_ranges_stats = (0, 255)
    input_arrays = q8_converter.get_input_arrays()
    q8_converter.quantized_input_stats = {
        input_arrays[0]: (0.0, 1.0)
    }  # mean_value, std_dev
    tflite_model = q8_converter.convert()

    tflite_model_file = pathlib.Path(
        saved_model_dir + "/" + model_name_prefix + "_q8.tflite"
    )
    tflite_model_file.write_bytes(tflite_model)


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

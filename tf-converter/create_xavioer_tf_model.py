import sys
import pathlib

from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf
import numpy as npimport

import xavier_mobilenet_yolo2_backbone as xmyolo2


def main(saved_model_dir, model_name_prefix):

    input_shape = (64, 64, 1)
    alpha = 0.25
    model = xmyolo2.MobileNetYolo2Backbone(input_shape=input_shape, alpha=alpha)

    print("MobileNetYolo2Backbone ", input_shape, "alpha = ", alpha, " Summary:")
    print(model.summary())

    # here we pretrained model no need use SaveModel
    # here we will pass model directly to TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # if you want to save the TF Lite model use below steps or else skip
    tflite_model_file = pathlib.Path(
        saved_model_dir + "/" + model_name_prefix + ".tflite"
    )
    tflite_model_file.write_bytes(tflite_model)

    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_DEFAULT,  .OPTIMIZE_FOR_SIZE]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
    tflite_quant_model = converter.convert()
    tflite_quant_model_file = pathlib.Path(
        saved_model_dir + "/" + model_name_prefix + "_quantized.tflite"
    )
    tflite_quant_model_file.write_bytes(tflite_quant_model)

    # print("TFLite quantized MobileNet 224, 224, 3), alpha=0.25 Summary:")
    # print(tflite_model.summary())


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

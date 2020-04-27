import sys
import pathlib

from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf
import numpy as npimport

import mobilenet_v1_backbone as mobilenet_v1

# usage: python converter3.py ~/tmp '(256, 256, 3)'


def main(saved_model_dir, input_shape):

    # input_shape = (64, 64, 1)
    alpha = 0.25
    use_original_mobilenet = True

    if use_original_mobilenet:
        # Original MobileNet
        model = tf.keras.applications.MobileNet(
            input_shape=input_shape,
            alpha=alpha,
            weights=None,
            include_top=True,
            pooling="avg",
            classes=5,
        )
    else:
        # MobileNet by Andrew Luo
        model = mobilenet_v1.MobileNet(
            input_shape=input_shape, alpha=alpha, include_top=True
        )

    print("MobileNet V1 ", input_shape, " alpha= ", alpha, " Summary:")
    print(model.summary())

    # here we pretrained model no need use SaveModel
    # here we will pass model directly to TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    model_name = "mobilenet_%0.2f" % (alpha) + "_%dx%dx%d" % (input_shape)
    # if you want to save the TF Lite model use below steps or else skip
    tflite_model_file = pathlib.Path(saved_model_dir + "/" + model_name + ".tflite")
    tflite_model_file.write_bytes(tflite_model)

    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_DEFAULT,  .OPTIMIZE_FOR_SIZE]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
    tflite_quant_model = converter.convert()
    tflite_quant_model_file = pathlib.Path(
        saved_model_dir + "/" + model_name + "_q8.tflite"
    )
    tflite_quant_model_file.write_bytes(tflite_quant_model)

    # print("TFLite quantized MobileNet 128, 128, 3), alpha=0.25 Summary:")
    # print(tflite_quant_model.summary())


if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = "."

    if len(sys.argv) > 2:
        input_shape = eval(sys.argv[2])
    else:
        input_shape = (128, 128, 3)

    print("input_shape = ", input_shape)

    main(model_dir, input_shape)

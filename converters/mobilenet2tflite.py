import sys
import pathlib

from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf
import numpy as npimport


# usage: python mobilenet2tflite.py model_output_dir input_shape model_type
#       model_output_dir: tflite model files output directory
#       input_shape: as'(256, 256, 3)' etc
#       model_type:
#          "standard" or "std" or "original" for original MobileNet V1
#          "varian" or "andrew" or "luo" for MobileNet V1 variant by Andrew Luo
#          default to Xaview MobileNet V1 yolo2


def main(saved_model_dir, input_shape=(128, 128, 3), model_type=None):

    alpha = 0.25
    model_type = model_type.lower()

    if model_type in "standard std original":
        # Original MobileNet
        model = tf.keras.applications.MobileNet(
            input_shape=input_shape,
            alpha=alpha,
            weights=None,
            include_top=True,
            pooling="avg",
            classes=5,
        )
        model_name = "mobilenet_v1_a%0.2f" % (alpha) + "_%dx%dx%d" % (input_shape)
    elif model_type in "variant andrew luo":
        # MobileNet by Andrew Luo
        import mobilenet_v1_backbone as mobilenet_v1

        model = mobilenet_v1.MobileNet(
            input_shape=input_shape, alpha=alpha, include_top=True
        )
        model_name = "mobilenet_v1a_a%0.2f" % (alpha) + "_%dx%dx%d" % (input_shape)
    else:
        # Xaview MobileNet yolo2
        import xavier_mobilenet_yolo2_backbone as xmyolo2

        model = xmyolo2.MobileNetYolo2Backbone(input_shape=input_shape, alpha=alpha)
        model_name = "xmnet_yolo2_a%0.2f" % (alpha) + "_%dx%dx%d" % (input_shape)

    print(model_name, " Summary:")
    print(model.summary())

    # here we pretrained model no need use SaveModel
    # here we will pass model directly to TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # if you want to save the TF Lite model use below steps or else skip
    tflite_model_file = pathlib.Path(saved_model_dir + "/" + model_name + ".tflite")
    print("Saving TFLite model to: ", tflite_model_file)
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

    tflite_model_file = pathlib.Path(saved_model_dir + "/" + model_name + "_q8.tflite")
    print("Saving TFLite Q8 model to: ", tflite_model_file)
    tflite_model_file.write_bytes(tflite_model)

    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_DEFAULT,  .OPTIMIZE_FOR_SIZE]
    # q8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    q8_converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
    tflite_quant_model = q8_converter.convert()
    tflite_quant_model_file = pathlib.Path(
        saved_model_dir + "/" + model_name + "_q8.tflite"
    )
    tflite_quant_model_file.write_bytes(tflite_quant_model)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "."

    if len(sys.argv) > 2:
        input_shape = eval(sys.argv[2])
    else:
        input_shape = (128, 128, 3)

    if len(sys.argv) > 3:
        model_type = sys.argv[3]
    else:
        model_type = "xavier"

    print("Model Ourput Dir = ", output_dir)
    print("Input_shape = ", input_shape, " Model_Type = ", model_type)

    main(saved_model_dir=output_dir, input_shape=input_shape, model_type=model_type)

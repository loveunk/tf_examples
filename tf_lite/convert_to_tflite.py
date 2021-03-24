"""model conversion example for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.lite as tflite

saved_model_dir = './mobilenet/'

model = tf.keras.applications.MobileNet()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--type',
        default='keras_model',
        choices=['saved_model', 'keras_model', 'concrete_functions'],
        help='type of source model')
    args = parser.parse_args()

    if (args.type == 'saved_model'):
        tf.saved_model.save(model, saved_model_dir)

        # Converting a SavedModel to a TensorFlow Lite model.
        converter = tflite.TFLiteConverter.from_saved_model(saved_model_dir)

    elif (args.type == 'keras_model'):
        # Converting a tf.Keras model to a TensorFlow Lite model.
        converter = tflite.TFLiteConverter.from_keras_model(model)

    elif (args.type == 'concrete_functions'):
        # Converting ConcreteFunctions to a TensorFlow Lite model.
        tf.saved_model.save(model, saved_model_dir)
        loaded = tf.saved_model.load(saved_model_dir)
        concrete_func = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        converter = tflite.TFLiteConverter.from_concrete_functions([concrete_func])

    else:
        assert('type `{}` not supported'.format(args.type))

    tflite_model = converter.convert()

    open(os.path.join(saved_model_dir, "converted_model.tflite"), "wb").write(tflite_model)

    #print(tflite_model)
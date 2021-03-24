# TensorFlow Lite Example

## Convert a pre-trained model to TFLite

With following commands, a MobileNet tflite model will be saved in `./mobilenet`.

```python
python convert_to_tflite.py --type saved_model
# supported src types in tf 2.x: 'saved_model', 'keras_model', 'concrete_functions'
```

## Infer with TFLite

```python
python label_image.py --image ./grace_hopper.bmp
                      --model_file mobilenet/converted_model.tflite
```

Output (in a Ubuntu PC):

```shell
0.817774: 653:military uniform
0.117174: 458:bow tie, bow-tie, bowtie
0.025351: 440:bearskin, busby, shako
0.009244: 835:suit, suit of clothes
0.005195: 716:pickelhaube
time: 59.519ms
```


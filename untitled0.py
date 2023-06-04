# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:59:32 2023

@author: aidan
"""
import tensorflow as tf

print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices("GPU"))

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':
    print(get_available_gpus())
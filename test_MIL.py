import tensorflow as tf
import numpy as np
import h5py
import time
import sys
import os
instance_list ={}
instance_list.update(zip([v.strip().split(' ')[0] for v in open('./vocabulary/instance_list.txt','r').readlines()],range(

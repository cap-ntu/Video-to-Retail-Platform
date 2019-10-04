from __future__ import print_function

import tensorflow as tf
import ausioset.vggish_params as vggish_params
import audioset.vggish_slim as vggish_slim

from tensorflow.python.tools import freeze_graph
from baselines.TFS.transform_pb_to_server_model import *

print('\nTesting your install of VGGish\n')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'


with tf.Graph().as_default(), tf.Session() as sess:
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    save_server_models(sess, features_tensor, embedding_tensor)


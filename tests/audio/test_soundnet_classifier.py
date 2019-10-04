import tensorflow as tf
import _init_paths
from models.scene.soundnet_classifier import SoundNetClassifier

if __name__ == '__main__':
    # Load text module graph
    PATH_TO_TEST_AUDIO = 'test_airport.wav'
    PATH_TO_PB = '../../weights/soundnet/soundnet_fr.pb'

    # Scene start&end timestamp in milliseconds
    sc_start = 0
    sc_end = 2000

    soundnet_graph = tf.Graph()
    with soundnet_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_PB, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    soundnet = SoundNetClassifier(soundnet_graph)
    print(soundnet.classify_scene(PATH_TO_TEST_AUDIO, sc_start, sc_end))
    print(soundnet.classify_frame(PATH_TO_TEST_AUDIO, fr=10))

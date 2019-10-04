import tensorflow as tf
import _init_paths
from models.object.audioset_feature_extractor import AudiosetFeatureExtractor

if __name__ == '__main__':
    # Load text module graph
    PATH_TO_TEST_AUDIO = 'test_DB/audios/BBT0624.wav'
    PATH_TO_VGG_GRAPH = '../weights/audioset/vggish_fr.pb'
    PATH_TO_PCA_PARAMS = '../weights/audioset/vggish_pca_params.npz'

    # Scene start&end timestamp in milliseconds
    sc_start = 0
    sc_end = 2000

    vgg_graph = tf.Graph()
    with vgg_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_VGG_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    audioset_feature_extractor = AudiosetFeatureExtractor(vgg_graph, PATH_TO_PCA_PARAMS)

    embeddings = audioset_feature_extractor.extract(PATH_TO_TEST_AUDIO, sc_start, sc_end)
    print(embeddings[0])
    print(embeddings.shape)

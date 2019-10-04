import _init_paths

from trainer.scene.soundnet import SoundNet
import tensorflow as tf
import json
import numpy as np

local_config = {
    'batch_size': 64,
    'train_size': np.inf,
    'save_interval': 1000,
    'epoch': 200,
    'eps': 1e-5,
    'learning_rate': 1e-3,
    'beta1': 0.9,
    'load_size': 22050 * 10,
    'sample_size': 22050 * 5,
    'sample_rate': 22050,
    'augment_factor': 2,
    'augment_hop': 22050 * 0.5,
    'name_scope': 'SoundNet',
    'phase': 'train',
    'dataset_name': 'dcase2018',
    'subname': 'wav',
    'checkpoint_dir': '../../weights/soundnet',
    'dump_dir': 'output',
    'model_dir': None,
    'load_mode': 'pb',
    'pb_name': 'soundnet_fr.pb',
    'param_g_dir': 'PretrainedModel/sound8.npy',
    'label_csv': '/home/lzy/dcase2018/development/meta.csv'
}




if __name__ == '__main__':

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as session:
        # Build model
        model = SoundNet(session, local_config)
        print(model.layers[35].name)
        print(model.sound_input_placeholder.name)
        result = model.predict_file('../test_DB/test_airport.wav', fr=10)
        # result = json.dumps(result)
        # with open('test_result.out', 'w') as f:
        #     f.write(result)

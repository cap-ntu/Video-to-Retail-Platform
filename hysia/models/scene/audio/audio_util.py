import numpy as np
import librosa

local_config = {
    'batch_size': 64,
    'load_size': 22050 * 20,
    'phase': 'extract'
}


def load_from_list(name_list, config=local_config):
    assert len(name_list) == config['batch_size'] // config['augment_factor'], \
        "The length of name_list({})[{}] is not the same as batch_size[{}]".format(
            name_list[0], len(name_list), config['batch_size'])
    audios = np.zeros([config['batch_size'] // config['augment_factor'], config['load_size'], 1, 1])
    for idx, audio_path in enumerate(name_list):
        sound_sample, _ = load_audio(audio_path)
        audios[idx] = preprocess(sound_sample, config)

    return audios


def load_from_txt(txt_name, config=local_config):
    with open(txt_name, 'r') as handle:
        txt_list = handle.read().splitlines()

    audios = []
    for idx, audio_path in enumerate(txt_list):
        sound_sample, _ = load_audio(audio_path)
        audios.append(preprocess(sound_sample, config))

    return audios


# NOTE: Load an audio as the same format in soundnet
# 1. Keep original sample rate (which conflicts their own paper)
# 2. Use first channel in multiple channels
# 3. Keep range in [-256, 256]

def load_audio(audio_path, sr=None, mono=True):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=sr, mono=mono)

    return sound_sample, sr


def preprocess(raw_audio, config=local_config, is_train=True):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]

    # Make range [-256, 256]
    raw_audio *= 256.0

    # Make minimum length available
    length = config['load_size']
    if length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, length / raw_audio.shape[0] + 1)

    # Make equal training length
    if config['phase'] != 'extract':
        raw_audio = raw_audio[:length]

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we only need the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [1, -1, 1, 1])
    if not is_train:
        pad_w = int(config['sample_size'] / 2)
        raw_audio = np.pad(raw_audio, [(0, 0), (pad_w, pad_w), (0, 0), (0, 0)], mode='constant', constant_values=0)
    return raw_audio.copy()

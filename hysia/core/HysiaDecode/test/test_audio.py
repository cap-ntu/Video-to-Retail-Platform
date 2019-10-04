import sys
sys.path.insert(0, '../build')

import PyDecoder

test = PyDecoder.AudioDecoder()

test.ingestVideo("bigbang.mp4")

test.decodeClips()

test.saveWav("bigbang_python.wav")

dd = test.getData()

import scipy.io.wavfile as wav

s, d = wav.read('./bigbang_ffmpeg.wav')

import numpy as np

print(np.sum(np.equal(d, dd))/len(d))

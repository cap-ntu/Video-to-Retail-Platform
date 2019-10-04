# Desc: Script to test clipper model throughput.
# Author: Zhou Shengsheng
# Date: 19/02/19
# Reference:
#   https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py

import _init_paths
import base64
import pickle
import threading
import time
import argparse
import cv2
from requests_futures.sessions import FuturesSession
from PIL import Image
from models.scene.audio.audio_util import load_audio


class _ResultCounter(object):
    """Throughput calculator."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()
        self._start_time = -1
        self._end_time = 0

    def inc_done(self):
        with self._condition:
            self._done += 1
            if self._done == self._num_tests:
                self.set_end_time(time.time())
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def throttle(self):
        with self._condition:
            if self._start_time == -1:
                self._start_time = time.time()
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

    def set_start_time(self, start_time):
        self._start_time = start_time

    def set_end_time(self, end_time):
        self._end_time = end_time

    def get_throughput(self):
        if self._end_time == 0:
            self.set_end_time(time.time())
        # print(self._end_time - self._start_time)
        return self._num_tests / (self._end_time - self._start_time)


def responseCallback(resp, *args, **kwargs):
    """Clipper api response callback."""
    global model_name, result_counter
    try:
        # TODO: Add other model result handler
        if model_name == 'mmdet':
            result_bytes = base64.b64decode(resp.json()['output'])
            result = pickle.loads(result_bytes)
            # print(result)
        elif model_name == 'imagenet':
            result_bytes = base64.b64decode(resp.json()['output'])
            result = pickle.loads(result_bytes)
            # print(result)
        elif model_name == 'mtcnn':
            result_bytes = base64.b64decode(resp.json()['output'])
            result = pickle.loads(result_bytes)
            rectangles, points, duration = result
            # print('rectangles:', rectangles)
            # print('points:', points)
            # print('duration:', duration)
        elif model_name == 'places365':
            result_bytes = base64.b64decode(resp.json()['output'])
            result = pickle.loads(result_bytes)
        elif model_name == 'soundnet':
            result_bytes = base64.b64decode(resp.json()['output'])
            result = pickle.loads(result_bytes)
        elif model_name == 'sentence-embedding':
            result_bytes = base64.b64decode(resp.json()['output'])
            result = pickle.loads(result_bytes)
        else:
            raise Exception('model not support')
        result_counter.inc_done()
        result_counter.dec_active()
    except Exception as e:
        print(e)


def postClipperApi(data):
    """Send RESTful api to clipper for inference."""
    global session, model_name
    app_name = 'clipper-' + model_name
    headers = {"content-type": "aplication/json"}
    session.post(url="http://localhost:1337/{}/predict".format(app_name), headers=headers, json=data)


def prepareInputData():
    """Prepare input data for model inference. Different model has different input data preparation steps."""
    global model_name
    # TODO: Add other model input data preparation
    if model_name == 'mmdet':
        image_path = '../test1.jpg'
        img = cv2.imread(image_path)
        serialized_img = pickle.dumps(img)  # Use pickle to serialize input into bytes
        base64_img = base64.b64encode(serialized_img).decode()  # Bytes to unicode
        # input_data = {"input": [base64_img]}
        input_data = {"input": base64_img}
    elif model_name == 'imagenet':
        img = Image.open('../test_sofa1.jpg')
        serialized_img = pickle.dumps(img)
        base64_img = base64.b64encode(serialized_img).decode()
        input_data = {"input": base64_img}
    elif model_name == 'mtcnn':
        image_path = '../test1.jpg'
        img = cv2.imread(image_path)
        serialized_img = pickle.dumps(img)
        base64_img = base64.b64encode(serialized_img).decode()
        input_data = {"input": base64_img}
    elif model_name == 'places365':
        img = cv2.imread('../test_beach.jpg')
        serialized_img = pickle.dumps(img)
        base64_img = base64.b64encode(serialized_img).decode()  # Bytes to unicode
        input_data = {"input": base64_img}
    elif model_name == 'soundnet':
        # Load text module graph
        audio_file = '../audio/test_airport.wav'
        # Scene start&end timestamp in milliseconds
        task_type = 'classify_scene'
        # task_type = 'classify_frame'
        sc_start = 0
        sc_end = 2000
        frame = 10
        wav_data, sample_rate = load_audio(audio_file, sr=22050, mono=True)
        raw_input = (task_type, wav_data, sample_rate, sc_start, sc_end, frame)
        serialized_input = pickle.dumps(raw_input)
        base64_input = base64.b64encode(serialized_input).decode()  # Bytes to unicode
        input_data = {"input": base64_input}
    elif model_name == 'sentence-embedding':
        # Test sentence
        sentence = "I am a sentence for which I would like to get its embedding."
        raw_input = sentence
        serialized_input = pickle.dumps(raw_input)
        base64_input = base64.b64encode(serialized_input).decode()  # Bytes to unicode
        input_data = {"input": base64_input}
    else:
        raise Exception('model not support')
    return input_data


def testInferenceThroughput():
    """Test clipper model inference throughput."""
    global model_name, num_tests, result_counter
    printProgressBar(0, num_tests, prefix='Testing progress:', suffix='Complete', length=50)
    for i in range(num_tests):
        result_counter.throttle()
        postClipperApi(prepareInputData())
        printProgressBar(i + 1, num_tests, prefix='Testing progress:', suffix='Complete', length=50)
    return result_counter.get_throughput()


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar.
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def parseArgs():
    """Parse command line args."""
    parser = argparse.ArgumentParser(description="Test clipper model inference throughput.")
    parser.add_argument("--model_name", type=str, default="mmdet", help="Model to test.")
    parser.add_argument("--num_tests", type=int, default=100, help="Total tests to run.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent jobs.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parseArgs()

    model_name = args.model_name
    num_tests = args.num_tests
    concurrency = args.concurrency
    session = FuturesSession()
    session.hooks['response'] = responseCallback
    result_counter = _ResultCounter(num_tests, concurrency)

    print('Testing clipper model inference: model_name={}, num_tests={}, concurrency={}'
          .format(model_name, num_tests, concurrency))
    tps = testInferenceThroughput()
    print('{} model inference TPS: {}/s'.format(model_name, tps))

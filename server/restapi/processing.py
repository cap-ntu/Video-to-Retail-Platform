"""
Hysia video process pipeline
Author: Jiang Wenbo
Email: jian0085@e.ntu.edu.sg
"""
import json
import math
import os
import os.path as osp
import pickle
import subprocess

import cv2
import imageio
import numpy as np
import tqdm
from django.conf import settings
from django.db import transaction

from hysia.models.scene.shot_detecor import Shot_Detector
from restapi.models import Scene
from restapi.rpc_client import RpcClient
from restapi.serializers import FrameSerializer

# Do set HYSIA_BUILD=TRUE when reset django. As PyDecoder will seeking for CUDA if HYSIA_BUILD is not set. This will
# lead to an libcuda.so not found error if you do not have cuda support during building (e.g. in docker)
try:
    build_flag = os.environ['HYSIA_BUILD'].upper() == 'TRUE'
except KeyError:
    build_flag = False
if not build_flag:
    # Import Hysia's own hardware decoding module
    from hysia.core.HysiaDecode.build import PyDecoder


# Decode media with FFmpeg
def video_decode(path):
    # Decode video/image with ffmpeg
    vid = imageio.get_reader(path, 'ffmpeg')
    return vid


def audio_decode(video_path, audio_path):
    command = "ffmpeg -y -i %s -ab 160k -ac 2 -ar 44100 -vn %s" % (video_path, audio_path)
    # Extract audio from video file through ffmpeg
    print("Extracting audio content from video...")
    return subprocess.call(command, shell=True)


def audio_postproc(video, audio_response, fr):
    # Save audio response
    filename = str(video.video_file).split('/')[-1].split('.')[0] + "_audio.json"
    path = osp.join(settings.MEDIA_ROOT, "json", filename)
    with open(path, "w") as fw:
        json.dump(audio_response, fw)
    video.audio_json = path
    video.fr = fr
    return


def audio_process(video, video_path, models_to_apply):
    audio_name = str(video.video_file).split("/")[-1].split(".")[0] + ".wav"
    audio_path = osp.join("audios", audio_name)
    video.audio_path = audio_path
    abs_audio_path = osp.join(settings.MEDIA_ROOT, audio_path)
    # Extract wav file
    audio_decode(video_path, abs_audio_path)
    # rpc call
    print("Processing wav file through rpc call...")
    with RpcClient(port="50052") as rpc_client:
        audio_response, fr = rpc_client.service_request(abs_audio_path.encode(), models_to_apply)
        audio_response = json.loads(audio_response)
        fr = int(fr)
    # Save audio result
    audio_postproc(video, audio_response, fr)
    return audio_response


def index_scene_feature(rpc_client, video, img, frame, idx, middle_time, scene_list):
    # Create new scene
    scene = Scene()
    scene.video = video
    scene.start_time = middle_time[idx][0]
    scene.end_time = middle_time[idx][1]
    scene.img_path = osp.join(
        "images/retrieved/",
        str(video.video_file).split("/")[-1].split(".")[0] + "_" + str(idx) + ".jpg"
    )
    cv2.imwrite(osp.join(settings.MEDIA_ROOT, scene.img_path), img)
    if frame is not None:
        scene.frame = frame

    # Scene feature extraction
    _, img_encoded = cv2.imencode('.jpg', img)
    buf = img_encoded.tobytes()
    scene_feature, scene_name = rpc_client.service_request(buf, "scene")
    scene_feature = np.array(json.loads(scene_feature)["features"])
    scene.img_feature = scene_feature.tobytes()
    scene.scene_name = scene_name
    # Audio feature extraction
    audio_feature, _ = rpc_client.service_request(
        osp.join(settings.MEDIA_ROOT, video.audio_path).encode(),
        "audio,%d,%d" % (scene.start_time, scene.end_time)
    )
    audio_feature = np.array(json.loads(audio_feature)["features"])
    scene.audio_feature = audio_feature.tobytes()
    # Subtitle feature extraction
    subtitle_feature = None
    if str(video.subtitle_file):
        subtitle_path = osp.join(settings.MEDIA_ROOT, str(video.subtitle_file))
        subtitle_feature, sentence = rpc_client.service_request(
            subtitle_path.encode(),
            "subtitle,%d,%d" % (scene.start_time, scene.end_time)
        )
        subtitle_feature = np.array(json.loads(subtitle_feature)["features"])
        scene.subtitle_feature = subtitle_feature.tobytes()
        scene.subtitle = sentence
    scene.save()
    scene_list.append({
        "SCENE_ID": scene.pk,
        "TV_NAME": str(video.video_file),
        "START_TIME": scene.start_time,
        "END_TIME": scene.end_time,
        "SCENE": scene.scene_name,
        "IMAGE": scene.img_path,
        "FEATURE": scene_feature,
        "SUBTITLE": scene.subtitle,
        "SUBTITLE_FEATURE": subtitle_feature if subtitle_feature else "unknown_feature",
        "AUDIO": scene.video.audio_path,
        "AUDIO_FEATURE": audio_feature,
    })
    return


def visual_postproc(video, video_path, json_response, splits_frame, splits_ms):
    middle_frame = list()
    middle_time = {}

    for i in range(len(splits_ms) - 1):
        temp = math.floor((splits_frame[i] + splits_frame[i + 1]) / 2.0)
        middle_frame.append(temp)
        middle_time[temp] = [splits_ms[i], splits_ms[i + 1]]

    decoder = PyDecoder.Decoder(settings.DECODING_HARDWARE)
    decoder.ingestVideo(video_path)
    decoder.decode()

    cur_scene_idx = -1
    cur_scene_end = -1
    scene_end = video.frame_cnt

    statistics = list()
    scene_list = list()
    # Instantiate feature client to avoid repetitive connection establishment
    rpc_client = RpcClient(port="50055")
    with tqdm.tqdm(total=scene_end, unit="frames") as pbar:
        # Bulk saving to increase speed
        with transaction.atomic():
            try:
                frame_idx = 0
                while True:
                    img = decoder.fetchFrame()
                    if img.size == 0:
                        break
                    # Generate json path wrt video path
                    filename = video_path.split('/')[-1].split('.')[0] + '_' + str(frame_idx) + '.json'
                    json_path = osp.join(settings.MEDIA_ROOT, 'json', filename)
                    decoded = json_response[frame_idx]
                    # Save json
                    with open(json_path, 'w') as fw:
                        json.dump(decoded, fw)
                    frame_serializer = FrameSerializer(data={"video": video.pk, 'json_path': json_path})
                    frame = None
                    if frame_serializer.is_valid():  # TODO: this does not work when json_path (aka. video name is
                                                     # longer than certain characters
                        frame = frame_serializer.save()
                    # Switch scene
                    if frame_idx > cur_scene_end:
                        cur_scene_idx += 1
                        cur_scene_end = splits_frame[cur_scene_idx + 1] if cur_scene_idx < len(
                            splits_frame) - 1 else scene_end - 1
                        statistics.append({
                            "start_frame": splits_frame[cur_scene_idx],
                            "end_frame": cur_scene_end,
                            "cur_scene_statistics": {}
                        })
                    # Update statistics
                    if decoded.get("detection_classes_names"):
                        for cls in decoded["detection_classes_names"]:
                            if cls in statistics[cur_scene_idx]["cur_scene_statistics"]:
                                statistics[cur_scene_idx]["cur_scene_statistics"][cls] += 1
                            else:
                                statistics[cur_scene_idx]["cur_scene_statistics"][cls] = 1

                    # Extract feature
                    if frame_idx in middle_frame:
                        # Create new scene
                        index_scene_feature(rpc_client, video, img, frame, frame_idx, middle_time, scene_list)
                    # Update pbar
                    pbar.update(1)
                    frame_idx += 1

            except imageio.core.format.CannotReadFrameError:
                print("io error caught")
                pass

    # Close feature client
    rpc_client.close()
    # Save the pickle file
    pkl_name = str(video.video_file).split("/")[-1].split(".")[0] + "_index.pkl"
    pkl_path = osp.join(settings.MEDIA_ROOT, "multi_features", pkl_name)
    with open(pkl_path, "wb") as f:
        pickle.dump(scene_list, f)
    video.pkl_path = pkl_path
    # Save the statistics
    statistics_name = str(video.video_file).split("/")[-1].split(".")[0] + "_statistics.json"
    statistics_path = osp.join(settings.MEDIA_ROOT, "statistics/" + statistics_name)
    with open(statistics_path, "w") as f:
        json.dump(statistics, f)
    video.statistics = statistics_path
    return


def visual_process(video, video_path, models_to_apply):
    # Process visual content
    # Read frame count through imageio
    # TODO: add get_frame_cnt function to Hysia decoder
    with imageio.get_reader(video_path, "ffmpeg") as vid:
        frame_cnt = vid.count_frames()
        video.frame_cnt = frame_cnt

    # Initiate Hysia GPU decoder
    decoder = PyDecoder.Decoder(settings.DECODING_HARDWARE)
    decoder.ingestVideo(video_path)
    # Start a decoding thread
    decoder.decode()
    # rpc call
    print("Processing frames through rpc call...")
    visual_responses = list()

    with tqdm.tqdm(total=frame_cnt, unit="frames") as pbar:
        with RpcClient(port="50051") as rpc_client:
            while True:
                frame = decoder.fetchFrame()
                # decoder returns empty frame once it reaches the end of video
                if frame.size == 0:
                    break
                _, frame_encoded = cv2.imencode('.jpg', frame)
                json_string, _ = rpc_client.service_request(buf=frame_encoded.tobytes(), meta=models_to_apply)
                visual_responses.append(json.loads(json_string))
                pbar.update(1)

    # Get scene splits
    print("Splitting scenes...")
    shot_detector = Shot_Detector()
    splits_frame, splits_ms = shot_detector.detect(video_path)
    # Processing statistics and save frames
    print("Computing statistics, extracting features and saving frame results...")
    visual_postproc(video, video_path, visual_responses, splits_frame, splits_ms)
    return visual_responses


# Process the video
def process(video):
    """

    :param
        video: The video object that represents a model instance
    :return:
        res: Json object that contains the prediction results for both visual and audio processing

    """

    video.being_processed = True
    video.save()
    try:
        res = dict()
        # Get absolute video path
        video_path = osp.join(settings.MEDIA_ROOT, str(video.video_file))

        # Get all models attached to video
        models_to_apply = ",".join([model.name for model in video.dlmodels.all()])
        # Process audio content
        res["audio_results"] = audio_process(video, video_path, models_to_apply)
        # Process visual content
        res["visual_results"] = visual_process(video, video_path, models_to_apply)
        # Mark video as processed and save
        video.processed = True
        video.being_processed = False
        video.save()
        return res
    # Capture any exception and return None
    except Exception as e:
        video.being_processed = False
        video.save()
        print(e)
        return None

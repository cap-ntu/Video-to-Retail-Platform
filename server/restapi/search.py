"""
Hysia image reverse search backend logic
Author: Jiang Wenbo
Email: jian0085@e.ntu.edu.sg
"""
import json
import os
import os.path as osp
from shutil import copyfile

from django.conf import settings

from restapi.models import Scene
from restapi.rpc_client import RpcClient


def search_scene(query_img, target_videos):
    try:
        img_abs_path = osp.join(settings.MEDIA_ROOT, str(query_img.img))
        # Request rpc service
        with RpcClient(50053) as rpc_client:
            # Request meta data
            meta = json.dumps({"text": query_img.text, "target_videos": target_videos})
            response, _ = rpc_client.service_request(img_abs_path.encode(), meta)
            response = json.loads(response)

        src_path_list = [res["IMAGE"] if res["IMAGE"].startswith("/")
                         else osp.join(settings.MEDIA_ROOT, res["IMAGE"]) for res in response]
        # Image path searchable by the server
        dst_file_list = [osp.join("images/temp", "_".join([str(query_img.owner.username), "top", str(i + 1)])
                         + ".jpg") for i, _ in enumerate(response)]
        # Group the search result by video name
        grouped = list()
        name2idx = dict()
        for i in range(len(response)):
            # Change the image path in response to point to the destination
            response[i]["IMAGE"] = dst_file_list[i]
            # Change key name to make it accessible by django template language
            if response[i].get("START TIME"):
                response[i]["START_TIME"] = response[i]["START TIME"]
                response[i]["END_TIME"] = response[i]["END TIME"]
            # TODO re_index TVQA
            response[i]["TV_NAME"] = response[i]["TV_NAME"].split("/")[-1]
            # Perform grouping
            if response[i]["TV_NAME"] in name2idx:
                grouped[name2idx[response[i]["TV_NAME"]]].append(response[i])
            else:
                grouped.append([response[i]])
                name2idx[response[i]["TV_NAME"]] = len(grouped) - 1

            # Add scenes highest price
            response[i]["HIGHEST_PRICE"] = Scene.objects.get(pk=int(response[i]["SCENE_ID"])).highest_price

        # Sort according to group size
        grouped.sort(key=lambda x: len(x), reverse=True)
        # Absolute image list
        dst_path_list = [osp.join(settings.MEDIA_ROOT, f) for f in dst_file_list]
        # Copy image from path outside the project to searchable path
        for src, dst in zip(src_path_list, dst_path_list):
            copyfile(src, dst)
        return grouped

    # Capture any exception and return None
    except Exception as e:
        return None


def search_product(time_stamp, video_name, host):
    retrieved_path = "images/product/retrieved"
    try:
        video_path = osp.join(settings.MEDIA_ROOT, video_name)
        # Delete images from previous search
        for img in os.listdir(osp.join(settings.MEDIA_ROOT, retrieved_path)):
            if img.endswith(".jpg"):
                os.remove(osp.join(settings.MEDIA_ROOT, retrieved_path, img))

        # Request rpc service
        with RpcClient(50054) as rpc_client:
            response, _ = rpc_client.service_request(str(time_stamp).encode(), video_path)
            response = json.loads(response)

        src_path_list = [res["IMAGE"] for res in response]
        # Image path searchable by the server
        dst_file_list = [osp.join(retrieved_path, "_".join([
            video_name.split("/")[-1].split(".")[0],
            str(time_stamp),
            "top",
            str(i + 1)
        ]) + ".jpg") for i, _ in enumerate(response)]

        for i in range(len(response)):
            # Change the image path in response to point to the destination
            response[i]["IMAGE"] = "http://" + host + osp.join(settings.MEDIA_URL, dst_file_list[i])

        dst_path_list = [osp.join(settings.MEDIA_ROOT, f) for f in dst_file_list]
        # Copy image from path outside the project to searchable path
        for src, dst in zip(src_path_list, dst_path_list):
            copyfile(src, dst)
        return response

    # Capture any exception and return None
    except Exception as e:
        print(e)
        return None

# Create your views here.
# Rest framework
import json
# Other imports
import os
import time

# Django imports
from django.conf import settings
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.db.models import Q
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

# Module imports
from restapi.models import Video, Image, DlModel, Scene
from restapi.processing import process
from restapi.search import search_product
from restapi.search import search_scene
from restapi.serializers import VideoSerializer, DlModelSerializer, UserSerializer, ImageSerializer
from restapi.utils import attach_models


#######################################
# Below are rest APIs for android app #
#######################################


# Get demo video for android app demo
class GetDemoVideoUrl(APIView):
    renderer_classes = (JSONRenderer,)

    def get(self, request):
        videos = Video.objects.all()
        # Pick the first video as the demo video
        demo_video = videos[0]
        demo_video_url = "http://" + request.get_host() + os.path.join(settings.MEDIA_URL, str(demo_video.video_file))
        return Response({
            "demo_video_url": demo_video_url,
            "demo_video_name": str(demo_video.video_file),
        })


# Get videos for a particular user
class GetUserVideoUrls(APIView):
    renderer_classes = (JSONRenderer,)
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        videos = request.user.videos.all()
        response = {"videos": []}
        for video in videos:
            item = dict()
            item["video_url"] = "http://" + request.get_host() + os.path.join(settings.MEDIA_URL, str(video.video_file))
            item["video_name"] = str(video.video_file)
            item["screen_shot"] = "http://" + request.get_host() + \
                                  os.path.join(settings.MEDIA_URL, video.scenes.all()[0].img_path)
            products = list()
            for scene in video.scenes.all():
                if scene.inserted_ad:
                    products.append(
                        {
                            "product_url": "http://" + request.get_host() + os.path.join(settings.MEDIA_URL,
                                                                                         scene.inserted_ad.img),
                            "start_time": scene.start_time,
                            "end_time": scene.end_time,
                        }
                    )
            item["products"] = products
            response["videos"].append(item)

        return Response(response)


# Process product search request
class SearchProduct(APIView):
    renderer_classes = (JSONRenderer,)

    def post(self, request):
        time_stamp = request.data["time_stamp"]
        video_name = request.data["video_url"]
        host = request.get_host()
        products = search_product(time_stamp, video_name, host)
        if products is None:
            return Response({"error": "invalid request"}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"products": products})


####################################################
# Below is the rest API for table retrieval request#
####################################################


class RetrieveTable(APIView):
    renderer_classes = (JSONRenderer,)

    def get(self, request, pk):
        video = Video.objects.get(pk=pk)
        content = list()
        for frame in video.frames.all():
            with open(str(frame.json_path), "r") as f:
                content.append(json.load(f))

        # Write the json list to an entire json file
        table_name = str(video.video_file).split("/")[-1].split(".")[0] + ".json"
        table_path = os.path.join(settings.MEDIA_ROOT, "tables", table_name)
        with open(table_path, "w") as f:
            json.dump(content, f)
        video.table = table_path
        return Response({"time": time.time(), "data": "success"})


##############################################
# Below are rest APIs for dashboard in react #
##############################################


# Logout view
class Logout(APIView):
    permission_classes = (IsAdminUser,)
    renderer_classes = (JSONRenderer,)

    def get(self, request):
        # simply delete the token to force a login
        # request.user.auth_token.delete()
        logout(request)
        return Response({"time": time.time(), "data": "success"}, status=status.HTTP_200_OK)


# Video view set has support all type of HTTP requests
class VideoViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAdminUser,)
    serializer_class = VideoSerializer
    queryset = Video.objects.all()
    renderer_classes = (JSONRenderer,)

    @parser_classes((FileUploadParser, MultiPartParser, FormParser,))
    def create(self, request, *args, **kwargs):
        print(request.data)
        error_msg = None
        serializer = VideoSerializer(data=request.data)
        json_response = None

        if serializer.is_valid():
            video = serializer.save(owner=request.user)
            if request.data.get("if_process"):

                attach_models(request, video)

                response = process(video)
                if response is None:
                    # Delete video in case of exception
                    video.delete()
                    error_msg = "Wrong media format or model servers are down!"
                else:
                    json_response = {
                        "time": time.time(),
                        "data": "success"
                    }
            else:
                json_response = {
                    "time": time.time(),
                    "data": "success"
                }
        else:
            error_msg = "Invalid form data!"

        if error_msg:
            return Response({"time": time.time(), "data": error_msg}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(json_response)

    def list(self, request, *args, **kwargs):
        response = super().list(request, *args, **kwargs)
        # Get cover image
        for video in response.data:
            scenes = Video.objects.get(pk=video["id"]).scenes.all()
            if len(scenes):
                video["cover"] = "http://" + request.get_host() + os.path.join(settings.MEDIA_URL, scenes[0].img_path)
            else:
                video["cover"] = ""

        return Response({
            "time": time.time(),
            "data": response.data,
        })

    def retrieve(self, request, *args, **kwargs):
        video = self.get_object()
        response = dict()
        response["time"] = time.time()
        data = dict()
        data["url"] = "http://" + request.get_host() + os.path.join(settings.MEDIA_URL, str(video.video_file))
        # Default frame count
        data["frame_cnt"] = 1000
        data["being_processed"] = video.being_processed
        data["result"] = dict()
        if video.processed:
            data["audio_url"] = "http://" + request.get_host() + os.path.join(settings.MEDIA_URL, str(video.audio_path))
            # Get video frame count
            data["frame_cnt"] = video.frame_cnt
            # Get visual results in json
            visual_res = list()
            for frame in video.frames.all():
                with open(str(frame.json_path), "r") as f:
                    visual_res.append(json.load(f))

            data["result"]["visual"] = visual_res

            # Get audio results in json
            if str(video.audio_json) != "":
                with open(str(video.audio_json)) as f:
                    data["result"]["audio"] = json.load(f)
                    if not data["result"]["audio"]:
                        data["result"]["audio"] = list()

            # Get statistics
            if str(video.statistics) != "":
                with open(str(video.statistics), "r") as f:
                    data["result"]["statistics"] = json.load(f)

            # Add models used
            data["models"] = [model.name for model in video.dlmodels.all()]

            # Get inserted products
            data["products"] = list()
            for scene in video.scenes.all():
                if scene.inserted_ad:
                    # Get the first scene as cover image
                    data["products"].append(
                        {
                            "product_path": "http://" + request.get_host() + os.path.join(settings.MEDIA_URL,
                                                                                          str(scene.inserted_ad.img)),
                            "start_time": scene.start_time,
                            "end_time": scene.end_time,
                        }
                    )

        data["processed"] = video.processed
        response["data"] = data
        return Response(response)

    def partial_update(self, request, *args, **kwargs):
        video = self.get_object()
        response = dict()
        response["time"] = time.time()
        # Reprocess
        if video.processed:
            # Delete all related objects
            video.delete_all_related()
        attach_models(request, video)
        # Process video with rpc
        response = process(video)
        if response is None:
            response["data"] = "Wrong media format or model servers are down!"
            return Response(response, status=status.HTTP_400_BAD_REQUEST)
        else:
            response["data"] = "success"
            return Response(response)

    def destroy(self, request, *args, **kwargs):
        super().destroy(request, *args, **kwargs)
        return Response({"time": time.time(), "data": "success"})


# Dlmodel view set has support all type of HTTP requests
class DlModelViewSet(viewsets.ModelViewSet):
    queryset = DlModel.objects.all()
    serializer_class = DlModelSerializer
    permission_classes = (IsAdminUser,)
    renderer_classes = (JSONRenderer,)

    def list(self, request, *args, **kwargs):
        response = super().list(request, *args, **kwargs)
        return Response({
            "time": time.time(),
            "data": response.data
        })

    def get_default_dlmodels(self, request, *args, **kwargs):
        default_dlmodels = DlModel.objects.filter(
            Q(name="SSD-inception") | Q(name="mtcnn") | Q(name="ctpn") | Q(name="res18-places365")
        )
        serializer = DlModelSerializer(default_dlmodels, many=True)
        return Response({
            "time": time.time(),
            "data": serializer.data,
        })

    def retrieve(self, request, *args, **kwargs):
        dlmodel = self.get_object()
        serializer = DlModelSerializer(instance=dlmodel)
        return Response({
            "time": time.time(),
            "data": serializer.data,
        })

    def destroy(self, request, *args, **kwargs):
        super().destroy(request, *args, **kwargs)
        return Response({"time": time.time(), "data": "success"})


# User view set has support all type of HTTP requests
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (IsAdminUser,)
    renderer_classes = (JSONRenderer,)

    def list(self, request, *args, **kwargs):
        response = super().list(request, *args, **kwargs)
        return Response({
            "time": time.time(),
            "data": response.data
        })

    def retrieve(self, request, *args, **kwargs):
        user = self.get_object()
        serializer = UserSerializer(instance=user)
        return Response({
            "time": time.time(),
            "data": serializer.data,
        })

    def destroy(self, request, *args, **kwargs):
        super().destroy(request, *args, **kwargs)
        return Response({"time": time.time(), "data": "success"})


# Handles scene search requests
class SceneSearch(APIView):
    renderer_classes = (JSONRenderer,)

    def post(self, request):
        target_videos = request.data.getlist("target_videos")
        if target_videos:
            # Take the file name
            target_videos = [
                str(Video.objects.get(pk=int(video_id)).video_file).split("/")[-1].split(".")[0]
                for video_id in target_videos
            ]
        else:
            target_videos = list()

        serializer = ImageSerializer(data=request.data)
        error_msg = None
        img_group_list = None
        if serializer.is_valid():
            # Delete previous image if it's not inserted
            prev_imgs = request.user.images.all()
            for img in prev_imgs:
                if not img.inserted:
                    img.delete()

            query_img = serializer.save(owner=request.user)

            # Search similar images in database
            img_group_list = search_scene(query_img, target_videos)
            if img_group_list is None:
                error_msg = "Wrong media format or model servers are down!"
            else:
                for group in img_group_list:
                    for img in group:
                        img["IMAGE"] = "http://" + request.get_host() + os.path.join(settings.MEDIA_URL, img["IMAGE"])
        else:
            error_msg = "Invalid form data"

        if error_msg:
            return Response({"time": time.time(), "data": error_msg}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({"time": time.time(), "data": img_group_list})


# Handles product insert requests
class ProductInsert(APIView):
    renderer_classes = (JSONRenderer,)

    def post(self, request):
        if request.data.get("scene_id"):
            scene_id = int(request.data["scene_id"])
            scene = Scene.objects.get(pk=scene_id)
            # Get the latest target product uploaded by the user
            query_img = Image.objects.filter(owner=request.user).order_by("-pk")[0]
            # Mark as inserted
            query_img.inserted = True
            query_img.save()
            scene.inserted_ad = query_img
            scene.save()
            return Response({"time": time.time(), "data": "success"})
        else:
            return Response({"time": time.time(), "data": "scene_id not provided"}, status=status.HTTP_400_BAD_REQUEST)

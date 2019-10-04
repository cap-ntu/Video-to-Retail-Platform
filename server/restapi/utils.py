import os
from channels.db import database_sync_to_async
from restapi.models import DlModel, Scene, Image
from django.contrib.auth.models import User


def get_all_dlmodels():
    object_models = DlModel.objects.filter(type=DlModel.OBJECT)
    face_models = DlModel.objects.filter(type=DlModel.FACE)
    text_models = DlModel.objects.filter(type=DlModel.TEXT)
    scene_models = DlModel.objects.filter(type=DlModel.SCENE)
    return {
        "object_models": object_models,
        "face_models": face_models,
        "text_models": text_models,
        "scene_models": scene_models
    }


# Attach models from request to video
def attach_models(request, video):
    # Check object models
    for model in request.data.getlist("models"):
        video.dlmodels.add(DlModel.objects.get(name=model))


# Attach default models to video
def attach_default_models(video):
    # Attach default object models
    video.dlmodels.add(DlModel.objects.get(name="SSD-inception"))
    # Attach default face models
    video.dlmodels.add(DlModel.objects.get(name="mtcnn"))
    # Attach default text models
    video.dlmodels.add(DlModel.objects.get(name="ctpn"))
    # Attach default scene models
    video.dlmodels.add(DlModel.objects.get(name="res18-places365"))
    return


@database_sync_to_async
def get_scene(pk):
    return Scene.objects.get(pk=pk)


@database_sync_to_async
def save_scene(scene):
    scene.save()


@database_sync_to_async
def get_user(username):
    return User.objects.get(username=username)


@database_sync_to_async
def get_latest_upload(user):
    return Image.objects.filter(owner=user).order_by("-pk")[0]


@database_sync_to_async
def mark_as_inserted(upload):
    upload.inserted = True
    upload.save()


@database_sync_to_async
def save_upload(upload):
    upload.save()

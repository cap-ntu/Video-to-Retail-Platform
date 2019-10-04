from rest_framework import serializers
from restapi.models import Video, Frame, DlModel, Image
from django.contrib.auth.models import User


class DlModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = DlModel
        fields = "__all__"


class VideoSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')
    dlmodels = DlModelSerializer(read_only=True, many=True)

    class Meta:
        model = Video
        fields = "__all__"


class FrameSerializer(serializers.ModelSerializer):
    class Meta:
        model = Frame
        fields = ['video', 'json_path']


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = "__all__"


class ImageSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')

    class Meta:
        model = Image
        fields = ["owner", "img", "text"]

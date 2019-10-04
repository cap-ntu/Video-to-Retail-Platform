from __future__ import unicode_literals
from django.db import models

# Create your models here.
# -*- coding: utf-8 -*-

from django.db import models
from django.db import transaction
from django.conf import settings
from django.utils import timezone
import os.path as osp
import os
from datetime import datetime, timedelta
# Create your models here.


def try_remove(file_path):
    try:
        os.remove(file_path)
    except:
        pass


# Model object that represents a pre-trained deep learning model
class DlModel(models.Model):
    OBJECT = "OBJECT"
    FACE = "FACE"
    TEXT = "TEXT"
    SCENE = "SCENE"
    CHOICES = (
        (OBJECT, "object"),
        (FACE, "face"),
        (TEXT, "text"),
        (SCENE, "scene"),
    )
    # Type
    type = models.TextField(max_length=20, choices=CHOICES, default=OBJECT)
    # Unique name
    name = models.CharField(max_length=100, blank=False, null=False, default='', unique=True)
    # Frozen graph
    model = models.FileField(upload_to='models/', blank=True, null=True)
    # Label file
    label = models.FileField(upload_to='labels/', blank=True, null=True)

    # Custom delete method to make sure the related files are deleted upon model deletion
    def delete(self, *args, **kwargs):
        self.model.delete(save=False)
        self.label.delete(save=False)
        super().delete(*args, **kwargs)


# Video object that represent a video uploaded
class Video(models.Model):
    # Created time
    created = models.DateTimeField(auto_now_add=True)
    # Unique label for the video
    name = models.CharField(max_length=100, blank=False, null=False, default='')
    # Video file
    video_file = models.FileField(upload_to='videos/', blank=False, null=False)
    # Subtitle file
    subtitle_file = models.FileField(upload_to='subtitles/', blank=True, null=True, default=None)
    # Owner (Django default user model)
    owner = models.ForeignKey('auth.User', related_name='videos', on_delete=models.CASCADE, verbose_name='')
    # Path to its wav file
    audio_path = models.CharField(max_length=100, default='')
    # Frame count
    frame_cnt = models.IntegerField(default=0)
    # Audio processing frame rate
    fr = models.IntegerField(default=0)
    # Path to audio results (absolute path)
    audio_json = models.CharField(max_length=100, default='')
    # Path to its statistics (absolute path)
    statistics = models.CharField(max_length=100, default='')
    # Associated models
    dlmodels = models.ManyToManyField(DlModel)
    # If processed
    processed = models.BooleanField(default=False)
    # If processing is in progress
    being_processed = models.BooleanField(default=False)
    # Path to aggregated pickle file (absolute path)
    pkl_path = models.CharField(max_length=100, default='')
    # Path to the entire prediction table (absolute path)
    table = models.CharField(max_length=100, default='')

    # Custom delete method to make sure the related files are deleted upon model deletion
    def delete(self, *args, **kwargs):
        self.video_file.delete(save=False)
        if self.subtitle_file:
            self.subtitle_file.delete(save=False)
        # Delete related frames
        self.delete_all_related()
        super().delete(*args, **kwargs)

    # Delete all related fields in video
    def delete_all_related(self):
        # Delete all related objects
        # Bulk deletion to speed up
        with transaction.atomic():
            for frame in self.frames.all():
                frame.delete()
            for scene in self.scenes.all():
                scene.delete()
        self.dlmodels.clear()
        try_remove(str(self.statistics))
        try_remove(osp.join(settings.MEDIA_ROOT, str(self.audio_path)))
        try_remove(self.audio_json)
        try_remove(self.pkl_path)

    class Meta:
        ordering = ('created',)


# Image object for searching
class Image(models.Model):
    # Owner
    owner = models.ForeignKey('auth.User', related_name="images", on_delete=models.CASCADE, verbose_name='')
    # Query image
    img = models.FileField(upload_to='images/uploaded', blank=False, null=False)
    # Query text
    text = models.TextField(max_length=None, blank=False, null=False)
    # Indicating if the query image is inserted into some video or not
    # If it's inserted, it should not be cleared
    inserted = models.BooleanField(default=False)

    # Custom delete method to make sure the related file is deleted upon model deletion
    def delete(self, *args, **kwargs):
        self.img.delete(save=False)
        super().delete(*args, **kwargs)


# Frame object that stores analysing results
class Frame(models.Model):
    # Foreign key
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='frames')
    # Path to json results (absolute path)
    json_path = models.CharField(max_length=100, blank=False, null=False, default='', unique=True)

    # Custom delete method to make sure the related file is deleted upon model deletion
    def delete(self, *args, **kwargs):
        try_remove(str(self.json_path))
        super().delete(*args, **kwargs)


# Video scenes for searching and bidding
class Scene(models.Model):
    # Foreign key
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='scenes')
    # Start time
    start_time = models.IntegerField(default=0)
    # End time
    end_time = models.IntegerField(default=0)
    # Scene name
    scene_name = models.CharField(max_length=20, default="")
    # Image path
    img_path = models.CharField(max_length=100, default="")
    # Image feature encoded from numpy array
    img_feature = models.BinaryField(max_length=None)
    # Subtitle
    subtitle = models.TextField(max_length=None)
    # Subtitle Feature encoded from numpy array
    subtitle_feature = models.BinaryField(max_length=None)
    # Object json path encoded from numpy array
    frame = models.OneToOneField(Frame, on_delete=models.DO_NOTHING)
    # Refers to the inserted advertisement
    inserted_ad = models.OneToOneField(Image, null=True, on_delete=models.DO_NOTHING)
    # Audio feature encoded from numpy array
    audio_feature = models.BinaryField(max_length=None)
    # Highest bidding price
    highest_price = models.IntegerField(default=0)
    # Bidding start time
    bidding_start_time = models.DateTimeField(auto_now=True)
    # duration
    duration = models.DurationField(default=timedelta(days=1))

    @property
    def in_progress(self):
        if self.bidding_start_time + self.duration > datetime.now():
            return True
        else:
            return False

    # Custom delete method to make sure the related file is deleted upon model deletion
    def delete(self, *args, **kwargs):
        try_remove(osp.join(settings.MEDIA_ROOT, str(self.img_path)))
        super().delete(*args, **kwargs)

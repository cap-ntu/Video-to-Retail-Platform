from django import forms
from restapi.models import Video, DlModel, Image
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .processing import process


class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ('name', 'video_file', 'subtitle_file')


class SignUpForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'email',)


class DlModelForm(forms.ModelForm):
    class Meta:
        model = DlModel
        fields = ('model', 'label',)


class ImageForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add id name and attributes
        self.fields['img'].widget.attrs.update({
            'id': 'file-input',
            'accept': 'image/*',
            'onchange': 'loadImage(event)'
        })
        self.fields['text'].widget.attrs.update({'id': 'textarea-input'})

    class Meta:
        model = Image
        fields = ('img', 'text')

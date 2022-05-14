from django import forms
from .models import Scan


class ImageForm(forms.ModelForm):

    class Meta:
        model = Scan
        fields = ('scan',)
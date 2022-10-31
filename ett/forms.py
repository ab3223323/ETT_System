from django import forms
from .models import *
class UploadModelForm(forms.ModelForm):
    class Meta:
        model = pic
        fields = ['file']
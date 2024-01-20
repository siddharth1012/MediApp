from django import forms
from core_app.models import ImageRecognition

class ImageRecognitionForm(forms.ModelForm):
    
    class Meta:
        model = ImageRecognition
        fields = ['image']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.fields['image'].widget.attrs.update({'class':'form-control'})
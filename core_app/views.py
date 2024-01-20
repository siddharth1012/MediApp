from django.shortcuts import render
from django.http import HttpResponse
from core_app.forms import ImageRecognitionForm
from core_app.machinelearning import pipeline_model
from django.conf import settings
from core_app.models import ImageRecognition
import os

MEDIA_ROOT = settings.MEDIA_ROOT

def index(request):
    return render(request, 'index.html')

def malaria(request):
    form = ImageRecognitionForm()
    
    if request.method == 'POST':
        form = ImageRecognitionForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            primary_key = save.pk
            imageobj = ImageRecognition.objects.get(pk=primary_key)
            filepath = str(imageobj.image)
            filepath = os.path.join(MEDIA_ROOT,filepath)
            result = pipeline_model(filepath,"mal")
            print(result)
            return render(request, 'malaria.html', {'form':form, 'upload':True, 'result':result})
            
    return render(request, 'malaria.html', {'form':form, 'upload':False})

def brain_tumor(request):
    form = ImageRecognitionForm()
    
    if request.method == 'POST':
        form = ImageRecognitionForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            primary_key = save.pk
            imageobj = ImageRecognition.objects.get(pk=primary_key)
            filepath = str(imageobj.image)
            filepath = os.path.join(MEDIA_ROOT,filepath)
            result = pipeline_model(filepath,"brain")
            print(result)
            return render(request, 'brain_tumor.html', {'form':form, 'upload':True, 'result':result})
            
    return render(request, 'brain_tumor.html', {'form':form, 'upload':False})

def oct(request):
    form = ImageRecognitionForm()
    
    if request.method == 'POST':
        form = ImageRecognitionForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            primary_key = save.pk
            imageobj = ImageRecognition.objects.get(pk=primary_key)
            filepath = str(imageobj.image)
            filepath = os.path.join(MEDIA_ROOT,filepath)
            result = pipeline_model(filepath,"oct")
            print(result)
            return render(request, 'oct.html', {'form':form, 'upload':True, 'result':result})
            
    return render(request, 'oct.html', {'form':form, 'upload':False})

def breast_cancer(request):
    form = ImageRecognitionForm()
    
    if request.method == 'POST':
        form = ImageRecognitionForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            primary_key = save.pk
            imageobj = ImageRecognition.objects.get(pk=primary_key)
            filepath = str(imageobj.image)
            filepath = os.path.join(MEDIA_ROOT,filepath)
            result = pipeline_model(filepath,"breast")
            print(result)
            return render(request, 'breast_cancer.html', {'form':form, 'upload':True, 'result':result})
            
    return render(request, 'breast_cancer.html', {'form':form, 'upload':False})

def diabetes(request):
    form = ImageRecognitionForm()
    
    if request.method == 'POST':
        form = ImageRecognitionForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            primary_key = save.pk
            imageobj = ImageRecognition.objects.get(pk=primary_key)
            filepath = str(imageobj.image)
            filepath = os.path.join(MEDIA_ROOT,filepath)
            result = pipeline_model(filepath,"dia_ret")
            print(result)
            return render(request, 'diabetes.html', {'form':form, 'upload':True, 'result':result})
            
    return render(request, 'diabetes.html', {'form':form, 'upload':False})

def detect(request):
    return render(request, 'detect.html')

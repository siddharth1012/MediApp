import numpy as np
import cv2
from cv2 import resize, INTER_AREA
from PIL import Image
import os
from keras.models import load_model
from keras.preprocessing import image
from keras.layers import Flatten
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
from django.conf import settings

STATIC_DIR = settings.STATIC_DIR

def get_model(name = None):
    model_name = []
    if(name=='mal'):
        model_name.append({"model": load_model(os.path.join(STATIC_DIR,"weights/malaria.h5")), "type": name})
    elif(name=='brain'):
        model_name.append({"model": load_model(os.path.join(STATIC_DIR,"weights/brain_tumor.h5")), "type": name})
    elif(name=='oct'):
        model_name.append({"model": load_model(os.path.join(STATIC_DIR,"weights/retina_OCT.h5")), "type": name})
    elif(name=='dia_ret'):
        model_name.append({"model": load_model(os.path.join(STATIC_DIR,"weights/diabetes_retinopathy.h5")), "type": name})
    elif(name=='breast'):
        model_name.append({"model": load_model(os.path.join(STATIC_DIR,"weights/breastcancer.h5")), "type": name})
    return model_name


def resize_image(image):
    resized_image = cv2.resize(image, (128,128)) #Resize all the images to 128X128 dimensions
    if(len(resized_image.shape)!=3):
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB) #Convert to RGB
    return resized_image

def translate_malaria(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  para_prob="Probability of the cell image to be Parasitized: {:.2f}%".format(y_proba_Class1)
  unifected_prob="Probability of the cell image to be Uninfected: {:.2f}%".format(y_proba_Class0)

  total = para_prob + " " + unifected_prob
  total = [para_prob,unifected_prob]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The cell image shows strong evidence of Malaria."
      return total,prediction
  else:
      prediction="Inference: The cell image shows no evidence of Malaria."
      return total,prediction

"""This function also does the same thing as above. Since it's a two class classification problem, 
we can subtract one probability percentage values from 100 to get the other value."""
def translate_cancer(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  can="Probability of the histopathology image to have cancer: {:.2f}%".format(y_proba_Class1)
  norm="Probability of the histopathology image to be normal: {:.2f}%".format(y_proba_Class0)

  total = [can,norm]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The histopathology image shows strong evidence of Invasive Ductal Carcinoma."
      return total,prediction
  else:
      prediction="Inference: The cell image shows no evidence of Invasive Ductal Carcinoma."
      return total,prediction

"""Tis function will compute the values for the brain cancer model"""
def translate_brain(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  tumor="Probability of the MRI scan to have a brain tumor: {:.2f}%".format(y_proba_Class1)
  normal="Probability of the MRI scan to not have a brain tumor: {:.2f}%".format(y_proba_Class0)

  total = [tumor, normal]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The MRI scan has a brain tumor."
      return total,prediction
  else:
      prediction="Inference: The MRI scan does not show evidence of any brain tumor."
      return total,prediction

"""For multi class problems, we will obtain each of the class probabilities for each of the 
classes. We will send this values to frontend using a jsonfy object. The final jsonfy object will
contain """
def translate_oct(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100
  y_proba_Class3 = preds.flatten().tolist()[3] * 100

  cnv="Probability of the input image to have Choroidal Neo Vascularization: {:.2f}%".format(y_proba_Class0)
  dme="Probability of the input image to have Diabetic Macular Edema: {:.2f}%".format(y_proba_Class1)
  drusen="Probability of the input image to have Hard Drusen: {:.2f}%".format(y_proba_Class2)
  normal="Probability of the input image to be absolutely normal: {:.2f}%".format(y_proba_Class3)

  total = [cnv,dme,drusen,normal]
  
  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
  statements = ["Inference: The image has high evidence of Choroidal Neo Vascularization in the retinal pigment epithelium.",
               "Inference: The image has high evidence of Diabetic Macular Edema in the retinal pigment epithelium.",
               "Inference: The image has high evidence of Hard Drusen in the retinal pigment epithelium.",
               "Inference: The retinal pigment epithelium layer looks absolutely normal."]
  
  
  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction

def translate_retinopathy(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100
  y_proba_Class3 = preds.flatten().tolist()[3] * 100

  mild="Probability of the input image to have Mild Diabetic Retinopathy: {:.2f}%".format(y_proba_Class0)
  mod="Probability of the input image to have Moderate Diabetic Retinopathy: {:.2f}%".format(y_proba_Class1)
  norm="Probability of the input image to be Normal: {:.2f}%".format(y_proba_Class2)
  severe="Probability of the input image to have Severe Diabetic Retinopathy: {:.2f}%".format(y_proba_Class3)

  total = [mild,mod,norm,severe]
  
  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
  statements = ["Inference: The image has high evidence for Mild Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has high evidence for Moderate Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has no evidence for Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has high evidence for Severe Nonproliferative Diabetic Retinopathy Disease."]
  
  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction

def pipeline_model(name,type_):
    # pipeline model
    result = []
    if(type_=="mal" or type_=='brain'):
        test_image = image.load_img(name, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        data=test_image

    elif(type_=='oct'):
        test_image = cv2.imread(name)                  #Read image using the PIL library
        test_image = resize_image(test_image)          #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                  #Convert the image to numpy array
        test_image = test_image/255                        #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)    #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image

    elif(type_=='dia_ret'):
        test_image = Image.open(name)                                  #Read image using the PIL library
        test_image = test_image.resize((128,128), Image.LANCZOS)     #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                              #Convert the image to numpy array
        test_image = test_image/255                                    #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)                #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image

    elif(type_=='breast'):
        test_image = Image.open(name)                                  #Read image using the PIL library
        test_image = test_image.resize((50,50), Image.LANCZOS)        #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                              #Convert the image to numpy array
        test_image = test_image/255                                    #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)                #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image


    model = get_model(type_)[0]

    if(type_=='mal'):
         preds, pred_val = translate_malaria(model["model"].predict(data))
         result = [f"{pred}" for pred in preds] + [f"{pred_val}"]
      
    elif(type_=='brain'):
         preds, pred_val = translate_brain(model["model"].predict(data))
         result = [f"{pred}" for pred in preds] + [f"{pred_val}"]
  
    elif(type_=='breast'):
         preds, pred_val = translate_cancer(model["model"].predict(data))
         result = [f"{pred}" for pred in preds] + [f"{pred_val}"]

    elif(type_=='oct'):
         preds, pred_val = translate_oct(model["model"].predict(data))
         result = [f"{pred}" for pred in preds] + [f"{pred_val}"]

    elif(type_=='dia_ret'):
         preds, pred_val = translate_retinopathy(model["model"].predict(data))
         result = [f"{pred}" for pred in preds] + [f"{pred_val}"]
    else:
      warn = "Feeding blank image won't work. Please enter an input image to continue."
      
    return result
                            
  
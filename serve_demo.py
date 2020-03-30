# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:15:40 2020

@author: Kitt Miller
"""

import torch
import numpy as np
from torchvision import transforms, datasets, models
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import json
import tarfile
import os
import logging
from PIL import Image
import requests

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    logger.info('Loading the model.')
    with open(os.path.join(model_dir, 'model_data.json')) as json_file:
        data = json.load(json_file)
    n_classes = data['num_classes']
    model = models.densenet169(pretrained=False, num_classes = n_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'),map_location=torch.device('cpu') ))
    model.to('cpu')
    logger.info('Done loading model')
    
    return {'model': model, 'data': data}

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
    
def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    
def predict_fn(input_data, model):
    data = model['data']
    model = model['model']
    logger.info('Beginning predict function')
    
    load_trns = transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=data['data_stats']['unlabeled']['mean'],
                                  std=data['data_stats']['unlabeled']['std'])])
    
    myfile = requests.get(input_data['url'])
    open('temp.jpg', 'wb').write(myfile.content)
    image = Image.open('temp.jpg')
    image = load_trns(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    image.to('cpu')
    model.to('cpu').eval()
    pred = model(image)
    index = np.argmax(pred.cpu().data.numpy())
    return str('Prediction: ' + str(data['classes'][index]))
        
        
        
        
        
        
        
        
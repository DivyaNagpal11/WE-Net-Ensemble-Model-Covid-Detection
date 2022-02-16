# -*- coding: utf-8 -*-

'''
This module consists of several popular pretrained models on imagenet weights
   - Additional pre-processing layers have been added for custom input
   - Last FCL layers are truncated and custom layer are introduced.

The following models are imported from this module to implement transfer learning:
    a. vgg16_model, 
    b. resnet_model, 
    c. InceptionV3_model, 
    d. densenet201_model,  
    e. Xception_model
'''


import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras import layers


class TransferLearning:
    
    def __init__(self, image_size, n_classes):
        self.image_size = image_size
        self.n_classes = n_classes
        self.dense = 256
        return
    
    
    def vgg16_model(self):
        vgg16 = VGG16(input_shape=self.image_size + [3], weights='imagenet',
                      include_top=False)
        
        for layer in vgg16.layers:
            layer.trainable = False
        
        new_model = vgg16.layers[-2].output
        new_model = AveragePooling2D(pool_size=(4, 4))(new_model) 
        new_model = Flatten(name="flatten")(new_model)
        new_model = Dense(self.dense, activation="relu")(new_model)
        new_model = Dropout(0.3)(new_model)
        prediction = Dense(len(self.n_classes), activation="softmax")(new_model)
        
        model = Model(inputs=vgg16.input, outputs= prediction)
        return model
    
    
    def resnet_model(self):
        Resnet50 = ResNet50(input_shape=self.image_size + [3],
                            weights='imagenet', include_top=False)

        for layer in Resnet50.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        
        new_model = Resnet50.layers[-2].output
        new_model = AveragePooling2D(pool_size=(4, 4))(new_model)
        new_model = Flatten(name="flatten")(new_model)
        new_model = Dense(self.dense, activation="relu")(new_model)
        new_model = Dropout(0.4)(new_model) #changed from 0.3->0.4 because train acc shows a bit overfit like 0.99 acc
        prediction = Dense(len(self.n_classes), activation="softmax")(new_model)
    
        model = Model(inputs=Resnet50.input, outputs= prediction)
        
        return model
    

    def InceptionV3_model(self):
        baseModel = InceptionV3(input_shape=self.image_size + [3],
                                weights='imagenet', include_top=False)
    
        for layer in baseModel.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True 
            else:
                layer.trainable = False
                    
        last_layer = baseModel.get_layer('mixed7')
        last_output = last_layer.output
        
        new_model = layers.MaxPooling2D(pool_size=(4, 4))(last_output) 
        new_model = layers.Flatten()(new_model)
        new_model = layers.Dense(self.dense, activation='relu')(new_model)
        new_model = layers.Dropout(0.4)(new_model)
        prediction = layers.Dense(len(self.n_classes), activation='softmax')(new_model)
    
        model = Model(inputs = baseModel.input, outputs = prediction)
        return model


    def densenet201_model(self):
        baseModel = DenseNet201(input_shape=self.image_size + [3], 
                                weights='imagenet', include_top=False)

        for layer in baseModel.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(self.dense, activation="relu")(headModel)
        headModel = Dropout(0.4)(headModel)
        headModel = Dense(len(self.n_classes), activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)
        return model
    
    
    def Xception_model(self):
        baseModel = Xception(input_shape=self.image_size + [3],
                             weights='imagenet', include_top=False)

        for layer in baseModel.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True 
            else:
                layer.trainable = False
        
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(self.dense, activation="relu")(headModel)
        headModel = Dropout(0.4)(headModel)
        headModel = Dense(len(self.n_classes), activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)
        
        return model


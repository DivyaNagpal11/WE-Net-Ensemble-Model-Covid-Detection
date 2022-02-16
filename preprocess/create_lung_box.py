# -*- coding: utf-8 -*-

'''
Performs pre-processing for a given Dataset, while splitting it into Train-Val-Test
Folders before it is leveraged for the model trainng
The pre-processing involves:
    (i) Load the disoriented CXRs from various sources in the given folder
    (ii) Crop the CXR around the Thoracic/Lung region(RoI) using the self trained UNet model
    (iii) Peforms appropriate medical image enhancement using image processing techniques
            - CLAHE or Histogram equalization
            - filtering/Blurring/denoising
    (iv) Resize based on appropraite interpolation
    (v) save it to the a desired directory
    
Steps:
    (i) Load the CXRs and Unet Lung segmentation model 
    (ii) Predict the Lung segmentation mask for each CXR using the Unet model
            - Set the Threshold for prediction
    (iii) Contour detection around the lung mask and crop only around the lung contours(based on area)
    (iv) Image enhancement and resizing
    (v) Save to out_path

'''


import cv2
import numpy as np
import os
from tqdm import tqdm 
from skimage.transform import resize
from os import makedirs
import matplotlib.pyplot as plt
from keras.models import model_from_yaml
from sklearn.model_selection import train_test_split
import random


class LungBox:
    
    def __init__(self, image_path, dest_path, model_path, model_name):
        self.image_path = image_path
        self.dest_path = dest_path
        self.model_path = model_path
        self.model_name = model_name
        self.dim = 256
        
           
    def get_images(self, path):
        '''Used to fetch image ids in the provided path'''
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        image_ids = []
        for image_path in image_paths:
            image_id = os.path.split(image_path)[-1].split(".")[0]
            image_ids.append(image_id)
        return image_ids
    
     
    def adap_equalize(self, img):
        '''Used to perform CLAHE for the given image'''
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        try: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            pass
        img = img.astype('uint8')
        img = clahe.apply(img)
        return img
    
    
    def adap_resize(self, in_path, out_dirr, ids): 
        '''Used to resize and apply CLAHE on the given set of images ids''' 
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            name = id_ + '.png'
            img_out = cv2.imread(in_path + name, 0)  
           
            if img_out.shape != (self.dim, self.dim):
                interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>self.dim else  cv2.INTER_CUBIC
                img_out = cv2.resize(img_out, (self.dim, self.dim), interpolation = interpolation_type)
            
            img_out = self.adap_equalize(img_out)
            cv2.imwrite(out_dirr  + name ,img_out)
        
    
    def morph(self, img, option):
        '''    
        # This function performs opening for an given image.
        # Opening is essentially acheived by applying two filters:
        #     Erosion followed by dialiation
        # 
        # i.)  Erosion : Expands the features removing noise
        # ii.) Dialtion : Shrinks the feature and also useful in joining broken parts of an object
        # 
        # In cases like noise removal, erosion is followed by dilation. 
        # Because, erosion removes white noises, but it also shrinks our object. 
        # So we dilate it. Since noise is gone, they won’t come back, but our object area increases.
        #
        # Returns
        # -------
        # opening : Openned image i.e smoothed and shrinked the expanded features of the image.
        '''
        rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        elpscekrnl1 =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
        elpscekrnl2 =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    
        if option == 1:
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, elpscekrnl1)
            return opening
        
        elif option == 2:
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, elpscekrnl2)
            return closing
        
        elif option == 3:
            tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, rectkernel)
            return tophat
    
        else:
            return img
        
    
    def union(self, b1, b2, height, width):
        """
        Parameters
        ----------
        a : 1st Recatangle's box coordinates as dereived form the cv2.boundingRect() over the contours
        b : 2nd Rectangle's box coordinates as dereived form the cv2.boundingRect() over the contours
    
        Returns
        -------
        x : Top-Left corner X coordinate
        y : Top-Left corner Y coordinate
        w : Final width of the Union rectongle
        h : Final height of the Union rectongle
    
        """
        x1 = max(min(min(b1[:,0]),min(b2[:,0])),2)
        y1 = max(min(min(b1[:,1]),min(b2[:,1])),2)
        
        x2 = min(max(max(b1[:,0]),max(b2[:,0])),width-5)
        y2 = min(max(max(b1[:,1]),max(b2[:,1])),height-5)

        return (x1, y1, x2, y2)

    
    def plot(self, img, mask, show_img, crop, _id):
        '''Used to plot raw image with predicted lung mask, followed by bounding box and cropped image'''
        plt.clf() 
        plt.figure(figsize=(12, 4))
        plt.subplot(141)
        plt.title(str(_id))
        plt.imshow(img, cmap='gray')

        plt.subplot(142)
        plt.title('Lung Mask prediction')
        plt.imshow(mask, cmap='gray')
        
        plt.subplot(143)
        plt.title('Bounding box RoI')
        plt.imshow(show_img, cmap='gray')
        
        plt.subplot(144)
        plt.title('Cropped RoI')
        plt.imshow(crop, cmap='gray')
        plt.savefig('/imgs/'+_id+'.png')

        
    def create_lung_box(self, cxr, mask, _id):
        """
        Outputs a cropped region around the lungs for a given CXR
    
        Parameters
        ----------
        cxr : Given CXR image whose lungs mask is determined
        mask: lungs mask as predicted by the lungs segmentation model
        _id : name of the image for the plot
            
        Returns
        -------
        crop : Cropped Image around the lungs in the given CXR made with rotated rectangels
               around the countours to maximize the RoI
        ratio : Ratio of the area of the two lung contours 
    
        """
        
        if mask.shape!=cxr.shape:
            size = cxr.shape
            mask = (mask > 0).astype(np.uint8)
            mask = mask*255
            mask = resize(mask, size , mode='constant',  preserve_range=True)
            mask = (mask > 0).astype(np.uint8)
            mask = mask*255
    
        maskout = cxr*mask
        img = cxr
        height, width = img.shape[:2]
        
        ##Extracting contours out from the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)
        
        try: 
            a1 = cv2.contourArea(cnt[0])
            a2 = cv2.contourArea(cnt[1])
            ratio = a2/a1
            
            show_img = img.copy()
            cv2.drawContours(show_img, cnt, -1, (0, 255, 0),1)
            plt.imshow(show_img, cmap='gray')
            
            rect1 = cv2.minAreaRect(cnt[0])
            box1 = cv2.boxPoints(rect1)
            box1 = np.int0(box1)
            
            width1 = rect1[1][0]
            height1 = rect1[1][1]
            area_b1= width1*height1
            cv2.drawContours(show_img,[box1],0,(255,255,255),2)
    
            rect2 = cv2.minAreaRect(cnt[1])
            box2 = cv2.boxPoints(rect2)
            box2 = np.int0(box2)
            
            width2 = rect2[1][0]
            height2 = rect2[1][1]
            area_b2= width2*height2
            cv2.drawContours(show_img,[box2],0,(255,255,255),2)
    
            x1, y1, x2, y2 = self.union(box1,box2,height, width)
            
            cv2.rectangle(show_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,0),2)
            crop = img[y1:y2, x1:x2]
            self.plot(img, mask, show_img, crop, _id)
                    
        except Exception as e:
            print(e)
            print('No crop!! - ', _id)
            try:
                print("area_b1: ", area_b1, "area_b2: ", area_b2, "ratio: ", ratio)
            except:
                pass
            crop = cxr
            ratio = 0
            
        return crop, ratio
    
    
    def preprocessing(self, in_path, out_dir, ids, seg_model):
        '''This function reads the images, triggers the image preprocessing 
        functions, and creates a lung box using segmentation model'''
        x = np.zeros((len(ids), self.dim, self.dim), dtype=np.uint8) 
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            name = id_ + '.png'
            image = cv2.imread(in_path + name, 0)
            size = image.shape
            img = image
            if img.shape != (self.dim, self.dim): 
                img = cv2.resize(img, (self.dim, self.dim), interpolation = cv2.INTER_AREA)
            img = self.adap_equalize(img)
            img = np.expand_dims(np.array(img), axis = 0)
            preds = seg_model.predict(img, verbose=1)
            preds_t = (preds > 0.5).astype(np.uint8)
            preds_t = np.squeeze(preds_t)
            post_process = False
            if post_process:
                    preds_t = self.morph(preds_t, 1)
                    preds_t = self.morph(preds_t, 2)
                    preds_t = self.morph(preds_t, 1)
            
            mask = resize(preds_t, size , mode='constant',  preserve_range=True)
            mask = (mask > 0).astype(np.uint8)

            img_out, ratio = self.create_lung_box(image, mask, id_)
            if ratio == 0:
                print("Cropping failed for: ", name)

            # Final Resizing of the images
            if img_out.shape != (self.dim, self.dim):
                interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>self.dim else  cv2.INTER_CUBIC
                img_out = cv2.resize(img_out, (self.dim, self.dim), interpolation = interpolation_type)

            cv2.imwrite(out_dir + name ,img_out)
            x[n] = img_out
            return x

    
    def load_seg_model(self, modelpath, modelname):
        '''
        Loads a model
           -> a YAML file to load the model architecture
           -> a .h5/hdf5 file to load the pretrained model weights
        '''
        yaml_file = open(modelpath + modelname + '.yaml', 'r')
        model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(model_yaml)
        model.load_weights(modelpath + modelname + '.h5')
        return model
    
    
    def get_folders(self, path):
        '''Fetches the folders in the given directory'''
        return next(os.walk(path))[1]
    
    
    def create_folder_structure(self, out_path, class_):
        '''Creates a folder structure for storing train, val and test datasets'''
        train_path = out_path + '/train/'
        test_path =  out_path + '/test/'
        val_path =  out_path + '/val/'
        
        class_train_out = train_path + str(class_)  + '/'
        class_val_out = val_path+ str(class_)   + '/'
        class_test_out = test_path+ str(class_)  + '/'
    
        makedirs(class_train_out)
        makedirs(class_val_out)
        makedirs(class_test_out)
        
        return class_train_out, class_val_out, class_test_out


    def create_bounding_box(self):
        '''Used for splitting the dataset into train, val and test folders.
           Also used for triggering the preprocessing pipeline of getting the cropped region of interest'''
        
        model = self.load_seg_model(self.model_path, self.model_name)
        cropped = False
        classes = self.get_folders(self.image_path)
         
        for n, class_ in tqdm(enumerate(classes), total=len(classes)):

            class_train_out, class_val_out, class_test_out = self.create_folder_structure(self.dest_path, class_)
            class_in_path = self.image_path + str(class_) + '/'
        
            ids = self.get_images(class_in_path)
            random.shuffle(ids)
            
            train_ids, test_val_ids  = train_test_split(ids, test_size=0.4, shuffle=True)
            val_ids, test_ids  = train_test_split(test_val_ids, test_size=0.5, shuffle=True)
                  
            if cropped == False:
                self.preprocessing(class_in_path, class_train_out, train_ids, model, self.dim)
                self.preprocessing(class_in_path, class_val_out, val_ids, model, self.dim)
                self.preprocessing(class_in_path, class_test_out, test_ids, model, self.dim)
             
            else:
                self.adap_resize(class_in_path, class_train_out, train_ids, self.dim)
                self.adap_resize(class_in_path, class_val_out, val_ids, self.dim)
                self.adap_resize(class_in_path, class_test_out, test_ids, self.dim)
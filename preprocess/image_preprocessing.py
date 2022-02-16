# -*- coding: utf-8 -*-

import os
from tqdm import tqdm 
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from os import makedirs, path
import cv2
from keras.preprocessing.image import ImageDataGenerator


class ImagePreprocessing:
    
    def __init__(self,image_path, mask_path, merged_mask_path, dest_path, aug_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.merged_mask_path = merged_mask_path
        self.dest_path = dest_path
        self.aug_path = aug_path
        self.cxr_ids = []
        self.lung_masks_ids = []
    
    
    def get_images(self, path):
        '''Used to fetch image ids in the provided path'''
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        image_ids = []
        for image_path in image_paths:
            image_id = os.path.split(image_path)[-1].split(".")[0]
            image_ids.append(image_id)
        return image_ids


    def get_ids(self):
        '''Used to trigger get_images() to fetch ids in the provided path'''
        self.cxr_ids = self.get_images(self.image_path)
        self.lung_masks_ids = self.get_images(self.mask_path)
        
    
    def find_missing_ids(self):
        '''Used to fetch missing image ids and lung masks ids in the provided path'''
        k=0
        print('List image ids that do not have a corresponding lung mask: ')
        for n, id_ in tqdm(enumerate(self.cxr_ids), total=len(self.cxr_ids)):
            cxr_sample = self.cxr_ids[n] 
            if cxr_sample not in self.lung_masks_ids:
                print(k, ' - LUNG MASK ID NOT FOUND: ', cxr_sample)
                k=k+1
        print('total chest x-rays that do not have a corresponding lung mask: ',k)
        
        j=0
        print('List Lung mask ids that do not have a corresponding chest x-ray: ')
        for n, id_ in tqdm(enumerate(self.lung_masks_ids), total=len(self.lung_masks_ids)):
            mask_sample = self.lung_masks_ids[n] 
            if mask_sample not in self.cxr_ids:
                print(j, '- CXR ID NOT FOUND:', mask_sample)
                j=j+1
        print('total Lung masks that do not have a the corresponding CXR: ',j)
        
        
    def merge_masks(self):
        '''Used to merge left and right lung masks in a single image and save it'''
        img_height = img_width = 256
        
        if path.exists(self.merged_mask_path)==False:
            makedirs(self.merged_mask_path)
        os.chdir(self.merged_mask_path)
        
        x = np.zeros((len(self.cxr_ids), img_height, img_width), dtype=np.uint8)
        y = np.zeros((len(self.cxr_ids), img_height, img_width,1), dtype=np.bool)

        for n, id_ in tqdm(enumerate(self.cxr_ids), total=len(self.cxr_ids)):   
            name = id_ + '.png'
            img = imread(self.image_path + '/' + name)
            img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
            x[n] = img  
            
            merged_mask = np.zeros((img_height, img_width), dtype=np.bool)
            mask1 = imread(self.mask_path + '/leftMask/' + name)
            mask1 = np.expand_dims(resize(mask1, (img_height, img_width), mode='constant',  
                                              preserve_range=True), axis=-1)
            
            mask2 = imread(self.mask_path + '/rightMask/' + name)
            mask2 = np.expand_dims(resize(mask2, (img_height, img_width), mode='constant',  
                                              preserve_range=True), axis=-1)
            merged_mask = np.maximum(mask1, mask2)  

            cv2.imwrite(name,merged_mask)
            y[n] = merged_mask 
            
            
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
         
    
    def image_manipulations(self, in_img_path, dim, mask):
        '''Used to perform image manipulations techniques for the given image'''
        image = cv2.imread(in_img_path,0)
        img = image 
        
        if mask==False:
            img = self.adap_equalize(img)
    
        if mask:
            if img.shape != (dim,dim):
                img = resize(img, (dim, dim), mode='constant', preserve_range=True)
        else:
            if img.shape != (dim,dim):
                interpolation_type =  cv2.INTER_AREA if img.shape[1]>dim else  cv2.INTER_CUBIC
                img = cv2.resize(img, (dim, dim), interpolation = interpolation_type)
        img = img.astype('uint8')
        
        return img


    def get_manipulated_image(self, ids, in_dir, out_dir, mask):
        '''Used to trigger image_manipulations() function for the images and save them'''
        makedirs(out_dir)
        img_width = 256
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):   
            name = id_ + '.png'
            image_path = in_dir + '/' + name 
            img = self.image_manipulations(image_path, img_width, mask)
            cv2.imwrite(out_dir + 'raw_' + name ,img)
        
        
    def data_split(self):
        '''Used to split datasets into train, val and test datasets'''
        cxrs = 'prep_cxrs/'
        lung_masks = 'LungsMasks/'

        colnames = ["train", "val","test"]
        data = pd.read_csv('train_val_test_ids.csv', names=colnames)
        train_ids, val_ids, test_ids = data.Train.tolist(), data.Val.tolist(), data.Test.tolist()
        val_ids = [x for x in val_ids if str(x) != 'nan']
        test_ids = [x for x in test_ids if str(x) != 'nan']
        
        self.get_manipulated_image(train_ids, self.image_path + cxrs, self.dest_path + 'train/' + cxrs, mask = False)
        self.get_manipulated_image(train_ids , self.merged_mask_path + lung_masks ,self.dest_path + 'train/' + lung_masks, mask = True)
        
        self.get_manipulated_image(val_ids, self.image_path + cxrs, self.dest_path + 'val/' + cxrs, mask = False)
        self.get_manipulated_image(val_ids , self.merged_mask_path + lung_masks , self.dest_path + 'val/' + lung_masks, mask = True)
        
        self.get_manipulated_image(test_ids, self.image_path + cxrs, self.dest_path + 'test/' + cxrs, mask = False)
        self.get_manipulated_image(test_ids , self.merged_mask_path + lung_masks, self.dest_path + 'test/' + lung_masks, mask = True)
        
    
    def augment_train_data(self):
        '''Used to augment both images and corresponding masks in the train dataset'''
        n_times = 3
        img_width, img_height = 256
        
        cxrs = 'prep_cxrs/'
        lung_masks = 'LungsMasks/'
        des_path = self.dest_path + 'train/'
        
        ids = self.get_images(des_path + cxrs)

        total = len(ids)
        x = np.zeros((len(ids), img_height, img_width), dtype=np.uint8)
        y = np.zeros((len(ids), img_height, img_width), dtype=np.bool)
        
        print('Fetching images and masks')
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            name = id_ + '.png'
            img = imread(des_path + cxrs + name)
            if img.shape != (img_height, img_width):
                img = cv2.resize(img, (img_height, img_width), interpolation = cv2.INTER_AREA)
            x[n] = img  
        
            mask = np.zeros((img_height, img_width), dtype=np.bool)
            mask = imread(des_path + lung_masks + name)
            if mask.shape != (img_height, img_width):
                mask = resize(mask, (img_height, img_width), mode='constant', preserve_range=True)  
            y[n] = mask   

        x = np.array(x)
        x = np.expand_dims(x, axis = -1)
        y = np.array(y)
        y = np.expand_dims(y, axis = -1)
        seed = 32
        
        img_data_gen_args =  dict(rotation_range=3,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode="nearest", brightness_range=[0.6,1.3])
        
        mask_data_gen_args = dict(rotation_range=3,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode="nearest", brightness_range=[0.6,1.3],
                             preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) 
        
        image_data_generator = ImageDataGenerator(**img_data_gen_args)
        image_data_generator.fit(x, augment=True, seed=seed)
        
        mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
        mask_data_generator.fit(y, augment=True, seed=seed)
        
        makedirs(self.aug_path + cxrs)
        makedirs(self.aug_path + lung_masks)
                 
        image_generator = image_data_generator.flow(x, seed=seed, batch_size=1, save_to_dir=self.aug_path + cxrs, save_prefix="cxr", save_format="png")
        mask_generator = mask_data_generator.flow(y, seed=seed, batch_size=1, save_to_dir=self.aug_path + lung_masks, save_prefix="mask", save_format="png")
        
        total = 0
        for img in image_generator:
            total += 1
            if total == n_times:
                break 
                
        total = 0
        for mask in mask_generator:
            total += 1
            if total == n_times:
                break
        
        
    def run_preprocessing(self):
        '''Used to trigger functions'''
        self.get_ids()
        self.find_missing_ids()
        self.merge_masks()
        self.data_split()
        self.augment_train_data()
        
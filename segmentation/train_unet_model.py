# -*- coding: utf-8 -*-

import unet_model as um
import os
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
from skimage.io import imread
from skimage.transform import resize
from os import makedirs, path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from varname import nameof
from datetime import datetime
from contextlib import redirect_stdout


class TrainSegmentation:
    
    def __init__(self, source_path):
        self.source_path = source_path
        self.height, self.width = 256
        return
    
    
    def get_images(self, path):
        '''Used to fetch image ids in the provided path'''
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        image_ids = []
        for image_path in image_paths:
            image_id = os.path.split(image_path)[-1].split(".")[0]
            image_ids.append(image_id)
        return image_ids
        
    
    def load_images_and_masks(self, path, ids):
        '''Used to load images and their coressponding masks'''
        cxrs = 'prep_cxrs/'
        lung_masks = 'LungsMasks/'
    
        x = np.zeros((len(ids), self.height, self.width), dtype=np.uint8)
        y = np.zeros((len(ids), self.height, self.width), dtype=np.bool)
        print('Fetching images and masks') 
        
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            name = id_ + '.png'
            img = cv2.imread(path + cxrs + name,0)
    
            if img.shape != (self.height, self.width):
                img = resize(img, (self.height, self.width), mode='constant', preserve_range=True)
            x[n] = img  
            
            mask = np.zeros((self.height, self.width), dtype=np.bool)
            mask = imread(path + lung_masks + name)
            if mask.shape != (self.height, self.width):
                mask = resize(mask, (self.height, self.width), mode='constant', preserve_range=True)
            y[n] = mask   
        return x, y


    def append_multiple_lines(self, file_name, lines_to_append):
        '''Used to write content to a file'''
        with open(file_name, "a+") as file_object:
            appendEOL = False
            file_object.seek(0)
            data = file_object.read(100)
            if len(data) > 0:
                appendEOL = True
            for line in lines_to_append:
                if appendEOL == True:
                    file_object.write("\n")
                else:
                    appendEOL = True
                file_object.write(line)
    
    
    def confusion_matrix_seg(self, y_true, y_pred_t, setname, mydir):
        '''Used to generate confusion matrix and some other performance evaluation metrics like IoU Score'''
        tp = np.logical_and(y_true==True, y_pred_t==True)
        tn = np.logical_and(y_true==False, y_pred_t==False)
        fp = np.logical_and(y_true==True, y_pred_t==False)
        fn = np.logical_and(y_true==False, y_pred_t==True)
        
        cmat = [[np.sum(tp), np.sum(fn)], [np.sum(fp), np.sum(tn)]]
    
        plt.figure(figsize = (6,6))
        plt.title(setname)
        sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1, linewidth=2.)
        plt.xlabel("predictions")
        plt.ylabel("real values")
        plt.savefig(mydir + setname + 'confusion_mat_seg.png')
        
        try:
            plt.show()
        except:
            pass
        
        iou = np.sum(tp)/(np.sum(tp)+np.sum(fn)+np.sum(fp))
        f1 = (2*np.sum(tp))/((2*np.sum(tp))+np.sum(fn)+np.sum(fp))
        precision =  np.sum(tp)/(np.sum(tp)+np.sum(fp)) 
        
        acc = (np.sum(tp)+np.sum(tn))/(np.sum(tp)+np.sum(tn)+np.sum(fn)+np.sum(fp))
        sensitivity= np.sum(tp)/(np.sum(tp)+np.sum(fn)) 
        specificity = np.sum(tn)/(np.sum(tn)+np.sum(fp)) 
        
        print('IoU score: ',iou,'\nF1 score:', f1,'\nPrecision:', precision,'\nSensitivity:', sensitivity,'\nSpecificity:', specificity,'\nAccuracy:', acc)
        return([iou, f1, precision, sensitivity, specificity, acc])
                  
             
    def plot(self, pred, x_img, y_truth, mydir, id_):
        '''Used to plot the test image along with its ground truth and geenrated prediction comparison'''
        plt.cla()
        plt.clf() 
        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(x_img, cmap='gray')
        
        plt.subplot(232)
        plt.title('Ground truth')
        plt.imshow(x_img*y_truth, cmap='gray')
        
        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(x_img*pred, cmap='gray')
        plt.savefig(mydir + id_ +'_cxr_segmt.png')
        
    
    def get_data(self):
        '''Used to read data from the folders'''
        train_path = self.source_path + '/train/'
        val_path =  self.source_path + '/val/'
        test_path =  self.source_path + '/test/'
        
        train_ids = self.get_images(train_path)
        val_ids = self.get_images(val_path)
        test_ids = self.get_images(test_path)
        
        x_train, y_train = self.load_images_and_masks(train_path, train_ids)
        x_val, y_val = self.load_images_and_masks(val_path, val_ids)
        x_test, y_test = self.load_images_and_masks(test_path, test_ids)

        x_train = np.expand_dims(np.array(x_train), axis = -1)
        y_train = np.expand_dims(np.array(y_train), axis = -1)
        
        x_val = np.expand_dims(np.array(x_val), axis = -1)
        y_val = np.expand_dims(np.array(y_val), axis = -1)
        
        x_test = np.expand_dims(np.array(x_test), axis = -1)
        y_test = np.expand_dims(np.array(y_test), axis = -1)
        
        return x_train, y_train, x_val, y_val, x_test, y_test
        
    def training(self):
        '''Used for training the unet model'''
        img_channels = 1
        
        x_train, y_train, x_val, y_val, x_test, y_test = self.get_data()
        
        model = um.get_model(self.height, self.width, img_channels)
        model.summary()
        
        # View the trainable layers of the model
        for i, layers in enumerate(model.layers):
            print(i,layers.name, "-", layers.trainable)
        
        batch_size = 32
        steps_per_epoch = (len(x_train))//batch_size
        validation_steps = (len(x_val))//batch_size
        epochs=50
        
        result_dir = os.getcwd()+ "/SegResults/"+ str(self.width) +"_b"+ str(batch_size) + "_X"+ str(len(x_train)) + "_" + datetime.now().strftime('%m-%d__%H-%M-%S') +'/'
        makedirs(result_dir) 
        
        modelname = result_dir +  'Adap_3Xaug_unet_lungs_segmtn'
        checkpoint = ModelCheckpoint(modelname + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True,  mode='auto')
        early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.00001,  mode='auto', verbose=1, restore_best_weights = True)
        log_csv = CSVLogger(result_dir + 'my_logs.csv', separator=',', append=False)
        callbacks_list = [checkpoint, early_stop, log_csv]
        
        start = time.time()
        history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=epochs, batch_size = batch_size, callbacks=callbacks_list) 
        end = time.time()
        print('Training finished!')
        print('====started: ', start, '--finished: ',end, ' ====')

        # Saving the model to disk
        model_yaml = model.to_yaml()
        with open( modelname + ".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
            
        # Serialize weights to HDF5
        model.save_weights(modelname + ".h5")
        print("Saved model to disk")
        
        with open(result_dir + 'modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
                     
        line = ['========Time taken for model training: ', str((end-start)/60),  'minutes ========',
                '\nTraining Dataset path: ', str(path),
                '\nTotal train samples: ', str(len(x_train)),
                '\nresult_dir path: ', str(result_dir), 
                '\nINPUT IMAGE_SIZE: ',str(self.height), '\nset EPOCHS: ',str(epochs), 
                '\nvalidation_steps: ',str(validation_steps), '\nTrain batch_size: ',
                str(batch_size), '\nsteps_per_epoch: ',str(steps_per_epoch)]
        
        self.append_multiple_lines(result_dir + 'modelsummary.txt', line)
        
        # Convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 
        # Save to csv: 
        hist_csv_file =result_dir +  'history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        
        # Get predictions
        preds_train = model.predict(x_train, verbose=1)
        preds_val = model.predict(x_val, verbose=1)
        preds_test = model.predict(x_test, verbose=1)
        
        preds_train_t = (preds_train > 0.5).astype(np.uint8)
        preds_val_t = (preds_val > 0.5).astype(np.uint8)
        preds_test_t = (preds_test > 0.5).astype(np.uint8)
        
        # Get results
        print("Test_res: ")
        test_res = self.confusion_matrix_seg(y_test, preds_test_t, nameof(preds_test_t), result_dir)
        
        print("Val_res: ")
        val_res = self.confusion_matrix_seg(y_val, preds_val_t, nameof(preds_val_t), result_dir)
        
        print("Train_res: ")
        train_res = self.confusion_matrix_seg(y_train, preds_train_t, nameof(preds_train_t), result_dir)
        
        # Storing metric results to an excel sheet
        df1 = pd.DataFrame([train_res + [len(y_train)] +[batch_size, epochs, steps_per_epoch], 
                            val_res + [len(y_val)], test_res + [len(y_test)]],
                           index=["train","val","test"], 
                           columns=['IoU score', 'F1 score','Precision:','Sensitivity','Specificity',
                                    'Accuracy', 'Datapoints','Batchsize', 'Epochs', 'StepsPerEpoch'])
        
        df1.to_excel(result_dir + "/output"+ datetime.now().strftime('%m-%d__%H-%M-%S') +".xlsx")  
        
        # Perform a sanity check on some random test samples
        ix = random.randint(0, len(preds_test_t))
        self.plot(preds_test_t[ix], x_test[ix], y_test[ix], result_dir, self.get_images(self.source_path + '/test/')[ix])

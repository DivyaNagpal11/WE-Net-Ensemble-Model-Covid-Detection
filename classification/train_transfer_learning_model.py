# -*- coding: utf-8 -*-

'''
# =============================================================================
Steps:-
# 1. read image ids from the dataset path(train, val, and test)
# 2. Resize to 256 for getting mask predictions
# 3. Load segmentation model and get masks
# 4. Resize the mask to the original CXR img dimension
# 5. Cropping RoI in the input CXR img about the lungs as per cnt detection in masks
# 6. Resize the cropped CXR img to the input image dimension : 256
# 7. Data Aug
# 8. Denoising - optional preprocessing integrated in Data Aug
# 9. Feed in to the model and train - Early stopping
# 10. Confusion matrix and evaluation on the various metrices
# =============================================================================    
'''
from transfer_learning_models import TransferLearning as tl
import pandas as pd
from glob import glob
from contextlib import redirect_stdout
import matplotlib as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import layers
import time
import collections


class TrainClassification:
    
    def __init__(self, source_path, dest_path, model_name): 
        self.source_path = source_path
        self.model_name = model_name
        self.dest_path = dest_path
        self.batch_size = 64
        self.epochs= 20  
        self.dim = 256
        self.img_width = self.dim
        self.img_height = self.dim
        self.img_channels = 1
    
    def append_multiple_lines(self, file_name, lines_to_append):
        '''Opens an existing .txt file and writes to it.'''
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
                
    
    def plot_model_hist(self, history, dest_path):
        '''
        Parameters
        ----------
        history : history dataframe read from the csv generated during the model training
        dest_path : path to save the generated model loss and accuray plots
    
        Returns
        -------
        None.
    
        '''
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss/acc')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(dest_path + 'Train_val_loss.png')
        try:
            plt.show()
        except:
            pass
    
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(dest_path + 'Train_val_acc.png')
        try:
            plt.show()
        except:
            pass
    
    def train_function(self):
        '''Used to read datasets and train the model name specified by the user in the command line arguments'''
        img_size = [self.img_width, self.img_height]
        train_path = self.source_path + '/train/'
        val_path = self.source_path + '/val/'
        test_path = self.source_path + '/test/'

        class_folders = glob(train_path + '/*')


        if self.model_name == "vgg16":
            model = tl.vgg16_model(img_size, class_folders)
            trained_model = "VGG16_dl" 
        
        elif self.model_name == "resnet50":
            model = tl.resnet_model(img_size, class_folders)
            trained_model = "Resnet50_dl"
        
        elif self.model_name == "inceptV3":
            model = tl.InceptionV3_model(img_size, class_folders)
            trained_model = "InceptionV3_dl"
            
        elif self.model_name == "densenet201":
            model = tl.densenet201_model(img_size, class_folders)
            trained_model = "Densenet_dl" 
            
        elif self.model_name == "xception":
            model = tl.Xception_model(img_size, class_folders)
            trained_model = "Xception_dl"
        
        model.summary()

        a =[]
        for i, layers_ in enumerate(model.layers):
            a.append([i,layers.name, "-", layers_.trainable])
            print(i,layers_.name, "-", layers_.trainable)
       
        datagen = ImageDataGenerator(rescale=1./255, preprocessing_function= None)
        
        train_set = datagen.flow_from_directory(train_path, shuffle=True, target_size=img_size, batch_size=self.batch_size, class_mode='categorical', interpolation = "bicubic")
        val_set = datagen.flow_from_directory(val_path, shuffle=False,target_size=img_size, batch_size=1, class_mode='categorical',interpolation = "bicubic")
        test_set = datagen.flow_from_directory(test_path, shuffle=False, target_size=img_size, batch_size=1, class_mode='categorical', interpolation = "bicubic")
        
        steps_per_epoch = train_set.samples//train_set.batch_size 
        validation_steps = val_set.samples//val_set.batch_size 
        
        train_len = str(steps_per_epoch*self.batch_size) 
        modelpath = self.dest_path + "/" + trained_model + "/"

        print("[INFO] compiling model...", trained_model)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        ##Modelcheckpoint
        checkpoint = ModelCheckpoint(modelpath + ".hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,  mode='max')
        early_stop = EarlyStopping(monitor='val_accuracy', patience=3,  mode='max', verbose=1, restore_best_weights = True)
        log_csv = CSVLogger(self.dest_path + 'my_logs.csv', separator=',', append=False)
        callbacks_list = [checkpoint, early_stop, log_csv]
        
        start = time.time()
        history = model.fit(train_set, validation_data = val_set, epochs=self.epochs, steps_per_epoch = steps_per_epoch, 
                            validation_steps = validation_steps,callbacks = callbacks_list)
        end = time.time()
        print('Training finished! ->',trained_model)
        print('========Time taken: ', (end-start)/60,  'minutes ========')
        
        ################################################
        # Saving the model to disk
        # serialize model to YAML, saves the model architecture to YAML
        model_yaml = model.to_yaml()
        with open( modelpath + ".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights(modelpath + ".h5")
        model.save(self.dest_path + 'saved_model')
        print("Saved model to disk")
        
        ################################################
        # SAVING LOGS of training
        with open(self.dest_path + 'modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
                
        with open(self.dest_path + 'modelsummary.txt', 'a') as output:
            for row in a:
                output.write(str(row) + '\n')
      
        line = ['========Time taken for model training: ', str((end-start)/60),  'minutes ========',
                '\nTraining Dataset path: ', str(self.source_path), '\nTrainng Classes: ',str(train_set.class_indices), 
                '\nTotal train samples: ', (train_len), ' ::Out of: ', str(train_set.samples),
                '\nClasswise train sample support: ', str(collections.Counter(train_set.labels)),
                '\nresult_dir path: ', str(self.dest_path), 
                '\nINPUT IMAGE_SIZE: ',str(img_size), '\nset EPOCHS: ',str(self.epochs), 
                '\nvalidation_steps: ',str(validation_steps), '\nTrain batch_size: ',
                str(self.batch_size), '\nsteps_per_epoch: ',str(steps_per_epoch)]
        
        self.append_multiple_lines(self.dest_path + 'modelsummary.txt', line)
        
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 
        # save to csv: 
        hist_csv_file = self.dest_path +  'history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        
        # Plot the model history: accuracy and loss
        try:
            self.plot_model_hist(history, self.dest_path)
        except Exception as e:
            print("Exception due to ERROR: ", e)

        # confusion matrix and model evaluation
        eval(model, test_set, "test_set ", self.dest_path)
        model.evaluate(test_set, steps = test_set.samples, verbose =1)
        
        eval(model, val_set, "val_set ", self.dest_path)
        model.evaluate(val_set, steps = val_set.samples)
        
    
    

# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import RandomizedSearchCV
from glob import glob
from catboost import CatBoostClassifier
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import statistics
from ci_roc import get_ci_auc
from sklearn.metrics import f1_score
import xgboost as xgb
import pickle
from classification.transfer_learning_models import TransferLearning as tl
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from sklearn import svm
from sklearn.metrics import roc_curve 
from sklearn.metrics import matthews_corrcoef


class EnsembleModels:
    
    def __init__(self, test_images_path, saved_model_path):
        self.test_images_path = test_images_path
        self.saved_model_path = saved_model_path
        self.dim = 256
        self.batch_size = 64
        self.img_width = self.dim
        self.img_height = self.dim
        self.img_channels = 1
        self.img_size = [self.img_width, self.img_height]   
        self.m1_classes = []
        self.m2_classes = []
        self.m3_classes = []
        self.m4_classes = []
        self.m5_classes = []
        
        
    def read_images(self):
        image_datagen = ImageDataGenerator(rescale=1./255)
        test_set = image_datagen.flow_from_directory(self.test_images_path, shuffle=False, target_size=self.img_size, batch_size=1, class_mode='categorical', interpolation = "bicubic")
        return test_set
    
    
    def load_trained_models(self):
        class_folders = glob(self.test_images_path + '/*')
        
        model1 = tl.vgg16_model(self.img_size, class_folders)
        model1.load_weights(self.saved_model_path + '/VGG16_dl256mode1_22class_model256RAW_aug.hdf5')
        model1.summary()
        
        model2 = tl.resnet_model(self.img_size, class_folders)
        model2.load_weights(self.saved_model_path + '/Resnet50_dl256bn1_2class_model256RAW_aug.hdf5')
        model2.summary()
        
        model3 = tl.inceptionv3_model(self.img_size, class_folders)
        model3.load_weights(self.saved_model_path + '/InceptionV3_dl256bn1_2class_model256RAW_aug.hdf5')
        model3.summary()
        
        model4 = tl.densenet201_model(self.img_size, class_folders)
        model4.load_weights(self.saved_model_path + '/Densenet_dl256_2class_model256RAW_aug.hdf5')
        model4.summary()
        
        model5 = tl.Xception_model(self.img_size, class_folders)
        model5.load_weights(self.saved_model_path + '/Xception_dl256bn1_2class_model256RAW_aug.hdf5')
        model5.summary()

        return model1, model2, model3, model4, model5
    
    
    def get_preds(self):
        model1, model2, model3, model4, model5 = self.load_trained_models()
        ####### Getting level1 predictions i.e. prob scores for each class from each model
        models = [model1, model2, model3, model4, model5]
        test_set = self.read_images()
        preds = [model.predict(test_set) for model in tqdm(models)]
        return preds
    
    
    def dump_preds(self):
        preds = self.get_preds()
        ### Storing the preds in pickle
        file = open('preds.pkl','wb')
        pickle.dump(preds, file)
        file.close()

  
    def load_preds(self):     
        ### Loading the preds from pickle
        file = open('preds.pkl', 'rb')
        preds = pickle.load(file)
        file.close()
        return preds
                  
    def get_preds_argmax(self):
        preds = self.load_preds()
        self. m1_classes=list(np.argmax(preds[0], axis=-1))
        self.m2_classes=list(np.argmax(preds[1], axis=-1))
        self.m3_classes=list(np.argmax(preds[2], axis=-1))
        self.m4_classes=list(np.argmax(preds[3], axis=-1))
        self.m5_classes=list(np.argmax(preds[4], axis=-1))


    def hard_vote_ensemble(self): 
        #### Hard Vote Ensemble 
        individual_predictions=pd.DataFrame([self.m1_classes, self.m2_classes, self.m3_classes,
                                             self.m4_classes, self.m5_classes]).T
        individual_predictions["max_freq"]=[statistics.mode(individual_predictions.loc[i]) for i in individual_predictions.index]
        
        return individual_predictions
    
    
    def average_ensemble(self):
        ### Simple Average Ensemble
        preds_array=np.array(self.load_preds())
        summed = np.sum(preds_array, axis=0)
        ensemble_prediction_with_argmax = np.argmax(summed, axis=1)
        summed_df=pd.DataFrame(summed, columns=["0","1"])
        return ensemble_prediction_with_argmax
    
    
    def weighted_average(self):
        ####################################################################################
        #Grid search for the best combination of weights that gives maximum acuracy in weighted average ensemble 
        df = pd.DataFrame([])
        a = [range(1, 9),range(1, 9),range(1, 9),range(1, 9),range(1, 9)]
        combinations=list(itertools.product(*a))
        
        for i in combinations:
            wts = [i[0]/10,i[1]/10,i[2]/10, i[3]/10, i[4]/10]
            if sum(wts)==1:
                wted_preds1 = np.tensordot(self.load_preds(), wts, axes=((0),(0)))
                wted_ensemble_pred = np.argmax(wted_preds1, axis=1)
                weighted_accuracy = accuracy_score(self.read_images().classes, wted_ensemble_pred)
                df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                             'wt3':wts[2],'wt4':wts[3],'wt5':wts[4], 'acc':weighted_accuracy*100}, index=[0]), ignore_index=True)
                    
        max_acc_row = df.iloc[df['acc'].idxmax()]
        print("Grid Search")
        print("Max accuracy of ", max_acc_row[5], " obained with w1=", max_acc_row[0], " w2=",max_acc_row[1],
              " and w3=", max_acc_row[2], " and w4=", max_acc_row[3], " and w5=", max_acc_row[4])
        
        #Weighted Average ensemble using the best weights obtained
        weights = [0.1, 0.2, 0.1, 0.4, 0.2]
        
        #Use tensordot to sum the products of all elements over specified axes.
        weighted_preds = np.tensordot(self.load_preds(), weights, axes=((0),(0)))
        weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)
        return weighted_ensemble_prediction
    
    
    def stacked_generalization_xgboost_random_search_cv(self):
        #Stacking the predictions together
        preds = self.load_preds()
        stackX = None
        for i in range(0,len(preds)):
            if stackX is None:
                stackX = preds[i]
            else:
                stackX = np.dstack((stackX, preds[i]))
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
        
        ###################################################################
        #Hyperparameter tuning for XGBoost using RandomizedSearchCV
        X,y=pd.DataFrame(stackX),pd.DataFrame(self.read_images().classes)
        
        params = {
                'min_child_weight': [1,2,3,4,5,10,15],
                'gamma': [0.5, 1, 1.5, 0.25, 0.3,0.4],
                'subsample': [0.9, 0.8, 1.0, 0.7, 0.5, 0.6,0.4],
                'colsample_bytree': [0.9, 0.8, 1.0, 0.7, 0.5, 0.6,0.4,0.3],
                'max_depth': [2,3,4,5,6,7,8,9,10],
                'learning_rate':[0.1,0.01,0.02,0.03,0.035,0.04,0.045,0.05],
                'lambda':[0.1,0.2,0.3,0.4,0.5,0.01],
                'n_estimators':[600,500,400,300,100,200,350,450]
                }
        
        folds = 3
        param_comb = 50
        
        skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
        
        xgb1 = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='multi:softmax',num_class=2,
                            silent=True, nthread=1, eval_metric='error')
        
        random_search = RandomizedSearchCV(xgb1, param_distributions=params, 
                                            n_iter=param_comb, scoring='roc_auc', 
                                            n_jobs=4, cv=skf.split(X,y), 
                                            verbose=3, random_state=1001 )
        random_search.fit(X, y)
        
        print('\n Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)
        
        
    def cross_val(self, X, y, model):
        results_1=[]
        skf = StratifiedKFold(n_splits=3)
        for train_index,test_index in skf.split(X,y):
            x_train = X.iloc[train_index,:]
            y_train = y.iloc[train_index,:]
            x_test =  X.iloc[test_index,:]
            y_test =  y.iloc[test_index,:]
            model.fit(x_train, y_train)
            y_preds = model.predict(x_test)
            acc = accuracy_score(y_test, y_preds)
            f1 = f1_score(y_test, y_preds)
            mcc=matthews_corrcoef(y_test, y_preds)
            auc,ci = get_ci_auc(y_preds, y_test[0].values)
            fpr, tpr, thresholds = roc_curve(y_test, y_preds)
            results_1.append([acc,f1,mcc,auc,ci,fpr, tpr, thresholds])
        
        results_df_1=pd.DataFrame(results_1,index=["fold1","fold2","fold3"],
                              columns=["accuracy","f1_score","MCC","AUC","95% AUC CI","fpr", "tpr", "thresholds"]).T
        return results_df_1
    
    
    def stacked_generalization_stratified_k_fold(self):
        ####### Ensemble stacking using stratified k-fold
        ##Using the best params received from the Randomized Search CV
        preds = self.load_preds()
        stackX = None
        for i in range(0,len(preds)):
            if stackX is None:
                stackX = preds[i]
            else:
                stackX = np.dstack((stackX, preds[i]))
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
        
        X,y=pd.DataFrame(stackX),pd.DataFrame(self.read_images().classes)
        params={'subsample': 0.7, 'n_estimators': 400, 'min_child_weight': 4, 'max_depth': 7, 'learning_rate': 0.03, 'lambda': 0.3, 'gamma': 0.5, 'colsample_bytree': 0.6}
        xgb_model = xgb.XGBClassifier(**params)
        
        cat_model = CatBoostClassifier()
        ada_model = AdaBoostClassifier()
        gbm_model = lgb.LGBMClassifier()
        svm_model = svm.SVC()
        
        cat_cross_val_res=self.cross_val(X,y,cat_model)
        cat_cross_val_res['mean'] = cat_cross_val_res[0:4].mean(numeric_only=False, axis=1)
        
        xgb_cross_val_res=self.cross_val(X,y,xgb_model)
        xgb_cross_val_res['mean'] = xgb_cross_val_res[0:4].mean(numeric_only=False, axis=1)
        
        ada_cross_val_res=self.cross_val(X,y,ada_model)
        ada_cross_val_res['mean'] = ada_cross_val_res[0:4].mean(numeric_only=False, axis=1)
        
        gbm_cross_val_res=self.cross_val(X,y,gbm_model)
        gbm_cross_val_res['mean'] = gbm_cross_val_res[0:4].mean(numeric_only=False, axis=1)
        
        svm_cross_val_res=self.cross_val(X,y,svm_model)
        svm_cross_val_res['mean'] = svm_cross_val_res[0:4].mean(numeric_only=False, axis=1)

        return cat_cross_val_res, xgb_cross_val_res, ada_cross_val_res, gbm_cross_val_res, svm_cross_val_res
        
        
    def calculate_metrics(self):
        ### Calculating the metrics for each base model & ensemble models as well
        test_set = self.read_images()
        m1_classes = self.m1_classes 
        m2_classes = self.m2_classes 
        m3_classes = self.m3_classes 
        m4_classes = self.m4_classes 
        m5_classes = self.m5_classes
        individual_predictions = self.hard_vote_ensemble()
        ensemble_prediction_with_argmax = self.average_ensemble()
        weighted_ensemble_prediction = self.weighted_average()
        
        vgg_res=[accuracy_score(test_set.classes, m1_classes),
                 f1_score(test_set.classes, m1_classes),
                 matthews_corrcoef(test_set.classes, m1_classes),
                 get_ci_auc(np.array(m1_classes),test_set.classes)[0],
                 get_ci_auc(np.array(m1_classes),test_set.classes)[1],
                 roc_curve(test_set.classes, m1_classes)[0],
                 roc_curve(test_set.classes, m1_classes)[1],
                 roc_curve(test_set.classes, m1_classes)[2]]
        
        resnet_res=[accuracy_score(test_set.classes, m2_classes),
                    f1_score(test_set.classes, m2_classes),
                    matthews_corrcoef(test_set.classes, m2_classes),
                    get_ci_auc(np.array(m2_classes),test_set.classes)[0],
                    get_ci_auc(np.array(m2_classes),test_set.classes)[1],
                    roc_curve(test_set.classes, m2_classes)[0],
                    roc_curve(test_set.classes, m2_classes)[1],
                    roc_curve(test_set.classes, m2_classes)[2]]
        
        inception_res=[accuracy_score(test_set.classes, m3_classes),
                       f1_score(test_set.classes, m3_classes),
                       matthews_corrcoef(test_set.classes, m3_classes),
                       get_ci_auc(np.array(m3_classes),test_set.classes)[0],
                       get_ci_auc(np.array(m3_classes),test_set.classes)[1],
                       roc_curve(test_set.classes, m3_classes)[0],
                     roc_curve(test_set.classes, m3_classes)[1],
                     roc_curve(test_set.classes, m3_classes)[2]]
        
        densenet_res=[accuracy_score(test_set.classes, m4_classes),
                      f1_score(test_set.classes, m4_classes),
                      matthews_corrcoef(test_set.classes, m4_classes),
                      get_ci_auc(np.array(m4_classes),test_set.classes)[0],
                      get_ci_auc(np.array(m4_classes),test_set.classes)[1],
                      roc_curve(test_set.classes, m4_classes)[0],
                     roc_curve(test_set.classes, m4_classes)[1],
                     roc_curve(test_set.classes, m4_classes)[2]]
        
        xception_res=[accuracy_score(test_set.classes, m5_classes),
                      f1_score(test_set.classes, m5_classes),
                      matthews_corrcoef(test_set.classes, m5_classes),
                      get_ci_auc(np.array(m5_classes),test_set.classes)[0],
                      get_ci_auc(np.array(m5_classes),test_set.classes)[1],
                      roc_curve(test_set.classes, m5_classes)[0],
                    roc_curve(test_set.classes, m5_classes)[1],
                    roc_curve(test_set.classes, m5_classes)[2]]
        
        hard_voting_res=[accuracy_score(test_set.classes, individual_predictions["max_freq"].values),
                         f1_score(test_set.classes, individual_predictions["max_freq"].values),
                         matthews_corrcoef(test_set.classes, individual_predictions["max_freq"].values),
                         get_ci_auc(individual_predictions["max_freq"].values,test_set.classes)[0],
                         get_ci_auc(individual_predictions["max_freq"].values,test_set.classes)[1],
                         roc_curve(test_set.classes, individual_predictions["max_freq"].values)[0],
                         roc_curve(test_set.classes, individual_predictions["max_freq"].values)[1],
                         roc_curve(test_set.classes, individual_predictions["max_freq"].values)[2]]
        
        average_ensemble_res=[accuracy_score(test_set.classes, ensemble_prediction_with_argmax),
                              f1_score(test_set.classes, ensemble_prediction_with_argmax),
                              matthews_corrcoef(test_set.classes, ensemble_prediction_with_argmax),
                              get_ci_auc(ensemble_prediction_with_argmax,test_set.classes)[0],
                              get_ci_auc(ensemble_prediction_with_argmax, test_set.classes)[1],
                              roc_curve(test_set.classes, ensemble_prediction_with_argmax)[0],
                             roc_curve(test_set.classes, ensemble_prediction_with_argmax)[1],
                             roc_curve(test_set.classes, ensemble_prediction_with_argmax)[2]]
        
        weighted_average_ensemble=[accuracy_score(test_set.classes, weighted_ensemble_prediction),
                                   f1_score(test_set.classes, weighted_ensemble_prediction),
                                   matthews_corrcoef(test_set.classes, weighted_ensemble_prediction),
                                   get_ci_auc(np.array(weighted_ensemble_prediction),test_set.classes)[0],
                                   get_ci_auc(np.array(weighted_ensemble_prediction),test_set.classes)[1],
                                   roc_curve(test_set.classes, weighted_ensemble_prediction)[0],
                                 roc_curve(test_set.classes, weighted_ensemble_prediction)[1],
                                 roc_curve(test_set.classes, weighted_ensemble_prediction)[2]]
        
        
        results_df=pd.DataFrame([vgg_res,resnet_res,inception_res,densenet_res,
                                 xception_res,hard_voting_res,average_ensemble_res,
                                 weighted_average_ensemble], 
                                index=["VGG16","ResNet","InceptionV3","DenseNet",
                                       "Xception","Hard Voting","Average Ensemble",
                                       "Weighted Average"],
                                columns=["accuracy","f1_score","MCC","AUC","95% AUC CI","fpr", "tpr", "thresholds"])
        
        return results_df

    def trigger_functions(self):
        self.dump_preds()
        self.get_preds_argmax()
        results_df = self.calculate_metrics()
        self.stacked_generalization_xgboost_random_search_cv()
        cat_cross_val_res, xgb_cross_val_res, ada_cross_val_res, gbm_cross_val_res, svm_cross_val_res = self.stacked_generalization_stratified_k_fold()
        














# -*- coding: utf-8 -*-
"""
@author: Muhammad Zohaib Sarwar (PhD Candidate)
@Email: muhammad.z.sarwar@ntnu.no
ML Class Example
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.models import load_model
import time
from tensorflow.keras.losses import kullback_leibler_divergence
import time
from random import shuffle
import joblib  # save scaler
from tensorflow.keras.layers import LSTM, Activation,GaussianNoise,Concatenate,RepeatVector,Input,Lambda,Conv1D,MaxPooling1D,Flatten,UpSampling1D,Reshape,BatchNormalization
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.utils import plot_model
from scipy import signal
from tensorflow.keras import backend as K



#from keras.utils.vis_utils import plot_model
#import keras
# Setup GPU for training (use tensorflow v1.9 for LSTM)
#random.seed(2020)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # CPU:-1; GPU0: 1; GPU1: 0;

N=2200
def Generate_data(X_data0, y_data0, window_size=50):
    X_new_temp = []
    y_new_temp = []
    for ii in range(len(X_data0)):
        X_temp = X_data0[ii]
        y_temp = y_data0[ii]
        X_new = []
        y_new = []
        for jj in range(int(np.floor(len(X_temp) / window_size))):
            X_new.append(X_temp[jj * window_size:(jj + 1) * window_size])
            y_new.append(y_temp[(jj + 1) * window_size - 1, :])
            # y_new.append(y_temp[(jj + 1) * window_size - 1])

        X_new_temp.append(np.array(X_new))
        y_new_temp.append(np.array(y_new))

    X_data_new0 = np.array(X_new_temp)
    y_data_new0 = np.array(y_new_temp)

    return X_data_new0, y_data_new0


# Load data
#dataDir = 'D:/Zohaib_Phd Folder/Matlab Simulation/DeepLSTM-master/results/P01/Five_Axle/'  # Replace the directory

valu = scipy.io.loadmat('D:\Zohaib_Phd Folder\ML Class Data\Data_Damage00.mat')


#scaler_X = joblib.load(dataDir+'Trained_Fleet/scaler_X_021120_fleeet2_autoencoder_02112020_Final_Paper_FA_no_Profile_Front.save')
#scaler_y = joblib.load(dataDir+'Trained_Fleet/scaler_y_021120_fleeet2_autoencoder_02112020_Final_Paper_FA_no_Profile_Front.save')
"""""
X_data = mat['input_tff_front']
y_data = mat['target_tff_front']
train_indices = mat['trainIndd'] - 1
test_indices = mat['valindd'] - 1

X_data = np.reshape(X_data, [X_data.shape[0], X_data.shape[1], 1])

# Scale data
X_data_flatten = np.reshape(X_data, [X_data.shape[0]*X_data.shape[1], X_data.shape[2]])
#scaler_X = MinMaxScaler(feature_range=(-1, 1))

scaler_X = StandardScaler()
scaler_X.fit(X_data_flatten)
X_data_flatten_map = scaler_X.transform(X_data_flatten)
X_data_map = np.reshape(X_data_flatten_map, [X_data.shape[0], X_data.shape[1], X_data.shape[2]])

y_data_flatten = np.reshape(y_data, [y_data.shape[0]*y_data.shape[1], 1])
#scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_y = StandardScaler()
scaler_y.fit(y_data_flatten)
y_data_flatten_map = scaler_y.transform(y_data_flatten)
y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], 1])
# Unknown data
mat2 = mat73.loadmat(dataDir+'Raw Accleration/Input_Data_Set_Fleet/Fleet_Data_paper_FA_no_profile_00_Big_Data.mat')


X_pred = mat2['input_tff_front']
y_pred_ref = mat2['target_tff_front']
X_pred = np.reshape(X_pred, [X_pred.shape[0], X_pred.shape[1], 1])

# Scale data
X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0]*X_pred.shape[1], X_pred.shape[2]])
X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]])

y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0]*y_pred_ref.shape[1], 1])
y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], 1])

windowsize = 1
# X_data_new, y_data_new = Generate_data(X_data_map, y_data_map, windowsize)

# X_data_new = np.reshape(X_data_new, [X_data_new.shape[0], X_data_new.shape[1], X_data_new.shape[2]])
# y_data = np.reshape(y_data, [y_data.shape[0], y_data.shape[1], 1])

X_train = X_data_map[0:N]
y_train = y_data_map[0:N]
X_test =X_data_map[N:]
y_test = y_data_map[N:]

# X_pred, y_pred_ref = Generate_data(X_pred_map, y_pred_ref_map, windowsize)
X_pred = np.reshape(X_pred_map, [X_pred_map.shape[0], X_pred_map.shape[1], X_pred_map.shape[2]])
#y_pred_ref = np.reshape(y_pred_ref, [y_pred_ref.shape[0], y_pred_ref.shape[1], 1])

# n_epoch = 200
data_dim = X_train.shape[2]  # number of input features
timesteps = X_train.shape[1]
num_classes = y_train.shape[2]  # number of output features
batch_size =64

rms = RMSprop(lr=0.001, decay=0.0001)
adam = Adam(lr=0.001, decay=0.0001)

# input_sig = Input(shape=(timesteps,data_dim)) 

# x = Conv1D(512,4, activation='relu', padding='same')(input_sig) 
# x1 = MaxPooling1D(2)(x) 
# x2 = Conv1D(264,4, activation='relu', padding='same')(x1) 
# x3 = MaxPooling1D(2)(x2) 
# x4 = Conv1D(128,4, activation='relu', padding='same')(x3) 
# x5 = MaxPooling1D(2)(x4) 
# x6 = Conv1D(64,4, activation='relu', padding='same')(x5) 
# x7 = MaxPooling1D(2)(x6) 

# flat = Flatten()(x7) 
# encoded = Dense(32,activation = 'relu',kernel_regularizer=tf.keras.regularizers.l1(1e-5))(flat) 
# encoded = Dense(16,activation = 'relu')(encoded) 

# print("shape of encoded {}".format(K.int_shape(encoded))) 
 
# dec_1=Reshape((1, 16))(encoded)
# # DECODER  
# dec_1 = Conv1D(64, 4, activation='relu', padding='same')(dec_1) 
# dec_1 = UpSampling1D(2)(dec_1) 
# dec_1 = Conv1D(128,4, activation='relu', padding='same')(dec_1) 
# dec_1 = UpSampling1D(2)(dec_1) 
# dec_1 = Conv1D(264, 4, activation='relu', padding='same')(dec_1) 
# dec_1 = UpSampling1D(2)(dec_1) 
# dec_1 = Conv1D(512,4, activation='relu', padding='same')(dec_1) 
# upsamp1 = UpSampling1D(2)(dec_1) 
# flat = Flatten()(upsamp1) 
# decoded = Dense(timesteps,activation = 'linear' )(flat) 
# decoded = Reshape((timesteps, num_classes))(decoded) 
 
# print("shape of decoded {}".format(K.int_shape(decoded))) 
 
# print("shape of decoded {}".format(K.int_shape(decoded))) 
 
input_sig = Input(shape=(timesteps,data_dim)) 


a_x = Conv1D(512,3,activation='relu', padding='same')(input_sig)    
x3 = MaxPooling1D(3)(a_x) 



b_x = Conv1D(256,3,activation='relu', padding='same')(x3)   
x3 = MaxPooling1D(2)(b_x) 

c_x = Conv1D(64,3, activation='relu', padding='same')(x3)    
x3  = MaxPooling1D(2)(c_x) 


d_x = Conv1D(32,3, activation='relu', padding='same')(x3)   
x3  = MaxPooling1D(1)(d_x) 



flat = Flatten()(x3) 
encoded = Dense(4000,activation = 'relu')(flat) 

encoded = Dense(16,activation = 'relu',kernel_regularizer=tf.keras.regularizers.l1(1e-4))(encoded) 

##
print("shape of encoded {}".format(K.int_shape(encoded))) 

dec_1 = Dense(4000,activation = 'relu')(encoded) 
dec_1= Reshape((125,32))(dec_1)


dec_1 = UpSampling1D(2)(dec_1) 
dec_1 = Conv1D(64, 3,activation='relu', padding='same')(dec_1) 


dec_1 = UpSampling1D(2)(dec_1) 
dec_1 = Conv1D(256, 3,activation='relu', padding='same')(dec_1) 

dec_1 = UpSampling1D(3)(dec_1) 
dec_1 = Conv1D(512, 3,activation='relu', padding='same')(dec_1) 



# dec_1 = UpSampling1D(5)(dec_1) 
# dec_1 = Conv1D(1024, 3,activation='relu', padding='same')(dec_1) 
# dec_1 = BatchNormalization()(dec_1)

decoded = Conv1D(data_dim,1, padding='same', activation = 'linear')(dec_1) 


model = Model(input_sig, decoded) 
encoder=Model(input_sig,encoded)

model.summary()

model.compile(loss='mean_squared_error',  # categorical_crossentropy, mean_squared_error, mean_absolute_error
              optimizer=adam,  # RMSprop(), Adagrad, Nadam, Adagrad, Adadelta, Adam, Adamax,
              metrics=['mse'])
best_loss = 100
train_loss = []
test_loss = []
history = []


with tf.device('/GPU:0'):
#def no_gpu():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    start = time.time()

    epochs = 3500
    for e in range(epochs):
        print('epoch = ', e + 1)

        Ind = list(range(len(X_data_map)))
        shuffle(Ind)
        ratio_split = 0.70
        Ind_train = Ind[0:round(ratio_split*len(X_data_map))]
        Ind_test = Ind[round(ratio_split*len(X_data_map)):]

        X_train = X_data_map[Ind_train]
        y_train = y_data_map[Ind_train]
        X_test = X_data_map[Ind_test]
        y_test = y_data_map[Ind_test]

        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  #validation_split=0.3,
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  epochs=2
                  )
       
        score0 = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        train_loss.append(score0[0])      
        test_loss.append(score[0])

     
        if test_loss[e] < best_loss:
            best_loss = test_loss[e]
            #model.save(dataDir + 'Trained_Fleet/my_best_lstm_autoencoder_04112020_Final_Paper_FA_Full_Profile_combine_front_original.h5')
            #encoder.save(dataDir + 'Trained_Fleet/my_best_lstm_encoder_04112020_Final_Paper_FA_Full_Profile_combine_front_original.h5')
            #decoder.save(dataDir + 'Trained Model/my_best_lstm_decoder_31102020_Final_Paper_FA_with_Profile_combine.h5')
       
  
    end = time.time()
    running_time = (end - start)/3600
    print('Running Time: ', running_time, ' hour')
#Plot training and testing loss
plt.figure()
plt.plot(np.array(train_loss), 'b-')
plt.plot(np.array(test_loss), 'm-')

plt.figure()
plt.plot(np.log(np.array(train_loss)), 'b-')
plt.plot(np.log(np.array(test_loss)), 'm-')









#********************************************************Testing**************************************************************#
##Load the best mode
#model_best = load_model(dataDir + 'Trained_Fleet/my_best_lstm_autoencoder_04112020_Final_Paper_FA_no_Profile_combine_back.h5')




y_train_pred = model_best.predict(X_train,batch_size=batch_size,use_multiprocessing=True)
y_test_pred = model_best.predict(X_test,batch_size=batch_size,use_multiprocessing=True)
y_pure_preds = model_best.predict(X_pred,batch_size=batch_size,use_multiprocessing=True)

#Reverse map to original magnitude
X_train_orig = X_data[0:N]
y_train_orig = y_data[0:N]
X_test_orig = X_data[N:]
y_test_orig = y_data[N:]

X_pred_orig = mat2['input_tff_front']
y_pred_ref_orig = mat2['target_tff_front']


y_train_pred_flatten = np.reshape(y_train_pred, [y_train_pred.shape[0]*y_train_pred.shape[1], y_train_pred.shape[2]])
y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])

for sample in range(30):
    plt.figure()
    plt.plot(y_train_orig[sample,:], label='True')
    plt.plot(y_train_pred[sample][:, 0], label='Predict')
    plt.title('Training')
    plt.legend()
    
    
y_test_pred_flatten = np.reshape(y_test_pred, [y_test_pred.shape[0]*y_test_pred.shape[1], y_test_pred.shape[2]])
y_test_pred = scaler_y.inverse_transform(y_test_pred_flatten)
y_test_pred = np.reshape(y_test_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])

for sample in range(4):
    
    plt.figure()
    plt.plot(y_test_orig[sample,:], label='True')
    plt.plot(y_test_pred[sample][:, 0], label='Predict')
    plt.title('Testing')
    plt.legend()
    
y_pure_preds_flatten = np.reshape(y_pure_preds, [y_pure_preds.shape[0]*y_pure_preds.shape[1], y_pure_preds.shape[2]])
y_pure_preds = scaler_y.inverse_transform(y_pure_preds_flatten)
#y_pure_preds = np.reshape(y_pure_preds, [y_pred_ref.shape[0], y_pred_ref.shape[1], 1])
y_pure_preds = np.reshape(y_pure_preds, [y_pred_ref_map.shape[0],y_pred_ref_map.shape[1],y_pred_ref_map.shape[2]])

for sample in range(30):
    plt.figure()
    plt.plot(y_pred_ref_orig[sample,:],label='True')
    plt.plot(y_pure_preds[sample][:, 0], label='Predict')
    plt.title('Prediction')
    plt.legend()
    
#Save scaler


#joblib.dump(scaler_X, dataDir+'Trained_Fleet/scaler_X_021120_fleeet2_autoencoder_02112020_Final_Paper_FA_Full_Profile_Front_original.save')
#joblib.dump(scaler_y, dataDir+'Trained_Fleet/scaler_y_021120_fleeet2_autoencoder_02112020_Final_Paper_FA_Full_Profile_Front_original.save')





# scipy.io.savemat(dataDir+'Output_Data_Fleet_Training/New_Fleet_autoencoder_041120_FA_Final_paper_no_profile_00_Front_original_velocity_20_ELE.mat',
#                   {'y_train': y_train, 'y_train_orig': y_train_orig, 'y_train_pred': y_train_pred,
#                   'y_test': y_test, 'y_test_orig': y_test_orig, 'y_test_pred': y_test_pred,
#                   'y_pred_ref': y_pred_ref, 'y_pred_ref_orig': y_pred_ref_orig, 'y_pure_preds': y_pure_preds,
#                   'X_train': X_train, 'X_test': X_test, 'X_pred': X_pred})


#plot_model(model, to_file='Five_Axle.png', show_shapes=True, show_layer_names=True)




"""""

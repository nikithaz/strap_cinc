# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 21:13:00 2021

@author:Nikitha 20203108 
@affiliation:Systemic Change Group, Industrial Design, TU/e
"""
import tensorflow as tf
tf.random.set_seed(123)
import numpy as np
# tf.compat.v1.disable_eager_execution()
OPT = tf.keras.optimizers.Adamax(learning_rate=1e-3)

def get_model(num_classes): 
    input_layer = tf.keras.layers.Input((None,1))  #min 17
    x = tf.keras.layers.Conv1D(filters=8,kernel_size=11,strides=1,activation='relu')(input_layer)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=13,kernel_size=5,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=13,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=16,kernel_size=1,strides=1,activation='relu', name = "last")(x)   
    x = tf.keras.layers.Conv1D(filters=num_classes,kernel_size=1,strides=1,activation='softmax')(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    predictions = tf.keras.layers.Activation(activation='softmax')(x)
    model = tf.keras.Model(inputs = input_layer, outputs = predictions)
    print(model.summary())
    model.compile(optimizer = OPT, loss = 'categorical_crossentropy')
    return model

def get_model_cnc(num_classes): 
    input_layer = tf.keras.layers.Input((None,1))  #min 17
    x = tf.keras.layers.Conv1D(filters=64,kernel_size=8,strides=1,activation='relu')(input_layer)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=256,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=512,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=512,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=64,kernel_size=1,strides=1,activation='relu', name = "last")(x)   
    x = tf.keras.layers.Conv1D(filters=num_classes,kernel_size=1,strides=1)(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    predictions = tf.keras.layers.Activation(activation='sigmoid')(x)
    model = tf.keras.Model(inputs = input_layer, outputs = predictions)
    print(model.summary())
    model.compile(optimizer = OPT, loss = 'binary_crossentropy')
    return model


def get_model_base(num_classes):
    input_layer = tf.keras.layers.Input((None,1))  #min 230
    x = tf.keras.layers.Conv1D(filters=64,kernel_size=8,strides=1,activation='relu')(input_layer)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=128,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=256,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=512,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    
    x = tf.keras.layers.Conv1D(filters=128,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=128,kernel_size=3,strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(3)(x)
    
    x = tf.keras.layers.Conv1D(filters=16,kernel_size=1,strides=1,activation='relu', name='last')(x)
    
    
    x = tf.keras.layers.Conv1D(filters=num_classes,kernel_size=1,strides=1)(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    predictions = tf.keras.layers.Activation(activation='softmax')(x)
    
    model = tf.keras.Model(inputs = input_layer, outputs = predictions)
    print(model.summary())
    model.compile(optimizer = OPT, loss = 'mse')
    return model

def get_model_base_2d(num_classes):
    input_layer = tf.keras.layers.Input((None,None,1))  #min 2200
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(2,12),strides=1,activation='relu')(input_layer)
    x = tf.keras.layers.MaxPool2D(1,4)(x)
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(1,8),strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(1,4)(x)
    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(1,8),strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(1,2)(x)
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(1,4),strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(1,2)(x)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(1,4),strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(1,2)(x)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(1,4),strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(1,2)(x)
    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(1,4),strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(1,2)(x)
    x = tf.keras.layers.Conv2D(filters=218,kernel_size=(1,2),strides=1,activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(1,3)(x)
    
    x = tf.keras.layers.Conv2D(filters=16,kernel_size=1,strides=1,activation='relu', name='last')(x)
    
    x = tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPool2D()(x)
    
    predictions = tf.keras.layers.Activation(activation='softmax')(x)
    
    model = tf.keras.Model(inputs = input_layer, outputs = predictions)
    print(model.summary())
    model.compile(optimizer = OPT, loss = 'binary_crossentropy',metrics = 'accuracy')
    return model

if __name__ == "__main__":
    model_dim = "2D"
    if model_dim == "1D":
        classifier_loader = get_model_cnc
        signal_samples = [200,300, 500, 900]
    elif model_dim == "2D":
        classifier_loader = get_model_base_2d
        signal_samples = [2200]
    
    
    model = classifier_loader(2)
    
    N = 100
    
    
    
    for sample_size in signal_samples:
        if model_dim == "1D":
            x = np.random.randn(N,sample_size,1)
        elif model_dim == "2D":
            x = np.random.randn(N,2,sample_size,1)
            
        y = np.array([[1, 0] if i>0 else [0, 1] for i in np.random.randn(N)])
    
        
        model.fit(x,y,epochs=1,batch_size=16)
    
    model.predict(x)
#     #%% explainability
#     import matplotlib.pyplot as plt
#     signal = x[:1]
#     # predict = model.predict(signal)
#     # target_class = np.argmax(predict[0])
#     conv_layer = model.get_layer('last')
#     heatmap_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
#     # with tf.GradientTape() as tape:
#     #     grads = tf.gradients(model.output[:,target_class],last_conv.output) 
        
#     with tf.GradientTape() as gtape:
#         conv_output, predictions = heatmap_model(signal)
#         loss = predictions[:, np.argmax(predictions[0])]
#         grads = gtape.gradient(loss, conv_output)
#         pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1))
        
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     max_heat = np.max(heatmap)
#     if max_heat == 0:
#         max_heat = 1e-10
#     heatmap /= max_heat
#     heatmap = np.resize(heatmap,signal.shape[1])
#     plt.plot(np.ndarray.flatten(signal))
#     plt.plot(heatmap)

# #%%

#     # pooled_grads = tf.keras.backend.mean(grads,axis=(0,1))
#     # iterate = tf.keras.backend.function([model.input],[pooled_grads,last_conv.output[0]])
#     pooled_grads_value,conv_layer_output = iterate(signal)
#     for i in range(conv_layer_output.shape[-1]):
#         conv_layer_output[:,i] *= pooled_grads_value[i]
#     heatmap = np.mean(conv_layer_output,axis=-1)
#     heatmap = np.resize(heatmap,signal.shape[1])
#     plt.plot(signal[0])
#     plt.plot(heatmap)

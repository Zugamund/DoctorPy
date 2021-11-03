import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, MaxPooling2D, Flatten

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import re
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.preprocessing import zoom_preprocessing, brght_preprocessing, simple_preprocessing


def conf_mat(x, y, model, process=simple_preprocessing):

    """
    Returns the confusion matrix in the form of a dataframe given the features x, targets y, the model and the preprocessing function.
    """
    
    test = np.array([process(im) for im in x.values])
    preds = model.predict(test.reshape(-1, 256, 256, 1)).argmax(axis=1)
    
    labels = ['COVID-19', 'NORMAL', 'Viral Pneumonia']
    
    y_true = pd.Series([labels[lab] for lab in y])
    y_pred = pd.Series([labels[lab] for lab in preds])
    
    return pd.crosstab(y_true, y_pred, rownames=['Actual labels'], colnames=['Predictions'])


def build_simple_model(lr=0.0003):

    """
    Builds a simple CNN model and compiles it.
    """
    
    num_classes = 3
    
    model = Sequential()
    
    model.add(Input((256, 256, 1)))

    model.add(Conv2D(filters=32, 
                     kernel_size=(5,5), 
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=16, 
                     kernel_size=(3,3), 
                     padding='valid', 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(rate=0.2))
    model.add(Flatten())

    model.add(Dense(units = 128, 
                    activation ='relu'))             
    model.add(Dense(units = num_classes, 
                    activation ='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def build_vgg(lr=0.0003):

    """
    Builds a VGG model and compiles it.
    """
    
    num_classes = 3
    model = Sequential()
    
    model.add(Input((256, 256, 1)))
       

    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     padding="same", 
                     activation="relu"))
    
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     padding="same", 
                     activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=128, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    model.add(Conv2D(filters=128, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=256, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    
    model.add(Conv2D(filters=256, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    
    model.add(Conv2D(filters=256, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    
    model.add(Conv2D(filters=512, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    
    model.add(Conv2D(filters=512, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    
    model.add(Conv2D(filters=512, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    
    model.add(Conv2D(filters=512, 
                     kernel_size=(3,3), 
                     padding="same", 
                     activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096,
                    activation="relu"))
    model.add(Dense(units=4096,
                    activation="relu"))
    model.add(Dense(units=num_classes, 
                    activation="softmax"))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
    return model


def batch_generator(x, y, batch_size=32, preprocess='simple', size=(256,256)):
    
    """
    Will generate a dataset based on the images whose path are contained in x and whose labels are contained in y.
    It is possible to specify the batch size which is 32 by default.
    """
    
    print('Preprocessing the images...')
    
    
    if preprocess == 'simple':
        X = [simple_preprocessing(path, size).reshape(size[0], size[1], -1) for path in x]
    elif preprocess == 'brght':
        X = [brght_preprocessing(path, size).reshape(size[0], size[1], -1) for path in x]
    elif preprocess == 'zoom':
        X = [zoom_preprocessing(path, size).reshape(size[0], size[1], -1) for path in x]
    else:
        pass
        
    print('Creating the dataset...')
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    
    print('Resizing the images...')
    
    dataset = dataset.batch(batch_size)
    print('Done!')
    
    return dataset

## GradCAM



def find_layer(model,num_conv=0):

    """
    Locates the num_conv-th convolutional layer of a given model.
    """
    
    layer_names = reversed(model.layers)
    conv_names= [layer_name.name for layer_name in layer_names if re.findall(r'conv', layer_name.name)]
    
    return conv_names[num_conv]

def compute_gradcam(model, image, alpha=0.9):

    """
    Computes the Grad-CAM of a model given an input image.
    """
    
    layer_name=find_layer(model)
    gradModel = Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])
    
    y_pred = model.predict(image.reshape(-1,256,256,1))
    preds = np.argmax(y_pred,axis = 1)
    
    with tf.GradientTape() as tape:

        inputs = tf.cast(image.reshape(-1,256,256,1), tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, preds[0]]
    grads = tape.gradient(loss, convOutputs)
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads

    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (256,256)
    heatmap = cv2.resize(cam.numpy(), (w, h))

    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) 
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    output = cv2.addWeighted((image.reshape(256,256)*255).astype('uint8'), 1 - alpha, heatmap , alpha, 0)
    return output

def plot_gradcam(model, image, y_true=None, alpha=0.9):

    """
    Plots the original image next to its Grad-CAM heatmap altered version.
    """
    
    output = compute_gradcam(model, image, alpha)
    label_list = ['COVID-19', 'NORMAL', 'Viral Pneumonia']
    y_pred = model.predict(image.reshape(-1,256,256,1))
    
    
    if y_true is not None:
        fig = plt.figure(figsize=(15,6))


        plt.subplot(131)

        plt.imshow(image.reshape(256,256), cmap='gray')
        plt.title(f"Image : {label_list[y_true]}")
        plt.axis('off')


        plt.subplot(132)

        ax = plt.subplot(132)
        im = ax.imshow(output,cmap='turbo')
        plt.title('Prédiction : {}, Probabilité : {}%'.format(label_list[np.argmax(y_pred)], round(y_pred.max()*100)))
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(visualize_heat_grad(image, out=output))
        plt.title('Heatmap blend')
        plt.axis('off')
        plt.show();
        
        
    else:
        fig = plt.figure(figsize=(15,6))
        plt.subplot(131)

        plt.imshow(image.reshape(256,256), cmap='gray')
        plt.title("Image")
        plt.axis('off')
        plt.subplot(132)
        ax = plt.subplot(132)
        im = ax.imshow(output,cmap='turbo')
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(visualize_heat_grad(image, out=output))
        plt.title('Heatmap blend')
        plt.axis('off')
        plt.show();
    
    return fig

def visualize_heat_grad(img, shape=(256, 256), model = None, out=None):

    """
    A function to blend a red filtered heatmap of a Grad-CAM on the original image.
    """


    if out is None:
        out = compute_gradcam(model, img)
    
    img_dim = np.empty((shape[0], shape[1], 3))
    
    for i in range(3):
        img_dim[:,:,i] = (img.reshape(shape[0], shape[1], -1)/256).mean(axis=2)
    
    img_dim[:, :, 0] = np.clip(out/256 + img_dim[:, :, 0], 0, 1)
    return img_dim

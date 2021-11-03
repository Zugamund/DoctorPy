import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPool2D,UpSampling2D
from tensorflow.keras.layers import Input,Dropout,concatenate
from tensorflow.keras.models import Model

from scipy import ndimage as ndi

import cv2
import numpy as np

from utils.preprocessing import resize_img, img_zoom, adjust_brightness_func

def get_model():

    """
    Creates the architecture for the unet model
    """

    inputs = Input((256, 256, 1))

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs, conv10)

    return model

def build_model(lr=1e-4, path='models/Unet_best.h5'):

    """
    Builds the unet model, loading the best weights.
    """

    model = get_model()

    opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.load_weights(path)

    return model


def cleaning_mask(unet_mask1,unet_mask2):

    """
    Makes the unet predictions smoother
    """

    kernel=np.ones((5, 5), 'uint8')
    dilate_mask1=cv2.dilate(unet_mask1,kernel ,iterations=2)
    dilate_mask2=cv2.dilate(unet_mask2,kernel ,iterations=2)
    thresh1 = cv2.adaptiveThreshold(dilate_mask1, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,253, 1)
    thresh2 = cv2.adaptiveThreshold(dilate_mask2, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,253, 1)
    label_objects1, nb_labels1 = ndi.label(thresh1)
    sizes1 = np.bincount(label_objects1.ravel())
    if len(sizes1)> 2:
        mask_sizes1 = sizes1 >= np.sort(sizes1)[-3]
        mask_sizes1[0] = 0
    else :
        mask_sizes1=sizes1

    label_objects2, nb_labels2 = ndi.label(thresh2)
    sizes2 = np.bincount(label_objects2.ravel())
    if len(sizes2)> 2: 
        mask_sizes2 = sizes2 >= np.sort(sizes2)[-3]
        mask_sizes2[0] = 0
    else :
        mask_sizes2=sizes2

    cleaned1 = ndi.binary_fill_holes(mask_sizes1[label_objects1])
    cleaned2 = ndi.binary_fill_holes(mask_sizes2[label_objects2])
    if (cleaned1.sum()>cleaned2.sum()):
        cleaned=cleaned1
    else:
        cleaned=cleaned2
    return cleaned


def normalize_array(arr, lower=0, upper=255):

    """
    Will normalize the values of the array
    """

    arr = (upper - lower) * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr


def predict_unet(img1, img2, model=None):

    """
    Will return the unet predictions given the two unet results on our two preprocessings
    """

    if model is None:
        model = build_model()

    pred_unet1 = model.predict(img1, verbose=0)
    pred_unet2 = model.predict(img2, verbose=0)
    return (normalize_array(pred_unet1, 0, 255)).reshape(256, 256), (normalize_array(pred_unet2, 0, 255)).reshape(256,
                                                                                                                  256)

def final_pp(path_image, model=None):

    """
    The unet pipeline chosen here, using two predictions that will be smoothed.
    Those predictions will be based on the zoomed and resized image, with and without a brightness preprocessing.
    """

    img = img_zoom(path_image)
    image_1 = resize_img(img).reshape(-1, 256, 256)
    image_2 = resize_img(adjust_brightness_func(img)).reshape(-1, 256, 256)
    image = resize_img(img)

    pred_unet1, pred_unet2 = predict_unet(image_1, image_2, model)

    predictions = cleaning_mask(pred_unet1.astype('uint8'), pred_unet2.astype('uint8'))

    return (image * predictions).reshape(256, 256)
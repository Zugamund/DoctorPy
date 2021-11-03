import pandas as pd
import numpy as np
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

import cv2
import seaborn as sns

from PIL import Image


from utils.unet_pp import *
from utils.preprocessing import *
from utils.model_builds import build_vgg, plot_gradcam

labels_list=['COVID-19', 'NORMAL', 'Viral Pneumonia']


covid_path = 'data/COVID-19/'
covid_files = [covid_path + file for file in os.listdir(covid_path) if file[-3:] == 'png']

normal_path = 'data/NORMAL/'
normal_files = [normal_path + file for file in os.listdir(normal_path) if file[-3:] == 'png']

vp_path = 'data/Viral Pneumonia/'
vp_files = [vp_path + file for file in os.listdir(vp_path) if file[-3:] == 'png']

process_df = pd.read_csv('data/streamlit/all_preprocesses.csv', index_col=0)
eda_df = pd.read_csv('data/streamlit/eda_st.csv', index_col=0)

target_train = pd.read_csv('data/streamlit/train_labels.csv', index_col=0)
target_test = pd.read_csv('data/streamlit/test_labels.csv', index_col=0)
target_val = pd.read_csv('data/streamlit/val_labels.csv', index_col=0)

tmp = target_train.iloc[0].copy()
target_train.iloc[0] = target_train.iloc[1].copy()
target_train.iloc[1] = tmp

tmp_2 = target_test.iloc[0].copy()
target_test.iloc[0] = target_test.iloc[2].copy()
target_test.iloc[2] = tmp_2

train_lda = np.load('data/streamlit/train_lda.npy', allow_pickle=True)
test_lda = np.load('data/streamlit/test_lda.npy', allow_pickle=True)

train_lda_unet = np.load('data/streamlit/lda_data_train_u.npy', allow_pickle=True)
test_lda_unet = np.load('data/streamlit/lda_data_test_u.npy', allow_pickle=True)
target_train_unet = np.load('data/streamlit/lda_target_train_u.npy', allow_pickle=True)
target_test_unet = np.load('data/streamlit/lda_target_test_u.npy', allow_pickle=True)

X_embedded = np.load('data/streamlit/tsne_train.npy', allow_pickle=True)




def brght_ctrst(img_path):

    """
    Takes a filepath as input and returns the brightness then contrast enhanced image.
    """

    return standardize_img_contrast(standardize_img_brightness(img_path), path=False)


def ctrst_brght(img_path):

    """
    Takes a filepath as input and returns the contrast then brightness enhanced image.
    """

    return standardize_img_brightness(standardize_img_contrast(img_path), path=False)




def clahe_preprocessing(img_path, clipLimit=8, tileGridSize=(11, 11)):

    """
    Applies a CLAHE filter preprocessing to an image that will first be enhanced and zoomed in on given its path.
    """

    clahe = cv2.createCLAHE(clipLimit, tileGridSize)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = standardize_img_brightness(standardize_img_contrast(img, path=False), path=False)
    img = img_zoom(img, path=False)
    img = clahe.apply(img.astype(np.uint8))
    img = resize_img(img)

    return img


def plot_images(data=[covid_files, normal_files, vp_files], labels=labels_list):

    """
    Plots random images for each class.
    """

    fig, ax = plt.subplots(3, 3, figsize=(16, 16))
    f1, f2, f3 = data
    l1, l2, l3 = labels

    for i in range(3):
        img1 = cv2.imread(f1[np.random.choice(len(f1))])
        img2 = cv2.imread(f2[np.random.choice(len(f2))])
        img3 = cv2.imread(f3[np.random.choice(len(f3))])

        ax.flat[3 * i].imshow(img1, cmap='gray')
        ax.flat[3 * i].axis('off')
        ax.flat[3 * i].set_title(l1)

        ax.flat[3 * i + 1].imshow(img2, cmap='gray')
        ax.flat[3 * i + 1].axis('off')
        ax.flat[3 * i + 1].set_title(l2)

        ax.flat[3 * i + 2].imshow(img3, cmap='gray')
        ax.flat[3 * i + 2].axis('off')
        ax.flat[3 * i + 2].set_title(l3)
    return fig


def plot_img_preprocessing(data=[covid_files, normal_files, vp_files], labels=labels_list,
                           func=img_zoom):

    """
    Plots random images for each class right above their preprocessed versions.
    """

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    f1, f2, f3 = data
    l1, l2, l3 = labels

    img1 = f1[np.random.choice(len(f1))]
    img2 = f2[np.random.choice(len(f2))]
    img3 = f3[np.random.choice(len(f3))]

    ax.flat[0].imshow(cv2.imread(img1, cv2.IMREAD_GRAYSCALE), cmap='gray')
    ax.flat[0].axis('off')
    ax.flat[0].set_title(l1)

    ax.flat[1].imshow(cv2.imread(img2, cv2.IMREAD_GRAYSCALE), cmap='gray')
    ax.flat[1].axis('off')
    ax.flat[1].set_title(l2)

    ax.flat[2].imshow(cv2.imread(img3, cv2.IMREAD_GRAYSCALE), cmap='gray')
    ax.flat[2].axis('off')
    ax.flat[2].set_title(l3)

    ax.flat[3].imshow(func(img1), cmap='gray')
    ax.flat[3].axis('off')
    ax.flat[3].set_title(l1)

    ax.flat[4].imshow(func(img2), cmap='gray')
    ax.flat[4].axis('off')
    ax.flat[4].set_title(l2)

    ax.flat[5].imshow(func(img3), cmap='gray')
    ax.flat[5].axis('off')
    ax.flat[5].set_title(l3)

    return fig


def load_eda(x):

    """
    Loads the dataframe for the EDA.
    """

    fig, ax = plt.subplots()
    ax = sns.histplot(data=eda_df, x=x, hue='label', alpha=0.8)
    return fig

def plot_lda():

    """
    Plots the results of the LDA applied to our images.
    """

    fig = plt.figure(figsize=(20, 15))
    plt.subplot(211)
    sns.scatterplot(x=train_lda[:, 0], y=train_lda[:, 1],
                    hue=target_train[:1000].values.reshape(-1), s=60)
    plt.title('Train')

    plt.subplot(212)
    sns.scatterplot(x=test_lda[:, 0], y=test_lda[:, 1],
                    hue=target_test[:200].values.reshape(-1), s=60)
    plt.title('Val')
    return fig

def plot_unet_lda():

    """
    Plots the results of the LDA applied to our U-Net preprocessed images.
    """

    fig = plt.figure(figsize=(20, 15))
    plt.subplot(211)
    sns.scatterplot(x=train_lda_unet[:, 0], y=train_lda_unet[:, 1],
                    hue=target_train_unet.reshape(-1), s=60)
    plt.title('Train')

    plt.subplot(212)
    sns.scatterplot(x=test_lda_unet[:, 0], y=test_lda_unet[:, 1],
                    hue=target_test_unet.reshape(-1), s=60)
    plt.title('Val')
    return fig

def plot_tsne():

    """
    Plots a Manifold Learning model's results when applied to our data.
    """

    fig = plt.figure()
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=target_train.values.reshape(-1), alpha=0.3)
    return fig

def plot_boxplot_preprocess(criterion, versions):

    """
    Will plot the boxplot of the selected criterion for the selected versions.
    """

    dic_crit = {
        'Intensité sur le bord': 'edge_intensity',
        'Moyenne': 'mean',
        'Ecart-type': 'std'
    }

    versions_dict = {
        'Originale': 'old',
        'Zoomée': 'zoom',
        'Filtre d\'intensité': 'brightness',
        'Filtre de contraste': 'contrast',
        'Contraste + intensité': 'ctrst+brght',
        'Intensité + contraste': 'brght+ctrst',
        'Filtre CLAHE': 'clahe'
    }

    version_choices = [versions_dict[v] for v in versions]

    df = process_df[process_df['version'].isin(version_choices)]

    fig = sns.catplot(kind='box', x='version', y=dic_crit[criterion], hue='label', data=df, height=8, aspect=1.6)

    return fig


def display_unet(model=None):

    """
    Plots a random image next to its segmented version.
    """

    l = [covid_files, normal_files, vp_files]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        path = l[i][np.random.choice(len(l[i]))]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ax.flat[i].imshow(img, cmap='gray')
        ax.flat[i+3].imshow(final_pp(path, model), cmap='gray')
        ax.flat[i].axis('off')
        ax.flat[i+3].axis('off')
    return fig

def read_img(file):

    """
    Reads and decodes an image file and converts it to black and white if necessary.
    """

    img = np.asarray(Image.open(file))
    if len(img.shape) >= 3:
        bw_img = img.mean(axis=2)
        return bw_img
    else:
        return img


def predict(file):

    """
    Takes a file as input and puts it through the whole pipeline to predict the class of the image and display its Grad-CAM prediction.
    """

    img = read_img(file)

    img = img_zoom(img, path=None)
    img = adjust_brightness_func(img)
    img = resize_img(img)
    img = img.reshape(1, 256, 256, 1)

    model = build_vgg()
    model.load_weights('models/saved_weights/VGG_weight_z_and_b_pp.h5')
    fig = plot_gradcam(model, img, alpha=0.7)
    pred = model.predict(img)
    label = labels_list[pred.argmax(axis=1)[0]]
    p = pred.max()

    return fig, label, p

def load_csv(path):

    df = pd.read_csv(path, index_col=0)

    return df
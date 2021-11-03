import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from PIL import Image
from PIL import ImageEnhance


def resize_img(img, size=(256, 256)):

    """
    Will resize an image given its target size.
    """


    height = img.shape[0]
    width = img.shape[1]


    if height < size[0] or width < size[1]: # If the image needs to be enlarged
        img = (cv2.resize(img.reshape(height, width), dsize=size, interpolation=cv2.INTER_CUBIC))

    elif height > size[0] or width > size[1]: # If the image needs to be shrinked
        img = (cv2.resize(img.reshape(height, width), dsize=size, interpolation=cv2.INTER_AREA))

    return img



def img_zoom(img, threshold=None, ratio=2/3, path=True):

    """
    Will zoom on the image by cutting off the sequence of columns below the threshold intensity on each side.

    If no threshold is specified, it will be calculated by a ratio with respect to the image's mean intensity.
    The default ratio is 2/3.
    """

    if path:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # Calculating the threshold if None
    if threshold is None:
        threshold = np.mean(img)*ratio

    # mini will keep in memory the highest index (from the left or the right) of the sequence of columns below the threshold
    mini = 0
    height = img.shape[0]
    width = img.shape[1]

    # Checking which columns are too dark from the left to the middle
    for i in range(width//2):


        if i > mini+1:
            break; # Making sure we don't delete columns in the middle of the image
        i_mean = np.mean(img[:, i])
        i_mean_reverse = np.mean(img[:, -i])
        if threshold > i_mean or threshold > i_mean_reverse:
            mini = i

    # Slicing the image and keeping the ratio, thus performing a zoom on the lungs
    return img[mini : height-mini, mini : height-mini]


def standardize_img_brightness(img, mean_target=120, path=True):

    """
    Will apply a translation to adjust the brightness of the given image to reach a certain mean.
    """

    if path:
        img = img_zoom(img)

    translator = mean_target - img.mean()
    return np.clip(img + translator, 0, 255)


def adjust_brightness_func(img, factor=None, target=120):
    """
    Takes an image as input and returns a "normalized" (intensity wise) image around the provided target
    Additionnally, it is possible to specify the enhancing factor for custom use.
    """

    if factor is None:
        factor = target / img.mean()

    enhancer = ImageEnhance.Brightness(Image.fromarray(img))
    # The image is now "enhanced" in PIL format

    # Returning the array version of the enhanced image
    return np.clip(img + (target - img.mean()), 0, 255)


def standardize_img_contrast(img, std_target=30, path=True):

    """
    Will apply a multiplication to adjust the contrast of the given image to reach a certain standard deviation.
    """


    if path:
        img = img_zoom(img)

    factor = np.sqrt(std_target/img.std())
    return np.clip(img*factor, 0, 255)


def lda_test(data_train, data_test, target_train, target_test, plot=True):

    """
    Will apply an LDA algorithm to reduce dimensionality and a Random Forest model to evaluate prediction on the reduced dataset.
    If the score is high, the biases are high.
    """


    lda = LDA()
    d_lda = lda.fit_transform(data_train, target_train)
    test_lda = lda.transform(data_test)

    if plot:
        fig = plt.figure(figsize=(20, 10))
        plt.subplot(121)
        sns.scatterplot(x=d_lda[:, 0], y=d_lda[:, 1],
                        hue=target_train, s=60)
        plt.title('Train')

        plt.subplot(122)
        sns.scatterplot(x=test_lda[:, 0], y=test_lda[:, 1],
                        hue=target_test, s=60)
        plt.title('Val')

    clf = RandomForestClassifier(random_state=42)
    clf.fit(d_lda, target_train)

    score_train = clf.score(d_lda, target_train)
    score_test = clf.score(test_lda, target_test)


    return score_test, score_train, fig


def thresh_inf(img, threshold=50, size=(256,256)):

    """
    Returns the frequency of pixels below a certain intensity threshold
    """

    return np.sum(img < threshold)/(size[0]*size[1])


def edge_brightness(img, thres=30):

    """
    Will return the average intensity of the pixels located on the edge of the image.
    The threshold will determine how many pixels are considered on the edge.
    """

    edge = 0
    center = 0
    for i in range(256):
        for j in range(256):
            v_center = (i >= thres and i < 256-thres)
            h_center = (j >= thres and j < 256-thres)
            if v_center and h_center:
                center += img[i, j]
            else:
                edge += img[i, j]
    return edge/(256*256), center/(256*256)


def brght_preprocessing(img_path, size=(256,256)):

    img = img_zoom(img_path)
    img = adjust_brightness_func(img)
    img = resize_img(img, size)

    return img

def zoom_preprocessing(img_path, size=(256,256)):

    img = img_zoom(img_path)
    img = resize_img(img, size)

    return img

def simple_preprocessing(img_path, size=(256,256)):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = resize_img(img, size)

    return img
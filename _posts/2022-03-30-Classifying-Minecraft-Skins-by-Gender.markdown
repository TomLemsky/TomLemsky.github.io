---
layout: post
title:  "Classifying Minecraft skins by gender"
date:   2022-03-30 17:00:41 +0100
---

For this project I wanted to experiment with a dataset that has not been
used before. I wanted to use images with few pixels to keep the filesizes small and easy to handle. I decided to use Minecraft skins as the dataset to
experiment with and found a website that labeled each skin with a gender
(male, female, interchangeable and other) and several tags. I wrote a
quick webscraper in Python to collect several thousand skins and their
labels.

The dataset
===========

|![Average of all images in dataset[Average of all images]{label="fig:sub1"}](/images/minecraft-plots/average of 98661.png) |
|:--:|
| *Average of all images in dataset* |

| ![Average of images labeled male[]{label="fig:sub2"}](/images/minecraft-plots/male average of 10000.png) | ![Average of images labeled female[]{label="fig:sub2"}](/images/minecraft-plots/female average of 26000.png) |
|:--:|
| *Average of images labeled male* | *Average of images labeled female* |

In the popular video game Minecraft the player can customize their
player avatar by creating a $$32 \times 64$$ color image called a *skin*
that serves as a texture for the player 3D model: The upper left
quadrant of the image gets mapped to the players head and the lower half
is for the legs, arms and body. The upper right region contains an
optional second head layer that can be used among other things for hair,
hats and glasses.

A skin can contain transparency but I removed the alpha channel to
simplify the data, since it will most of the time be completely opaque
or completely transparent. Therefore a skin can be represented by a
$$32 \times 64 \times 3 = 6144$$-dimensional vector. The dataset I
collected contains $$126\ 104$$ images in the *male* class, but only
$$35\ 346$$ images in the *female* class. To balance the classes, I
partitioned the images into a training set of $$50\ 000$$ images, a
validation set of size $$5\ 000$$ and a test set with $$5\ 000$$ images,
each with equal proportions of the classes. Duplicate skins were removed using a hash table.

Exploration of the data
-----------------------

To get a feel for the data, I computed the average of the images in the
dataset as well as the average for the male and female classes.
Interestingly, the average image for the female class contains very sharp eyes
and eyelashes, because apparently many female skins use exactly the same eye region. The average male eye region is much blurrier, because it contains more different kinds of eyes: The two-pixel eyes of the default skins and also larger eye versions or eyes that are placed at a different heights.

Using PCA on the images I found that the first principal component accounts
for 21.8% of the variance and only the first seven principal components
are needed to explain 50% of the variance. To explain 90% of the
variance, 326 components are required. The first principal component corresponds to the brightness of the skin and is mostly independent from unused regions and the second head layer. The second component seems to be very correlated to gender (see the plot in the next section), but also seems to be related to the color of the hair and clothes. As my below experiment with logistic regression and SVMs shows, the first two principal components can already be used to classify about 69% of skins correctly, but not more.

|![The first nine principal components of the dataset[]{label="fig:eigen"}](/images/minecraft-plots/9 pca.png) |
|:--:|
| *The first nine principal components of the dataset* |

Classification
==========================

| ![[]{label="fig:pca-lr"}](/images/minecraft-plots/gender per first two pca components_e.png) |
|:--:|
| *Visualization of the decision boundary of logistic regression trained on the first two principal components. The X and Y axes represent the first and second principal component. (Blue dots: skins labeled male; Pink dots: skins labeled female; Line: decision boundary of logistic regression)*|

To compare different approaches, I trained several machine learning models to classify the
images in the dataset into the *male* and *female* classes. The image above shows the
decision boundary of a logistic regression classifier trained on only the first two principal components.
The results of all the algorithms I tried can be found in the following table.
The best classical model with an accuracy of 90.0% was a Support Vector Machine with a
Radial Basis Function kernel, with the dimensionality of the data reduced to the first
500 principal components. SVM on whole images achieved a higher training score,
but it overfitted more, causing a lower test accuracy.

Because the images are quite small, I used a simple Convolutional Neural
Network with two 3x3 convolutional and two Dense Layers. Dropout was
used to prevent overfitting and aid generalization. This network reached
an accuracy of 95% on the test set, halving the error rate of the best
traditional model.


| Model                                                  | Accuracy on training set | Accuracy on test set |
|--------------------------------------------------------|--------------------------|----------------------|
| Nearest Centroid                                       | 0.713                    | 0.704                |
| 9-Nearest-Neighbor                                     | 0.818                    | 0.798                |
| 3-Nearest-Neighbor                                     | 0.878                    | 0.793                |
| 1-Nearest-Neighbor                                     | 1.000                    | 0.758                |
| 3-Nearest-Neighbor on first 100 principal components   | 0.930                    | 0.866                |
| Logistic Regression on first 2 principal components    | 0.684                    | 0.678                |
| SVM with RBF kernel on first 2 principal components    | 0.698                    | 0.691                |
| SVM with RBF kernel on first 50 principal components   | 0.883                    | 0.859                |
| SVM with RBF kernel on first 500 principal components  | 0.906                    | 0.900                |
| SVM with RBF kernel on whole images                    | 0.927                    | 0.896                |
| 4 layer ConvNet                                        | 0.960               | **0.952**           |

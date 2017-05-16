
#Some more Computational Photography: Merging and Blending Images using Gaussian and Laplacian Pyramids

The following problem appeared as an assignment in the **coursera course Computational Photography (by Georgia Tech, 2013)**. The following description of the problem is taken directly from the assignment's description.
   

## Introduction

In this article, we shall be putting together a *pyramid blending pipeline* that will allow us to blend any two images with a mask. The blending procedure takes the two images and the mask, and splits them into their red, green and blue channels. It then blends each channel separately.

Then a **laplacian pyramid** will be constructed for the two images. It will then scale the mask image to the range [0,1] and construct a **gaussian pyramid** for it. Finally, it will blend the two pyramids and collapse them down to the output image.

Pixel values of 255 in the *mask* image are scaled to the value 1 in the mask pyramid, and assign the strongest weight to the image labeled ‘white’ during blending. Pixel values of 0 in the mask image are scaled to the value 0 in the *mask* pyramid, and assign the strongest weight to the image labeled ‘black’ during blending.

The following figures describe the *pyramid blending process* with couple of sample images.

![png](https://sandipanweb.files.wordpress.com/2017/05/im113.png?w=871)

![png](https://sandipanweb.files.wordpress.com/2017/05/im210.png?w=896&h=285)

## Reduce

This function takes an image and *subsamples* it down to a quarter of the size (dividing the height and width by two). Before we subsample the image, however, we need to first smooth out the image.

The following **5x5 generating kernel** will be used for *convolution*, for the *reduce* and later for the *expand* function.

![png](https://sandipanweb.files.wordpress.com/2017/05/kern.png)

## Expand

This function takes an image and *supersamples* it to four times the size (multiplying the height and width by two). After increasing the size, we have to interpolate the missing values by running over it with a *smoothing filter*.

## Gaussian Pyramid

This function takes an image and builds a pyramid out of it. The first layer of this pyramid is the original image, and each subsequent layer of the pyramid is the reduced form of the previous layer. Within the code, these pyramids are represented as lists of arrays, so pyramid = [layer0, layer1, layer2, ...]. The reduce function implemented in the previous part needs to be used in order to implement this function.

## Laplacian Pyramid

This function takes a *gaussian pyramid* constructed by the previous function, and turns it into a *laplacian pyramid*. 
Like with *gaussian pyramids*, *laplacian pyramids* are represented as lists and the expand function implemented in the previous part needs to be used in order to implement this function.

## Blend

In this part, the *pipeline* will be set up by implementing the actual *blend* function, and then implementing a *collapse* function that will allow us to disassemble our laplacian pyramid into an output image.

*Blend* function takes three pyramids: 

* white - a *laplacian pyramid* of an image
* black - a *laplacian pyramid* of another image
* mask - a *gaussian pyramid* of a mask image

It should perform an *alphablend* of the two *laplacian pyramids* according to the mask pyramid. So, we need to blend each pair of layers together using the mask of that layer as the weight. Pixels where the mask is equal to 1 should be taken from the
white image, pixels where the mask is 0 should be taken from the black image. Pixels with value 0.5 in the mask should be an equal blend of the white and black images, etc. 

## Collapse

This function is given a *laplacian pyramid*, and is expected to *'flatten'* it to an image. We need to take the top layer, expand it, and then add it to the next layer. This results in a pyramid of one level less than we started with. We continue this process until we are left with only a single layer.

## Results

The following figures show a few pairs of input images, the masks, the blended output images, along with the *gassian* and *laplacian* pyramids.

### 1. Input Images

![png](https://sandipanweb.files.wordpress.com/2017/05/i1.png)

### Gaussian Pyramids

![jpeg](https://sandipanweb.files.wordpress.com/2017/05/sample_gauss_pyr.jpg)

### Laplacian Pyramids

![jpeg](https://sandipanweb.files.wordpress.com/2017/05/sample_laplace_pyr.jpg)


### Blended Laplacian Pyramids

![jpeg](https://sandipanweb.files.wordpress.com/2017/05/sample_outpyr.jpg)


### Blended Output Image

![png](https://sandipanweb.files.wordpress.com/2017/05/o1.png)


### 2. Input Images

![png](output_21_0.png)



### Gaussian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample1_laplace_pyr.jpg', width=400)




![jpeg](output_23_0.jpe)



### Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample1_laplace_pyr.jpg', width=400)




![jpeg](output_25_0.jpe)



### Blended Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample1_outpyr.png',  width=200)




![png](output_27_0.png)



### Blended Output Image


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\o2.png')




![png](output_29_0.png)



### Input Images


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\i3.png')




![png](output_31_0.png)



### Gaussian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample2_gauss_pyr.png', width=400)




![png](output_33_0.png)



### Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample2_laplace_pyr.png', width=400)




![png](output_35_0.png)



### Blended Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample2_outpyr.png',  width=200)




![png](output_37_0.png)



### Blended Output Image


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\o3.png')




![png](output_39_0.png)



### Input Images


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\i4.png')




![png](output_41_0.png)



### Gaussian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample3_gauss_pyr.png', width=400)




![png](output_43_0.png)



### Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample3_laplace_pyr.png', width=400)




![png](output_45_0.png)



### Blended Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample3_outpyr.png',  width=200)




![png](output_47_0.png)



### Blended Output Image


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\o4.png')




![png](output_49_0.png)



The next example shows how the pyramid pipeline can be used to **edit** an image **background**.

### Input Images


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\i5.png')




![png](output_52_0.png)



### Gaussian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample4_gauss_pyr.png', width=400)




![png](output_54_0.png)



### Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample4_laplace_pyr.png', width=400)




![png](output_56_0.png)



### Blended Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample4_outpyr.png',  width=200)




![png](output_58_0.png)



### Blended Output Image


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\o5.png')




![png](output_60_0.png)



As can be seen from the output image above, the pyramid blending pipleine did a good job in blending the two images.

### Input Images


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\i6.png')




![png](output_62_0.png)



### Gaussian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample6_gauss_pyr.png', width=400)




![png](output_64_0.png)



### Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample6_laplace_pyr.png', width=400)




![png](output_66_0.png)



### Blended Laplacian Pyramids


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\sample6_outpyr.png',  width=200)




![png](output_68_0.png)



### Blended Output Image


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw3_solved\\o6.png')




![png](output_70_0.png)



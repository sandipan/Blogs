Solving Some Image Processing and Computer Vision Problems With Python Libraries {.p-name}
================================================================================

In this article, a few image processing / computer vision problems and
their solutions with python libraries (scikit-image, PIL…

* * * * *

### Solving a few Image Processing Problems with Python {#fc42 .graf .graf--h3 .graf--leading .graf--title name="fc42"}

#### Solving a few Image Processing and Computer Vision Problems with Python libraries {#88b4 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="88b4"}

In this article, a few image processing/computer vision problems and
their solutions with python libraries (*scikit-image, PIL,
opencv-python*) will be discussed. Some of the problems are from the
exercises from [this
book](https://www.linkedin.com/pulse/my-book-hands-on-image-processing-python-sandipan-dey?articleId=6478757946401619968#comments-6478757946401619968&trk=prof-post)
(available [on
Amazon](https://www.amazon.com/Hands-Image-Processing-Python-interpretation-ebook/dp/B07J664F9S/ref=tmm_kin_swatch_0?_encoding=UTF8&qid=1546423691&sr=8-1)).
Here is the [GitHub
repository](https://github.com/PacktPublishing/Hands-On-Image-Processing-with-Python)
with the codes from the book and my
[blog](https://sandipanweb.wordpress.com/2018/07/30/some-image-processing-problems/)on
WordPress and a
[playlist](https://www.youtube.com/playlist?list=PLl2zq7-M9E2uD8u0X19ZDtg7-uYhyvThA)on
youtube. Also, here is the [github
repository](https://github.com/PacktPublishing/Python-Image-Processing-Cookbook)
of the codes for my new book (available on
[Amazon](https://www.amazon.com/Python-Image-Processing-Cookbook-processing/dp/1789537142)).

### Wave Transform {#c21d .graf .graf--h3 .graf-after--p name="c21d"}

1.  Use *scikit-image’s warp()* function to implement the
    *wave*transform.
2.  Note that wave transform can be expressed with the following
    equations:

![](https://cdn-images-1.medium.com/max/800/1*Rj44tPIsshrYMsoGy-3tuA.png)

We shall use the *mandrill* image to implement the wave transform. The
next python code fragment shows how to do it:

![](https://cdn-images-1.medium.com/max/800/1*WBz68hADp1nJWARBGrEuLQ.png)

The next figure shows the original *mandrill* input image and the output
image obtained after applying the *wave* transform.

![](https://cdn-images-1.medium.com/max/800/0*98RpCaEuIgTH-JBd)

![](https://cdn-images-1.medium.com/max/800/0*5Bu9YdJjrCDjx1Qd)

### 2. Swirl Transform {#1541 .graf .graf--h3 .graf-after--figure name="1541"}

1.  Use *scikit-image’s warp()* function to implement the *swirl*
    transform.
2.  Note that swirl transform can be expressed with the following
    equations

![](https://cdn-images-1.medium.com/max/800/1*8Xn0NS4kCIIUo-XF-XhhWQ.png)

We shall use the *mandrill* image to implement the wave transform. The
next python code fragment shows how to do it:

![](https://cdn-images-1.medium.com/max/800/1*WBPpGY4Xo02j9G1RA1z6ug.png)

The next figure shows the original *mandrill* input image and the output
image obtained after applying the *swirl*transform.

![](https://cdn-images-1.medium.com/max/800/0*Q3Oej5WB0AWWfhlr)

![](https://cdn-images-1.medium.com/max/800/0*GFxMGZa1KMWkWFEo)

Compare this with the output of the *scikit-image swirl()* function.

### 3. Very simple Face morphing with α-blending {#cc2e .graf .graf--h3 .graf-after--p name="cc2e"}

1.  Start from one face image (e.g., let image1 be the face of Messi)
    and end into another image (let image2 be the face of Ronaldo)
    iteratively, creating some intermediate images in between.
2.  At each iteration create an image by using a *linear combination* of
    the two image *numpy ndarrays* given by

![](https://cdn-images-1.medium.com/max/800/1*8aNmzqCsL4pcugJfQilguw.png)

​3. Iteratively increase *α* from 0 to 1.

The following code block shows how to implement it using matplotlib’s
image and pylab modules.

![](https://cdn-images-1.medium.com/max/800/1*NALbmhMBjY-F5JtQVkXT4A.png)

The next animation shows the simple face morphing:

![](https://cdn-images-1.medium.com/max/800/0*l-ZBWk2P50MsQWNi)

There are more sophisticated techniques to improve the quality of
morphing, but this is the simplest one.

### 4. Creating Instagram-like Gotham Filter {#9c92 .graf .graf--h3 .graf-after--p name="9c92"}

#### The Gotham filter {#4206 .graf .graf--h4 .graf-after--h3 name="4206"}

The Gotham filter is computed as follows (the steps taken from
[here](https://www.practicepython.org/blog/2016/12/20/instagram-filters-python.html)),
applying the following operations on an image, the corresponding python
code, input and output images are shown along with the operations (with
the following input image):

![](https://cdn-images-1.medium.com/max/800/0*Nh0tmGAyjTl6KppB)

1.  A mid-tone red contrast boost

![](https://cdn-images-1.medium.com/max/800/1*E01nsVXo2rIihVYIB4HlVA.png)

![](https://cdn-images-1.medium.com/max/800/0*Qot-ajQVAIBNA_EE)

​2. Make the blacks a little bluer

![](https://cdn-images-1.medium.com/max/800/1*4jGSafOssrwhZvH5gi5vXQ.png)

![](https://cdn-images-1.medium.com/max/800/0*A9qhzRaFBASvJd_L)

​3. A small sharpening

![](https://cdn-images-1.medium.com/max/800/1*fKnSyvuLNDX5K-m7RG_MCw.png)

![](https://cdn-images-1.medium.com/max/800/0*yqsCR2DigLCinsqW)

​4. A boost in blue channel for lower mid-tones\
5. A decrease in blue channel for upper mid-tones

![](https://cdn-images-1.medium.com/max/800/1*EJebHv7qKRN1o6_gRyKZhA.png)

![](https://cdn-images-1.medium.com/max/800/0*EwSkXy4fwq6rD2jp)

The output image obtained after applying the *Gotham filter* is shown
below:

![](https://cdn-images-1.medium.com/max/800/0*R_Is8aOvuXSONb6n)

### 5. Down-sampling with anti-aliasing using Gaussian Filter {#e945 .graf .graf--h3 .graf-after--figure name="e945"}

1.  Start with a large gray-scale image and reduce the image size 16
    times, by reducing both height and width by 4 times.
2.  Select every 4th pixel in the x and the y direction from the
    original image to compute the values of the pixels in the smaller
    image.
3.  Before down-sampling apply a Gaussian filter (to smooth the image)
    for anti-aliasing.
4.  Compare the quality of the output image obtained by down-sampling
    without a Gaussian filter (with aliasing).

The next code block performs the above steps. Since the Gaussian blur is
a low-pass filter, it removes the high frequencies from the original
input image, hence it’s possible to achieve sampling rate above the
**Nyquist** rate (by **sampling theorem**) to avoid **aliasing**.

![](https://cdn-images-1.medium.com/max/800/1*jMUegTvDIiufbNBQsTXcbg.png)

**Original Image**

![](https://cdn-images-1.medium.com/max/800/0*9F2_5lTtwEPIEU7g)

**Image blurred with Gaussian Filter LPF**

![](https://cdn-images-1.medium.com/max/800/0*1_OrTwLujvNe4fHT)

**Down-sampled Image from the original image (with aliasing)**

![](https://cdn-images-1.medium.com/max/800/0*f0RdK6h1DjPI8ZD-)

**Down-sampled Image from the blurred image (with anti-aliasing)**

![](https://cdn-images-1.medium.com/max/800/0*XGUMKZnjx-lNEzHC)

### 6. Some Applications of DFT {#97a5 .graf .graf--h3 .graf-after--figure name="97a5"}

### (a) Fourier Transform of a Gaussian Kernel is another Gaussian Kernel {#0a2f .graf .graf--h3 .graf-after--h3 name="0a2f"}

Also, the spread in the frequency domain inversely proportional to the
spread in the spatial domain (known as **Heisenberg’s inequality**).
Here is the proof:

![](https://cdn-images-1.medium.com/max/800/0*4I3wQSnmZrlGlSoH)

The following animation shows an example visualizing the Gaussian
contours in spatial and corresponding frequency domains:

![](https://cdn-images-1.medium.com/max/800/0*b2S50M-b6EYEuMF9)

### (b) Using DFT to up-sample an image {#a8e8 .graf .graf--h3 .graf-after--figure name="a8e8"}

1.  Let’s use the *lena gray-scale* image.
2.  First double the size of the by padding zero rows/columns at every
    alternate positions.
3.  Use *FFT* followed by an *LPF.*
4.  Finally use *IFFT* to get the output image.

The following code block shows the python code for implementing the
steps listed above:

![](https://cdn-images-1.medium.com/max/800/1*l-N1mG7ZSmViE9Mu9VJfBw.png)

The next figure shows the output. As can be seen from the next figure,
the LPF removed the high frequency components from the Fourier spectrum
of the padded image and with a subsequent inverse Fourier transform we
get a decent enlarged image.

![](https://cdn-images-1.medium.com/max/800/0*UhruTd8ac2qhL5pt)

### (c) Frequency Domain Gaussian Filter {#8218 .graf .graf--h3 .graf-after--figure name="8218"}

1.  Use an input image and use DFT to create the frequency 2D-array.
2.  Create a small Gaussian 2D Kernel (to be used as an LPF) in the
    spatial domain and pad it to enlarge it to the image dimensions.
3.  Use DFT to obtain the Gaussian Kernel in the frequency domain.
4.  Use the Convolution theorem to convolve the LPF with the input image
    in the frequency domain.
5.  Use IDFT to obtain the output image.
6.  Plot the frequency spectrum of the image, the gaussian kernel and
    the image obtained after convolution in the frequency domain, in 3D.

The following code block shows the python code:

![](https://cdn-images-1.medium.com/max/800/1*gUP49sysdlOMzuk2hM1b4A.png)

#### The original color temple image (time / spatial domain) {#1071 .graf .graf--h4 .graf-after--figure name="1071"}

![](https://cdn-images-1.medium.com/max/800/0*PJdN8xz8RZjGuGXk)

**The temple image (frequency domain)**

![](https://cdn-images-1.medium.com/max/800/0*j5ng83bJE8LpbSNx)

#### The Gaussian Kernel LPF in 2D (frequency domain) {#0a55 .graf .graf--h4 .graf-after--figure name="0a55"}

![](https://cdn-images-1.medium.com/max/800/0*yQ_K1PT7qmX_RmgN)

#### The Gaussian Kernel LPF (frequency domain) {#2353 .graf .graf--h4 .graf-after--figure name="2353"}

![](https://cdn-images-1.medium.com/max/800/0*5cllJhGnjO5CxOPM)

#### The smoothed temple image with the LPF (frequency domain) {#e0c1 .graf .graf--h4 .graf-after--figure name="e0c1"}

![](https://cdn-images-1.medium.com/max/800/0*-SpXBwzaSN6Q1iYx)

If we set the standard deviation of the LPF Gaussian kernel to be 10 we
get the following output as shown in the next figures. As can be seen,
the frequency response value drops much quicker from the center.

#### The smoothed temple image with the LPF with higher s.d. (frequency domain) {#cb61 .graf .graf--h4 .graf-after--p name="cb61"}

![](https://cdn-images-1.medium.com/max/800/0*EDUSBGY4yC3Zk1F0)

**The output image after convolution (spatial / time domain)**

![](https://cdn-images-1.medium.com/max/800/0*9MxnVQ0JRdewEKum)

#### (d) **Using the inverse filter to restore a motion-blurred image** {#d9c3 .graf .graf--h4 .graf-after--figure name="d9c3"}

1.  First create a motion blur kernel of a given shape.
2.  Convolve the kernel with an input image in the frequency domain.
3.  Get the motion-blurred image in the spatial domain with IDFT.
4.  Compute the inverse filter kernel and convolve with the blurred
    image in the frequency domain.
5.  Get the convolved image back in the spatial domain.
6.  Plot all the images and kernels in the frequency domain.

The following code block shows the python code:

![](https://cdn-images-1.medium.com/max/800/1*awkWbflVk1abF08ePa3khQ.png)

![](https://cdn-images-1.medium.com/max/800/0*lLTa-VNHzpqr5J3U)

**Frequency response of the input image**

![](https://cdn-images-1.medium.com/max/800/0*W4evFsC90RI4law7)

**(log) Frequency response of the motion blur kernel (LPF)**

![](https://cdn-images-1.medium.com/max/800/0*muu6sd4UNJspPDi4)

**Input image convolved with the motion blur kernel (frequency domain)**

![](https://cdn-images-1.medium.com/max/800/0*IWn732pSSDjAeN5e)

**(log) Frequency response of the inverse frequency filter kernel
(HPF)**

![](https://cdn-images-1.medium.com/max/800/0*a5eLJF0ES-JejxtR)

**Motion-blurred image convolved with the inverse frequency filter
kernel (frequency domain)**

![](https://cdn-images-1.medium.com/max/800/0*Faf92q5NxMIciFsl)

#### **(e) Impact of noise on the inverse filter** {#cc5c .graf .graf--h4 .graf-after--figure name="cc5c"}

1.  Add some random noise to the Lena image.
2.  Blur the image with a Gaussian kernel.
3.  Restore the image using inverse filter.

**With the original image**

Let’s first blur and apply the inverse filter on the noiseless blurred
image. The following figures show the outputs:

![](https://cdn-images-1.medium.com/max/800/0*waAHVuiqECDcoB4Q)

**(log) Frequency response of the input image**

![](https://cdn-images-1.medium.com/max/800/0*axFkPPa0SwxTXNtg)

**(log) Frequency response of the Gaussian blur kernel (LPF)**

![](https://cdn-images-1.medium.com/max/800/0*2zcQXPyhG00LLwcW)

**(log) Frequency response of the blurred image**

![](https://cdn-images-1.medium.com/max/800/0*9Iua9pFjt2bAPx1Q)

**(log) Frequency response of the inverse kernel (HPF)**

![](https://cdn-images-1.medium.com/max/800/0*KhFTyhaSBBqIBbY6)

**Frequency response of the output image**

![](https://cdn-images-1.medium.com/max/800/0*8ExHbmrHo9Yw9TwC)

**Adding noise to the original image**

The following python code can be used to add Gaussian noise to an image:

``` {#89c3 .graf .graf--pre .graf-after--p name="89c3"}
from skimage.util import random_noiseim = random_noise(im, var=0.1)
```

The next figures show the noisy Lena image, the blurred image with a
Gaussian Kernel and the restored image with the inverse filter. As can
be seen, being a high-pass filter, the inverse filter enhances the
noise, typically corresponding to high frequencies.

![](https://cdn-images-1.medium.com/max/800/0*ui4u3RHCmIaVvu7B)

#### (f) Use a notch filter to remove periodic noise from the following half-toned car image. {#ac4e .graf .graf--h4 .graf-after--figure name="ac4e"}

![](https://cdn-images-1.medium.com/max/800/0*t0zDD2cdexRvH0Yc)

1.  Use DFT to obtain the frequency spectrum of the image.
2.  Block the high frequency components that are most likely responsible
    fro noise.
3.  Use IDFT to come back to the spatial domain.

![](https://cdn-images-1.medium.com/max/800/1*8ZxzeHoTRQQDBfFqILJFdg.png)

**Frequency response of the input image**

![](https://cdn-images-1.medium.com/max/800/0*OWrPICgVTJJJgfrx)

**Frequency response of the input image with blocked frequencies with
notch**

![](https://cdn-images-1.medium.com/max/800/0*NCFW02eKfmJGPTl9)

**Output image**

![](https://cdn-images-1.medium.com/max/800/0*mIQFherl0deH5nVK)

**With a low-pass-filter (LPF):**

**Frequency response of the input image with blocked frequencies with
LPF**

![](https://cdn-images-1.medium.com/max/800/0*JUPWi7fdPthrrL6q)

**Output image**

![](https://cdn-images-1.medium.com/max/800/0*Lx5VOq_vDsT6mg_s)

### 7. Histogram Matching with color images {#6d57 .graf .graf--h3 .graf-after--figure name="6d57"}

As described [here](http://paulbourke.net/miscellaneous/equalisation/),
here is the algorithm:

1.  The cumulative histogram is computed for each image dataset, see the
    figure below.
2.  For any particular value (x\_i) in the input image data to be
    adjusted has a cumulative histogram value given by G(x\_i).
3.  This in turn is the cumulative distribution value in the reference
    (template) image dataset, namely H(x\_j). The input data value x\_i
    is replaced by x\_j.

![](https://cdn-images-1.medium.com/max/800/1*VNzpyMaBuW4lWf6k8zddyg.png)

**Input image**

![](https://cdn-images-1.medium.com/max/800/0*NxQjwABCnv9mooVZ)

**Template Image**

![](https://cdn-images-1.medium.com/max/800/0*86zLWJ41F-4TZNdi)

**Output Image**

![](https://cdn-images-1.medium.com/max/800/0*dennAZetYd36yZ5l)

The following figure shows how the histogram of the input image is
matched with the histogram of the template image.

![](https://cdn-images-1.medium.com/max/800/0*2tkMXVb-vOdfiWDk)

Another example:

**Input image**

![](https://cdn-images-1.medium.com/max/800/0*HqTtz9dMyaB_xLLx)

**Template Image**

![](https://cdn-images-1.medium.com/max/800/0*UDRCUXWE4Z1GA_6A)

**Output Image**

![](https://cdn-images-1.medium.com/max/800/0*ZbPY2KfwXwBcmLQu)

![](https://cdn-images-1.medium.com/max/800/0*Tzp71BUysCDILkTI)

### 8. Mathematical Morphology {#5362 .graf .graf--h3 .graf-after--figure name="5362"}

### (a) Automatically cropping an image {#f817 .graf .graf--h3 .graf-after--h3 name="f817"}

1.  Let’s use the following image. The image has unnecessary white
    background outside the molecule of the organic compound.

![](https://cdn-images-1.medium.com/max/800/0*lqGJGnh8oUA_oyjb)

1.  First convert the image to a binary image and compute the convex
    hull of the molecule object.
2.  Use the convex hull image to find the bounding box for cropping.
3.  Crop the original image with the bounding box.

The next python code shows how to implement the above steps:

![](https://cdn-images-1.medium.com/max/800/1*twqDnwu_3g1lMY-i75VRnQ.png)

![](https://cdn-images-1.medium.com/max/800/0*5bytfJ0WztDTTyW_)

![](https://cdn-images-1.medium.com/max/800/0*KrdOnCenjAhzKuP8)

![](https://cdn-images-1.medium.com/max/800/0*QKSN8TcI1dy9ZZg2)

This can also be found
[here](https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil/51703287#51703287).

### (b) Opening and Closing are Dual operations in mathematical morphology {#79e4 .graf .graf--h3 .graf-after--p name="79e4"}

1.  Start with a binary image and apply opening operation with some
    structuring element (e.g., a disk) on it to obtain an output image.
2.  Invert the image (to change the foreground to background and vice
    versa) and apply closing operation on it with the same structuring
    element to obtain another output image.
3.  Invert the second output image obtained and observe that it’s same
    as the first output image.
4.  Thus applying opening operation to the foreground of a binary image
    is equivalent to applying closing operation to the background of the
    same image with the same structuring element.

The next python code shows the implementation of the above steps.

![](https://cdn-images-1.medium.com/max/800/1*xfZoAUOjCuKkMsyMFVNrNA.png)

As can be seen the output images obtained are exactly same.

![](https://cdn-images-1.medium.com/max/800/0*yZ4jUSsVSmnyvoXu)

### 9. Floyd-Steinberg Dithering (to convert a grayscale to a binary image) {#7a45 .graf .graf--h3 .graf-after--figure name="7a45"}

The next figure shows the algorithm for error diffusion dithering.

![](https://cdn-images-1.medium.com/max/800/0*YJYAa2Nwj0DM-bKI)

![](https://cdn-images-1.medium.com/max/800/1*Fih4cX9nkn_njQLPhL43hg.png)

**The input image (gray-scale)**

![](https://cdn-images-1.medium.com/max/800/0*kiSB-jHcol655NhO)

**The output Image (binary)**

![](https://cdn-images-1.medium.com/max/800/0*vds3nxVu5Ia4qumu)

The next animation shows how an another input grayscale image gets
converted to output binary image using the error diffusion dithering.

![](https://cdn-images-1.medium.com/max/800/0*iVrxgc7FbQR4rPbs)

### 10. Sharpen a color image {#7c92 .graf .graf--h3 .graf-after--figure name="7c92"}

1.  First blur the image with an LPF (e.g., Gaussian Filter).
2.  Compute the detail image as the difference between the original and
    the blurred image.
3.  Now the sharpened image can be computed as a linear combination of
    the original image and the detail image. The next figure illustrates
    the concept.

![](https://cdn-images-1.medium.com/max/800/0*Orf4DKnd4LA1l4Gm)

The next python code shows how this can be implemented in python:

![](https://cdn-images-1.medium.com/max/800/1*a7vcCD09dMSAIkvVJe6eLg.png)

The next figure shows the output of the above code block. As cane be
seen, the output gets more sharpened as the value of alpha gets
increased.

![](https://cdn-images-1.medium.com/max/800/0*Q4VcbtcjT-uGaXy0)

The next animation shows how the image gets more and more sharpened with
increasing alpha.

![](https://cdn-images-1.medium.com/max/800/0*k2E6TbjrKSf9-ji0)

### 11. Edge Detection with LOG and Zero-Crossing Algorithm by Marr and Hildreth {#7ea3 .graf .graf--h3 .graf-after--figure name="7ea3"}

The following figure shows LOG filter and its DOG approximation.

![](https://cdn-images-1.medium.com/max/800/0*NdCJJ9nJru-FwcgE)

![](https://cdn-images-1.medium.com/max/800/0*auoLN3nPv160haXG)

In order to detect edges as a binary image, finding the zero-crossings
in the LoG-convolved image was proposed by Marr and Hildreth.
Identification of the edge pixels can be done by viewing the sign of the
LoG-smoothed image by defining it as a binary image, the algorithm is as
follows:

**Algorithm to compute the zero-crossing**

1.  First convert the LOG-convolved image to a binary image, by
    replacing the pixel values by 1 for positive values and 0 for
    negative values.
2.  In order to compute the zero crossing pixels, we need to simply look
    at the boundaries of the non-zero regions in this binary image.
3.  Boundaries can be found by finding any non-zero pixel that has an
    immediate neighbor which is is zero.
4.  Hence, for each pixel, if it is non-zero, consider its 8 neighbors,
    if any of the neighboring pixels is zero, the pixel can be
    identified as an edge.

The next python code and the output images / animations generated show
how to detect the edges from the zebra image with LOG + zero-crossings:

![](https://cdn-images-1.medium.com/max/800/1*BzI2gEigGj27Fig8OWbb7A.png)

**Original Input Image**

![](https://cdn-images-1.medium.com/max/800/0*gM10Wbq9vHDG0GNJ)

**Output with edges detected with LOG + zero-crossing at different sigma
scales**

![](https://cdn-images-1.medium.com/max/800/0*3QZFsQ6lQdb-qe7u)

![](https://cdn-images-1.medium.com/max/800/0*k3NG7ft5hV5euSJj)

With another input image

![](https://cdn-images-1.medium.com/max/800/0*zqsyW0xNYqSY5LvB)

**Output with edges detected with LOG + zero-crossing at different sigma
scales**

![](https://cdn-images-1.medium.com/max/800/0*O7ydgCdvi_c6Q-4-)

![](https://cdn-images-1.medium.com/max/800/0*PmDxfGOrqUDDMq3L)

### 12. Constructing the Gaussian Pyramid with scikit-image transform module’s reduce function and Laplacian Pyramid from the Gaussian Pyramid and the expand function {#c24d .graf .graf--h3 .graf-after--figure name="c24d"}

The *Gaussian Pyramid* can be computed with the following steps:

1.  Start with the original image.
2.  Iteratively compute the image at each level of the pyramid first by
    smoothing the image (with gaussian filter) and then downsampling
    it .
3.  Stop at a level where the image size becomes sufficiently small
    (e.g., 1×1).

The *Laplacian Pyramid* can be computed with the following steps:

1.  Start with the Gaussian Pyramid and with the smallest image.
2.  Iteratively compute the difference image in between the image at the
    current level and the image obtained by first upsampling and then
    smoothing the image (with gaussian filter) from the previous level
    of the Gaussian Pyramid.
3.  Stop at a level where the image size becomes equal to the original
    image size.

The next python code shows how to create a Gaussian Pyramid from an
image.

![](https://cdn-images-1.medium.com/max/800/1*whZUOxekgxsoG6oZlbn_SA.png)

### Some images from the Gaussian Pyramid {#2a06 .graf .graf--h3 .graf-after--figure name="2a06"}

![](https://cdn-images-1.medium.com/max/800/0*G0PqBUawCrxi9wPm)

![](https://cdn-images-1.medium.com/max/800/0*Pnah9zmwiKU9v04D)

![](https://cdn-images-1.medium.com/max/800/0*I1c0QxJ0b6ihGGkZ)

![](https://cdn-images-1.medium.com/max/800/0*iX27h9hKwL530vIR)

### Some images from the Laplacian Pyramid {#124e .graf .graf--h3 .graf-after--figure name="124e"}

![](https://cdn-images-1.medium.com/max/800/0*AqP0SX1Lp26d8Od6)

![](https://cdn-images-1.medium.com/max/800/0*i037N_DNh650FUKC)

![](https://cdn-images-1.medium.com/max/800/0*Bx1oBR_f_pbL6WK-)

### 13. Blending images with Gaussian and Laplacian pyramids {#644b .graf .graf--h3 .graf-after--figure name="644b"}

Here is the algorithm:

![](https://cdn-images-1.medium.com/max/800/0*D79Hfu6k_-02AOs_)

Blending the following input images A, B with mask image M

**Input Image A (Goddess Durga)**

![](https://cdn-images-1.medium.com/max/800/0*c2nK5AI628y10XIH)

**Input Image B (Lord Shiva)**

![](https://cdn-images-1.medium.com/max/800/0*zmngEx_iWi3RWi8p)

> ***Mask Image M***

![](https://cdn-images-1.medium.com/max/800/0*eLwpev883Pl7yeJk)

with the following python code creates the output image I shown below

![](https://cdn-images-1.medium.com/max/800/1*M35zf9QmaTokC5aqP0g1SA.png)

**Output Image I (Ardhanarishwara)**

![](https://cdn-images-1.medium.com/max/800/0*XQw8hvPZzagWR4FS)

The following animation shows how the output image is formed:

![](https://cdn-images-1.medium.com/max/800/0*tp0iOZ8cqd7gYwGI)

Another blending (horror!) example (from prof. dmartin)

![](https://cdn-images-1.medium.com/max/800/0*PIsLt8vy5mQzVd6_)

### 14. Removing Gaussian Noise from images by computing mean and median images with scikit-image {#4c4f .graf .graf--h3 .graf-after--figure name="4c4f"}

1.  Start with an input image.
2.  Create n (e.g, n=100) noisy images by adding i.i.d. Gaussian noise
    (with zero mean) to the original image, with *scikit-image*.
3.  Compute the mean (median) of the noisy images.
4.  Compare PSNR with the original image.
5.  Vary n and compare the results.

``` {#c107 .graf .graf--pre .graf-after--li name="c107"}
from skimage import img_as_floatfrom skimage.util import random_noisefrom skimage.measure import compare_psnrfrom skimage.io import imreadimport matplotlib.pylab as pltimport numpy as np
```

``` {#6e4c .graf .graf--pre .graf-after--pre name="6e4c"}
im = img_as_float(imread('../new images/parrot.jpg')) # original imagenp.random.seed(0)# generate n noisy images from the original image by adding Gaussian noisen = 25images = np.zeros((n, im.shape[0], im.shape[1], im.shape[2]))sigma = 0.2for i in range(n):    images[i,...] = random_noise(im, var=sigma**2)
```

``` {#336a .graf .graf--pre .graf-after--pre name="336a"}
im_mean = images.mean(axis=0)im_median = np.median(images, axis=0)plt.figure(figsize=(20,16))plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)plt.subplot(221), plt.imshow(im), plt.axis('off'), plt.title('Original image', size=20)plt.subplot(222), plt.imshow(images[0]), plt.axis('off'), plt.title('Noisy PSNR: ' + str(compare_psnr(im, images[0])), size=20)plt.subplot(223), plt.imshow(im_mean), plt.axis('off'), plt.title('Mean PSNR: ' + str(compare_psnr(im, im_mean)), size=20)plt.subplot(224), plt.imshow(im_median), plt.axis('off'), plt.title('Median PSNR: ' + str(compare_psnr(im, im_median)), size=20)plt.show()
```

The next figure shows the original image, a noisy image generated from
it by adding Gaussian noise (with 0 mean) to it and the images obtained
by taking mean / median over all the n noisy images generated. As can be
seen, the Gaussian noise in the images gets cancelled out by taking mean
/ median.

**with n = 25**

![](https://cdn-images-1.medium.com/max/800/0*lN0dP8-86RYRB9N3)

**with n=100**

![](https://cdn-images-1.medium.com/max/800/0*XjOwZMQ7FlxPA8bZ)

``` {#477e .graf .graf--pre .graf-after--figure name="477e"}
plt.hist(images[:,100,100,0], color='red', alpha=0.2, label='red')plt.hist(images[:,100,100,1], color='green', alpha=0.2, label='green')plt.hist(images[:,100,100,2], color='blue', alpha=0.2, label='blue')plt.legend()plt.grid()plt.show()
```

The next figure shows how a pixel value (that can be considered a random
variable) for a particular location in different noisy images follows
approximately a Gaussian distribution.

**Distribution of a pixel value at location (100,100) in the noisy
images**

![](https://cdn-images-1.medium.com/max/800/0*QIGdla4p7CvMlj-t)

``` {#91ca .graf .graf--pre .graf-after--figure name="91ca"}
ns = [25, 50, 100, 200]# mean_psnrs contain the PSNR values for different nplt.plot(ns, mean_psnrs, '.--', label='PSNR (mean)')plt.plot(ns, median_psnrs, '.--', label='PSNR (median)')plt.legend()plt.xlabel('n'),  plt.ylabel('PSNR')plt.show()
```

The following figure shows that the *PSNR* improves with large n (since
by *SLLN / WLLN*, the*sample mean* converges to *population mean* 0 of
the Gaussian noise). Also, for median the improvement in the image
quality is higher for larger values of *n*.

![](https://cdn-images-1.medium.com/max/800/0*w2umCi5y1iMVulpi)

### 15. Tracking Pedestrians with HOG-SVM with OpenCV / scikit-image {#2609 .graf .graf--h3 .graf-after--figure name="2609"}

1.  Start with a video with pedestrians.
2.  Capture the video / extract frames from the video.

For each frame

1.  Create HOG scale pyramid of the frame image.
2.  At each scale, use a sliding window to extract the corresponding
    block from the frame, compute the HOG descriptor features.
3.  Use *cv2*‘s HOGDescriptor\_getDefaultPeopleDetector() — a
    pre-trained SVM classifier on the HOG descriptor to classify whether
    the corresponding block contains a pedestrian or not.
4.  Run non-max-suppression to get rid of multiple detection of the same
    person.
5.  Use *cv2*‘s *detectMultiScale()* function to implement steps 3–4.

The code is adapted from the code
[here](https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)and
[here](https://www.learnopencv.com/histogram-of-oriented-gradients/).

``` {#b9d5 .graf .graf--pre .graf-after--p name="b9d5"}
# HOG descriptor using default people (pedestrian) detectorhog = cv2.HOGDescriptor()hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

``` {#c5d0 .graf .graf--pre .graf-after--pre name="c5d0"}
# run detection, using a spatial stride of 4 pixels,# a scale stride of 1.02, and zero grouping of rectangles# (to demonstrate that HOG will detect at potentially# multiple places in the scale pyramid)(foundBoundingBoxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.02, finalThreshold=0, useMeanshiftGrouping=False)
```

``` {#6b4f .graf .graf--pre .graf-after--pre name="6b4f"}
# convert bounding boxes from format (x1, y1, w, h) to (x1, y1, x2, y2)rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in foundBoundingBoxes])
```

``` {#a8b1 .graf .graf--pre .graf-after--pre name="a8b1"}
# run non-max suppression on the boxes based on an overlay of 65%nmsBoundingBoxes = non_max_suppression(rects, probs=None, overlapThresh=0.65)
```

*cv2* functions are used to extract HOG descriptor features and
pedestrian detection with SVM, whereas *scikit-image* functions are used
to visualize the HOG features. The animations below display the original
video, what HOG sees and the detected pedestrians after non-max
suppression. Notice there are a few false positive detection.

**Original Video**

![](https://cdn-images-1.medium.com/max/800/1*4lYia3Bl37oL425DE1e6mA.gif)

**HOG-descriptor features video (what HOG sees)**

![](https://cdn-images-1.medium.com/max/800/1*1oXZZBXOXaAHSqG-ghUWow.gif)

**Original Video with detected Pedestrians**

![](https://cdn-images-1.medium.com/max/800/1*vSX-DwxGicikA5_mY8krSQ.gif)

#### 16. Object Detection with YOLO v2 DarkNet / Keras / OpenCV (Deep Learning model) {#6a0d .graf .graf--h4 .graf-after--figure name="6a0d"}

![](https://cdn-images-1.medium.com/max/800/0*zfzSDxbKsxh_UCPq)

### 17. Semantic Segmentation with ENet / DeepLab (Deep Learning model) {#535d .graf .graf--h3 .graf-after--figure name="535d"}

**Input video and the segmented Output video**

![](https://cdn-images-1.medium.com/max/800/1*Tsgw_bS1iyv4IwE57b7p5Q.gif)

**Input video and the segmented Output video**

![](https://cdn-images-1.medium.com/max/800/0*PERteiDwmc20IIVW)

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [July 6,
2020](https://medium.com/p/c7f8a14fc16f).

[Canonical
link](https://medium.com/@sandipan-dey/solving-some-image-processing-and-computer-vision-problems-with-python-libraries-c7f8a14fc16f)

Exported from [Medium](https://medium.com) on January 8, 2021.


#Some Image Processing and Computational Photography: Convolution and Filtering 

The following problems appeared as an assignment in the **coursera course Computational Photography (by Georgia Tech university)**. The following descriptions of the problems are taken directly from the assignments' descriptions.


    #ipython nbconvert pcaiso.ipynb
    %matplotlib inline
    
    from IPython.display import HTML
    
    HTML('''<script>
    code_show=true; 
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>



## Introduction

In this article, we shall be playing around with images, *filters*, and *convolution*. We will begin by building a function that performs convolution. We will then experiment with constructing and applying a variety of filters. Finally, we will see one example of how these ideas can be used to create interesting effects in our photos, by finding and coloring edges in our images.

## Filtering

In this section, let's apply a few filters to some images. Filtering basically means replacing each pixel of an image by the linear combination of its neighbors. We need to understand the following concepts in this context:

1. Kernel (mask) for a filter: defines which neighbors to be considered and what weights are to be given.
2. Cross-Correleation vs. Convolution: determines how the kernel is going to be applied on the neighboring pixels to compute the linear combination.

### Convolution

The following figure describes the basic concepts of *cross-correlation* and *convolution*. Basically convolution flips the kernel before applying it to an image.


    from IPython.display import Image
    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\im1.png')




![png](output_6_0.png)



## Cross-Correlation vs. Convolution

The following figures show a custom kernel applied on an *impulse response* image changes the image both with cross-correlation and convolution. 

The **impulse response image** is shown below:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\impulse_response.png')




![png](output_8_0.png)



The below figure shows a **3x3 custom kernel** to be applied on the above **impulse response** image.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\kernel_custom.png', width=100, height=100)




![png](output_10_0.png)



The following figure shows the output images after applying **cross-correlation** and **convolution** with the above **kernel** on the above **impulse response** image. As can be seen, **convolution** produces the desired output.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\cc.png')




![png](output_12_0.png)



The following figures show the application of the same *kernel* on some *grayscale images* and the output images after *convolution*.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\all_conv_kernel_custom.png')




![png](output_14_0.png)



Now let's apply the following **5x5 flat box2 kernel** shown below. 

### The 5x5 Box2 Kernel Matrix


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\kernel_box2.png', width=250, height=250)




![png](output_16_0.png)



The following figures show the application of the *box kernel* above on some *grayscale images* and the output images after *convolution*. on some images and notice how the output images are **blurred**. 


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\all_conv_kernel_box2.png')




![png](output_18_0.png)



The following figures show the application of the **box kernels** of different **size** on the *grayscale image lena* and the output images after *convolution*. As expected, the **blur** effect increases as the size of the *box kernel* increases. 


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\lena_conv_kernel_box.png')




![png](output_20_0.png)



## Gaussian Filter
The following figure shows a **11x11 Gaussian Kernel** generated by taking *outer product* of the *densities* of two **1D i.i.d. Gaussians** with **mean 0** and **s.d. 3**.  


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\kernel_gaussian_5_3.png')




![png](output_22_0.png)



Here is how the **impulse response image** (enlarged) looks like after the application of the above **Gaussian Filer**.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\Impulse_gaussian_3.png')




![png](output_24_0.png)



The next figure shows the effect of **Gaussian filtering / smoothing (blur)** on some images.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\all_conv_kernel_gaussian3.png')




![png](output_26_0.png)



The following figure shows the **11x11 Gaussian Kernels** generated with **1D i.i.d. Gaussians** with different **bandwidths** 
**s.d.** values.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\gaussian_kernels.png')




![png](output_28_0.png)



The following figures show the application of the **Gaussian kernels** of different **bandwidth** on the *following grayscale image* and the output images after *filtering*. As expected, the **blur** effect increases as the *bandwidth* of the *Gaussian kernel* increases. 


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\a_gaussian_kernel.png')




![png](output_30_0.png)



## Sharpen Filter
The following figure shows a **11x11 Sharpen Kernel** generated by subtracting a *gaussian kernel* (with *s.d. 3*) from a scaled *impulse response kernel* with *2* at *center*.  


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\sharpen_kernel.png')




![png](output_32_0.png)



The next figure shows the effect of **Sharpen flitering** on some images.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\all_conv_kernel_sharpen3.png')




![png](output_34_0.png)



The following figures show the application of the **Sharpen kernels** of different **bandwidth** on the *following grayscale image* and the output images after *filtering*. As expected, the **sharpen** effect increases as the *bandwidth* of the *Gaussian kernel* increases. 


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\mri_sharpen_kernel.png')




![png](output_36_0.png)



## Median filter
One last thing we shall do to get a feel for is *nonlinear filtering*. So far, we have been doing everything by multiplying the input image pixels by various coefficients and summing the results together. A median filter works in a very different way, by simply choosing a single value from the surrounding patch in the image.

The next figure shows the effect of **Median flitering** on some images. As expected, with a **11x11** mask, some of the images are getting quite **blurred**.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\all_median.png')




![png](output_38_0.png)



## Drawing with edges

At a high level, we will be finding the intensity and orientation of the edges present in the image using convolutional filters, and then visualizing this information in an aesthetically pleasing way. 

### Image gradients and the sobel filter

The first thing it does is to find the **gradient** of the image. A *gradient* is a fancy way of saying “rate of change”. In order to find the edges in our image, we are going to look for places where pixels are rapidly changing in intensity. There are a variety of ways for doing this, and one of the most standard is through the use of **Sobel filters**, which have the following **kernels**:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\im2.png')




![png](output_40_0.png)



The following figure shows the basic concepts about *image gradients*.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\im6.png')




![png](output_42_0.png)



If we think about the x *sobel filter* being placed on a strong vertical edge:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\im3.png')




![png](output_44_0.png)



Considering the yellow location, the values on the right side of the kernel get mapped to brighter, and thus larger, values. The values on the left side of the kernel get mapped to darker pixels which are close to zero. The response in this position will be large and positive.

Compare this with the application of the kernel to a relatively flat area, like the blue location. The values on both sides are about equal, so we end up with a response that is close to zero. Thus, the **x-direction sobel filter** gives a strong response for **vertical edges**. Similarly, the **y-direction sobel filter** gives a strong response for **horizontal edges**.

The steps for **edge detection**:

1. Convert the image to grayscale.
2. *Blur* the image with a **gaussian kernel**. The purpose of this is to *remove noise* from the image, so that we find   responses only to significant edges, instead of small local changes that might be caused by our sensor or other factors.
3. Apply the two **sobel filters** to the image. 

### Edge orientations and magnitudes
Now we have the rate at which the image is changing in the x and y directions, but it makes more sense to talk about images in terms of *edges*, and their *orientations* and *intensities*. As we will see, we can use some trigonometry to transform between these two representations.

First, let’s see how our sobel filters respond to edges at different orientations from the following figures:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\im4.png')




![png](output_46_0.png)



The red arrow on the right shows the relative intensities of the response from the x and y **sobel filter**. We see that as we 
rotate the edge, the intensity of the response slowly shifts from the x to the y direction. We can also consider what would happen if the edge got less intense:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\im5.png')




![png](output_48_0.png)



### Mapping to color

We now have the magnitude and orientation of the edge present at every pixel. However, this is still not really in a form that we can examine visually. The angle varies from 0 to 180, and we don’t know the scale of the magnitude. What we have to do is to figure out a way to transform this information to some sort of color values to place in each pixel.

The edge orientations can be pooled into four bins edges that are roughly horizontal, edges that are roughly vertical, and edges at 45 degrees to the left and to the right. Each of these bins are assigned a color (vertical is yellow, etc). Then the magnitude is used to dictate the intensity of the edge. For example a roughly vertical edge of moderate intensity would be set to a medium yellow color, or the value (0, 100, 100).

The following figures show some of the *edges* extracted from some of the images previously used as inputs and color-mapped using *sobel filter*:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw1_solved\\edges.png')




![png](output_51_0.png)




#Some more Computational Photography: Creating Video Textures

The following problem appeared as an assignment in the **coursera course Computational Photography (by Georgia Tech, 2013)**. The following description of the problem is taken directly from the assignment's description.
   

## Introduction

In this article, we shall be applying our computational photography magics to video, with the purpose of creating **video textures**, or infinitely looping pieces of video. The input and output for the homework assignment is provided as a folder of images. The reasoning behind this, and suggestions for moving between different formats of video are given
in the video Appendix in section 4.

## Video Volume and SSD

Let's get more familiar with the *4d coordinate system* of a *video volume* (time x row x column x channel). 
To compute *SSD*, we need to find an image distance between every pair of frames in the video. SSD stands for sum of square distances, which as the name suggests, requires to take the pixelwise difference of the two frames, square it, and sum them all together.

## Diff2

This function takes in a difference matrix created by *ssd*, and updates it with dynamic information. The intuition behind this is as follows when considering the *transition cost* from *frame i* to *j*, we should not only look at the frames themselves, but also consider the preceding and following frames. So, we have:

![png](https://sandipanweb.files.wordpress.com/2017/05/im114.png)


Or, in other words, we are going to take a **weighting function**, and sum across the **ssd** outputs of the preceding and following frames in order to update the *transition costs* with *dynamic information*. We are going to use *binomial filter* for this purpose. The following figure shows the basics of a *binomial filter*:


![png](https://sandipanweb.files.wordpress.com/2017/05/im211.png)

## Find the biggest loop and synthesize loop

### find the biggest loop

Now that we have the costs of transitioning from one frame to another, we should find a suitable loop for our *video texture*. Simply taking the smallest transition distance here might not be desirable what if the resulting loop is only 1 frame long?
In order to state within the code that loop size matters, we will use a trick that is frequent in engineering disciplines.
We are going to find the transition which is optimal under a metric: **score(s, f) = α * (f − s) − diff2[f, s]**.

Note the two terms of the metric. The first is the difference between the final and starting frame of our loop. This term is large when the loop is large. The second term is the output of our *diff2* function, which tells us the *cost* of transitioning from finish to start. Subtracting this term turns it into a **'smoothness'** parameter. It is larger when the *transition* is less noticeable.

The last bit of wizardry is the *alpha* parameter. Because the size of the loop and the transition cost are likely to be in very different units, we introduce a new parameter to make them comparable. We can manipulate alpha to control the tradeoff between loop size and smoothness. Large alphas prefer large loop sizes, and small alphas prefer smoother transitions.
The find biggest loop function has to compute this score for every choice of *s* and *f*, and return the *s* and *f* that correspond to the largest score.

### synthesize loop

The finishing step is to take our video volume, and turn it back into a series of images, now cropping it to only contain the loop we found. Let's implement this function does just that. It is pretty much the inverse of the video volume function implemented earlier, except for this time we are starting with a full video volume, and returning a list of only the image frames between start and finish (inclusive).

The following figure explains how we are going to find the **video texture** from an input **candle video** with 90 frames.

![png](https://sandipanweb.files.wordpress.com/2017/05/im03.png)

## Input Video

Let's use the following *candle video* as the input video and extract **100 frames** from the video. As can be seen that there are some jumps from the end to the beginning of the video and our aim is to remove the jump and create a smooth **video texture** that can be played infinitely many times without any jump.

![gif](https://sandipanweb.files.wordpress.com/2017/05/anim_source.gif)

The following figure shows the *100x100 distance matrix* (**diff1**) computed in between any two video frames:

![png](https://sandipanweb.files.wordpress.com/2017/05/im32.png)

The following figure shows the *distance matrix* (**diff2**) computed after taking into account the *transition costs* (*weights*) and distances of the adjacent frames in a window when computing the distance between any two frames.


![png](https://sandipanweb.files.wordpress.com/2017/05/im44.png)

Finally, the following figure shows the *distance matrix* (**diff3**) computed after taking into account the *tradeoff* in between the *length* of the *video texture* and the *smoothness of transition* between the start and the end frame (with the controlling parameter *alpha=90*).


![png](https://sandipanweb.files.wordpress.com/2017/05/im53.png)



## Output Video Texture

The following video shows the **output video texture** produced, as can be seen the video now contains no jump and it's pretty smooth.

![gif](https://sandipanweb.files.wordpress.com/2017/05/anim_out.gif)


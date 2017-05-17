
#Some more Computational Photography: Creating Video Textures

The following problem appeared as an assignment in the **coursera course Computational Photography (by Georgia Tech, 2013)**. The following description of the problem is taken directly from the assignment's description.


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

In this article, we shall be applying our computational photography magics to video, with the purpose of creating **video textures**, or infinitely looping pieces of video. The input and output for the homework assignment is provided as a folder of images. The reasoning behind this, and suggestions for moving between different formats of video are given
in the video Appendix in section 4.

## Video Volume and SSD

Let's get more familiar with the *4d coordinate system* of a *video volume* (time x row x column x channel). 
To compute *SSD*, we need to find an image distance between every pair of frames in the video. SSD stands for sum of square distances, which as the name suggests, requires to take the pixelwise difference of the two frames, square it, and sum them all together.

## Diff2

This function takes in a difference matrix created by *ssd*, and updates it with dynamic information. The intuition behind this is as follows when considering the *transition cost* from *frame i* to *j*, we should not only look at the frames themselves, but also consider the preceding and following frames. So, we have:


    from IPython.display import Image
    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\im1.png', width=500, height=50)




![png](output_4_0.png)



Or, in other words, we are going to take a **weighting function**, and sum across the **ssd** outputs of the preceding and following frames in order to update the *transition costs* with *dynamic information*. We are going to use *binomial filter* for this purpose. The following figure shows the basics of a *binomial filter*:


    from IPython.display import Image
    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\im2.png')




![png](output_6_0.png)



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


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\im0.png')




![png](output_8_0.png)



## Input Video

Let's use the following *candle video* as the input video and extract **100 frames** from the video. As can be seen that there are some jumps from the end to the beginning of the video and our aim is to remove the jump and create a smooth **video texture** that can be played infinitely many times without any jump.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\source\\anim_source.gif')


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-8-9aebe42029ba> in <module>()
    ----> 1 Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\source\\anim_source.gif')
    

    C:\Users\Sandipan.Dey\Anaconda\envs\dato-env\lib\site-packages\IPython\core\display.pyc in __init__(self, data, url, filename, format, embed, width, height, retina, unconfined, metadata)
        731 
        732         if self.embed and self.format not in self._ACCEPTABLE_EMBEDDINGS:
    --> 733             raise ValueError("Cannot embed the '%s' image format" % (self.format))
        734         self.width = width
        735         self.height = height
    

    ValueError: Cannot embed the 'gif' image format


The following figure shows the *100x100 distance matrix* (**diff1**) computed in between any two video frames:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\im3.png')




![png](output_12_0.png)



The following figure shows the *distance matrix* (**diff2**) computed after taking into account the *transition costs* (*weights*) and distances of the adjacent frames in a window when computing the distance between any two frames.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\im4.png')




![png](output_14_0.png)



Finally, the following figure shows the *distance matrix* (**diff3**) computed after taking into account the *tradeoff* in between the *length* of the *video texture* and the *smoothness of transition* between the start and the end frame (with the controlling parameter *alpha=90*).


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\im5.png')




![png](output_16_0.png)



## Output Video Texture

The following video shows the **output video texture** produced, as can be seen the video now contains no jump and it's pretty smooth.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\out\\anim_out.gif')


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-13-6d51bf000eaf> in <module>()
    ----> 1 Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\Gatech - Computational Photography\\Assignments\\HW\\hw4_solved\\out\\anim_out.gif')
    

    C:\Users\Sandipan.Dey\Anaconda\envs\dato-env\lib\site-packages\IPython\core\display.pyc in __init__(self, data, url, filename, format, embed, width, height, retina, unconfined, metadata)
        731 
        732         if self.embed and self.format not in self._ACCEPTABLE_EMBEDDINGS:
    --> 733             raise ValueError("Cannot embed the '%s' image format" % (self.format))
        734         self.width = width
        735         self.height = height
    

    ValueError: Cannot embed the 'gif' image format



#Some more Image Processing: Ostu's Method, Hough Transform and Motion-based Segmentation with Python and OpenCV

Some of the following problems appeared in the lectures and the exercises in the **coursera course Image Processing (by NorthWestern University)**. Some of the following descriptions of the problems are taken from the exercise's description.


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



## 1 . Ostu's method for automatic thresholding to get binary images

We need to find a *thershold* to binarize an image, by separating the *background* from the *foreground*. Using *Ostu's method* we can automatically find the *global optimal threshold*, by maximizing the *between-class variance*. The following figure shows the outline for the technique.


    from IPython.display import Image
    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\im10.png', width=800)




![png](output_4_0.png)



The following figures show the **thresholds** computed and the output **binary** images obtained by using **Ostu's method** on a few *gray-scale* **low-contrast** images: 


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\o1.png')




![png](output_6_0.png)



The following figures show the output **binary** images obtained by using **Ostu's method** on a few *gray-scale* **high-contrast** images obtained using **histogram equalization**:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\o2.png')




![png](output_8_0.png)



The following figures show the distribution of the **between-class variance** for different images and the **htreshold** chosen by **Ostu's** method:


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\ostu.png')




![png](output_10_0.png)



## 2 . Hough Transform to find lines in images

The *Hough transform* can be used to find lines in an image by taking votes from each of the pixels for all possible orientations of lines, for different values of the parameters in a straight line's equation. The following figure describes the method.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\im14.png')




![png](output_12_0.png)



The following figures show a couple of binary images and the *votes* obtained by the image pixels in the *parametric space* of any straight line and how thresholding on the votes we can find the *most prominent line(s)* in the image (markd by *green*). The parameter values marked with *arrow(s)* represent the most prominent line(s) in the image. 


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\hog2.png')




![png](output_14_0.png)



The following results are inspired by the youtube video *How Hough Transform works* (by *Thales Sehn Korting*) which can be found here: https://www.youtube.com/watch?v=4zHbI-fFIlI. Again, by thresholding on the votes by the pixels for the differrent values of the straight line parameters, we can find the most prominent line(s) in the image (markd by green). The parameter values marked with arrow(s) represent the most prominent line(s) in the image.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\hog.png')




![png](output_16_0.png)



The next figure shows all the steps in how *Hough transform* can be used on any (*gray-level*) image to find lines (edges).


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\tiger_hough.png')




![png](output_18_0.png)




    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\umbc_hough.png')




![png](output_19_0.png)



## 3. Motion based Segmentation using Accumulative Difference Image
In this problem we basically want to separate the **moving objects** in *consectuive frames* of a video from the **non-moving objects**. The following figures show the problem statement that appeared in an assignment in the same course and also the theory to be used: 


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\im14_15.png', width=800)




![png](output_21_0.png)




    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\im15.png')




![png](output_22_0.png)



The following animation shows the motion of a moving rectangle. The next figure shows how the motion-based using ADI based techniques can be applied to separate out the moving rectangle from the static backgorund.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\motion.gif')


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-18-c0fff8c89473> in <module>()
    ----> 1 Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\motion.gif')
    

    C:\Users\Sandipan.Dey\Anaconda\envs\dato-env\lib\site-packages\IPython\core\display.pyc in __init__(self, data, url, filename, format, embed, width, height, retina, unconfined, metadata)
        731 
        732         if self.embed and self.format not in self._ACCEPTABLE_EMBEDDINGS:
    --> 733             raise ValueError("Cannot embed the '%s' image format" % (self.format))
        734         self.width = width
        735         self.height = height
    

    ValueError: Cannot embed the 'gif' image format


Agsin, the next animations show how the moving objects can be segemented from the non-moving ones from the consecutive frames of a video. First the frames from the video are *binarized* using **Ostu's** method and then **absolute ADI** is applied to separate out the moving objects.


    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\in.gif')


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-3b31079ad2e2> in <module>()
    ----> 1 Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\in.gif')
    

    C:\Users\Sandipan.Dey\Anaconda\envs\dato-env\lib\site-packages\IPython\core\display.pyc in __init__(self, data, url, filename, format, embed, width, height, retina, unconfined, metadata)
        731 
        732         if self.embed and self.format not in self._ACCEPTABLE_EMBEDDINGS:
    --> 733             raise ValueError("Cannot embed the '%s' image format" % (self.format))
        734         self.width = width
        735         self.height = height
    

    ValueError: Cannot embed the 'gif' image format



    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\motion_bin.gif')


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-20-4a76be10d634> in <module>()
    ----> 1 Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\motion_bin.gif')
    

    C:\Users\Sandipan.Dey\Anaconda\envs\dato-env\lib\site-packages\IPython\core\display.pyc in __init__(self, data, url, filename, format, embed, width, height, retina, unconfined, metadata)
        731 
        732         if self.embed and self.format not in self._ACCEPTABLE_EMBEDDINGS:
    --> 733             raise ValueError("Cannot embed the '%s' image format" % (self.format))
        734         self.width = width
        735         self.height = height
    

    ValueError: Cannot embed the 'gif' image format



    Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\motion_adi.gif')


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-21-35b398682992> in <module>()
    ----> 1 Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\motion_adi.gif')
    

    C:\Users\Sandipan.Dey\Anaconda\envs\dato-env\lib\site-packages\IPython\core\display.pyc in __init__(self, data, url, filename, format, embed, width, height, retina, unconfined, metadata)
        731 
        732         if self.embed and self.format not in self._ACCEPTABLE_EMBEDDINGS:
    --> 733             raise ValueError("Cannot embed the '%s' image format" % (self.format))
        734         self.width = width
        735         self.height = height
    

    ValueError: Cannot embed the 'gif' image format



    

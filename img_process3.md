
#Some more Image Processing: Ostu's Method, Hough Transform and Motion-based Segmentation with Python and OpenCV

Some of the following problems appeared in the lectures and the exercises in the **coursera course Image Processing (by NorthWestern University)**. Some of the following descriptions of the problems are taken from the exercise's description.

   
## 1 . Ostu's method for automatic thresholding to get binary images

We need to find a *thershold* to binarize an image, by separating the *background* from the *foreground*. Using *Ostu's method* we can automatically find the *global optimal threshold*, by maximizing the *between-class variance*. The following figure shows the outline for the technique.

![png](https://sandipanweb.files.wordpress.com/2017/05/im101.png)


The following figures show the **thresholds** computed and the output **binary** images obtained by using **Ostu's method** on a few *gray-scale* **low-contrast** images: 


![png](https://sandipanweb.files.wordpress.com/2017/05/o11.png)

The following figures show the output **binary** images obtained by using **Ostu's method** on a few *gray-scale* **high-contrast** images obtained using **histogram equalization**:


![png](https://sandipanweb.files.wordpress.com/2017/05/o21.png)


The following figures show the distribution of the **between-class variance** for different images and the **htreshold** chosen by **Ostu's** method:

![png](https://sandipanweb.files.wordpress.com/2017/05/ostu.png)

## 2 . Hough Transform to find lines in images

The *Hough transform* can be used to find lines in an image by taking votes from each of the pixels for all possible orientations of lines, for different values of the parameters in a straight line's equation. The following figure describes the method.

![png](https://sandipanweb.files.wordpress.com/2017/05/im141.png)

The following figures show a couple of binary images and the *votes* obtained by the image pixels in the *parametric space* of any straight line and how thresholding on the votes we can find the *most prominent line(s)* in the image (markd by *green*). The parameter values marked with *arrow(s)* represent the most prominent line(s) in the image. 

![png](https://sandipanweb.files.wordpress.com/2017/05/hog2.png)

The following results are inspired by the youtube video *How Hough Transform works* (by *Thales Sehn Korting*) which can be found here: https://www.youtube.com/watch?v=4zHbI-fFIlI. Again, by thresholding on the votes by the pixels for the differrent values of the straight line parameters, we can find the most prominent line(s) in the image (markd by green). The parameter values marked with arrow(s) represent the most prominent line(s) in the image.

![png](https://sandipanweb.files.wordpress.com/2017/05/hog.png)

The next figure shows all the steps in how *Hough transform* can be used on any (*gray-level*) image to find lines (edges). As expected, the more voting threshold is increased, the lesser lines / edges are detected (since the more prominent ones will be detected by the Hough transform, they are colored red).

![png](https://sandipanweb.files.wordpress.com/2017/05/tiger_hough.png)
![png](https://sandipanweb.files.wordpress.com/2017/05/umbc_hough.png)

## 3. Motion based Segmentation using Accumulative Difference Image
In this problem we basically want to separate the **moving objects** in *consectuive frames* of a video from the **non-moving objects**. The following figures show the problem statement that appeared in an assignment in the same course and also the theory to be used: 

![png](https://sandipanweb.files.wordpress.com/2017/05/im152.png)

The following animation shows the motion of a moving rectangle. 

![png](https://sandipanweb.files.wordpress.com/2017/05/motion.gif)

The next figure shows how the motion-based using ADI based techniques can be applied to separate out the moving rectangle from the static backgorund.

![png](https://sandipanweb.files.wordpress.com/2017/05/adi.png)

Again, the next animations show how the moving objects can be segemented from the non-moving ones from the consecutive frames of a video. First the frames from the video are *binarized* using **Ostu's** method and then **absolute ADI** is applied to separate out the moving objects.

![png](https://sandipanweb.files.wordpress.com/2017/05/in2.gif)

![png](https://sandipanweb.files.wordpress.com/2017/05/motion_bin.gif)

![png](https://sandipanweb.files.wordpress.com/2017/05/motion_adi.gif)

The next video is from some past cricket match with Sachin Tendulkar batting (taken from youtube) and the following one is the motion-segmented video with ADI:

![png](https://sandipanweb.files.wordpress.com/2017/05/sachin.gif)
![png](https://sandipanweb.files.wordpress.com/2017/05/sachin_adi.gif)

The next video is captured at a  live performance of a dance-drama written by Tagore and the following one is the motion-segmented video with ADI:

![png](https://sandipanweb.files.wordpress.com/2017/05/kalmrigaya.gif)
![png](https://sandipanweb.files.wordpress.com/2017/05/kalmrigaya_adi1.gif)


    

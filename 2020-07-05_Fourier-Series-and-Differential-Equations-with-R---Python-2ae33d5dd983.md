Fourier Series and Differential Equations with R / Python {.p-name}
=========================================================

Fourier Series and Differential Equations with some applications in R
and Python

* * * * *

### Fourier Series and Differential Equations with R / Python {#7816 .graf .graf--h3 .graf--leading .graf--title name="7816"}

#### Fourier Series and Differential Equations with some applications in R and Python {#9313 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="9313"}

In this article, a few applications of Fourier Series in solving
differential equations will be demonstrated. All the problems are taken
from the **edx Course: MITx — 18.03Fx: Differential Equations Fourier
Series and Partial Differential Equations**.

### Fourier Series: an introduction {#5436 .graf .graf--h3 .graf-after--p name="5436"}

First a basic introduction to the Fourier series will be given and then
we shall see how to solve the following ODEs / PDEs using Fourier
series:

1.  Find the steady state solution to un-damped / damped systems with
    pure / near **resonance**
2.  Solve the *4th* order differential equation for **beam bending**
    system with **boundary values**, using theoretical and numeric
    techniques.
3.  Solve the *PDE* for **Heat** / Diffusion using **Neumann** /
    **Dirichlet** boundary conditions
4.  Solve **Wave** equation (PDE).
5.  Remove noise from an audio file with **Fourier transform**.

### Some basics {#d4d4 .graf .graf--h3 .graf-after--li name="d4d4"}

Let f(t) be a periodic function with period L. Then the Fourier series
of f(t)f(t) is given by the following superposition

![](https://cdn-images-1.medium.com/max/800/0*2dOv32DS6_Tj_K2I)

The above formulae can be simplified as below, for even and odd
functions

![](https://cdn-images-1.medium.com/max/800/0*AsK9fs7ujwWTMoxP)

A few popular 2π2π periodic functions and their Fourier series are shown
below:

![](https://cdn-images-1.medium.com/max/800/0*HLHYFLo9OGLhPfmQ)

### Computing the Fourier coefficients for the Square wave {#7861 .graf .graf--h3 .graf-after--figure name="7861"}

![](https://cdn-images-1.medium.com/max/800/0*NMrxSkBXlCHDRJjb)

### Computing the coefficients in R {#3c93 .graf .graf--h3 .graf-after--figure name="3c93"}

Let’s first plot the square wave function using the following R code

``` {#084f .graf .graf--pre .graf-after--p name="084f"}
library(ggplot2)library(latex2exp)
```

``` {#cf10 .graf .graf--pre .graf-after--pre name="cf10"}
Sq = function(t) { ifelse(t > 0, ifelse(as.integer(t / pi) %% 2 == 0, 1, -1), ifelse(as.integer(t / pi) %% 2 == 1, 1, -1))}
```

``` {#d009 .graf .graf--pre .graf-after--pre name="d009"}
t = seq(-3*pi, 3*pi, 0.01)ggplot() + geom_line(aes(t, Sq(t)), size=1.5, col='red') +              ylab('f(t)') +           scale_x_continuous(breaks=seq(-3*pi, 3*pi, pi),              labels=c(TeX('$-3\\pi$'),TeX('$-2\\pi$'),                    TeX('$\\pi$'),0,TeX('$\\pi$'),                    TeX('$2\\pi$'),TeX('$3\\pi$'))) +           scale_y_continuous(breaks=c(-1,0,1),                     labels=c('-1', '0', '1')) +           ggtitle(TeX(paste('Square wave of period $2\\pi$'))) +           theme(plot.title = element_text(hjust = 0.5))
```

![](https://cdn-images-1.medium.com/max/800/0*F8W40XF1FVdijgmO)

``` {#0843 .graf .graf--pre .graf-after--figure name="0843"}
for (n in 1:6) { b_n = round(2/pi * integrate(function(t) sin(n*t), 0, pi)$value,                                                                 15) print(paste(b_n, ifelse(n %% 2 == 0, 0, 4/n/pi)))}#[1] "1.27323954473516 1.27323954473516"#[1] "0 0"#[1] "0.424413181578388 0.424413181578388"#[1] "0 0"#[1] "0.254647908947032 0.254647908947033"#[1] "0 0"
```

R **symbolic computation package** rSymPy can be used to compute the
Fourier coefficients as shown below:

``` {#1669 .graf .graf--pre .graf-after--p name="1669"}
library(rSymPy)t = Var("t")n = Var("n")sympy("integrate(2*1*sin(n*t)/pi,(t,0,pi))") # sq#[1] "-2*cos(pi*n)/(pi*n) + 2/(pi*n)"
```

The next animation shown how the first few terms in the Fourier series
approximates the periodic square wave function.

![](https://cdn-images-1.medium.com/max/800/0*1XmeDyd0lJCqtmWj)

Notice from the above animation that the convergence of the Fourier
series with the original periodic function is very slow near
discontinuities (the square wave has discontinuities at points
t=kπ,where k∈Z), which is known as **Gibbs phenomenon**.

### Computing the Fourier Coefficients of the Triangle Wave using Anti-derivatives {#b9eb .graf .graf--h3 .graf-after--p name="b9eb"}

Let’s aim at computing the Fourier coefficients for the 2π-periodic
triangle wave by using the coefficients of the 2π-periodic square wave,
using the anti-derivative property.

![](https://cdn-images-1.medium.com/max/800/0*0R3locxpJT-JnxM1)

First let’s write a few lines of code to plot the triangle wave.

``` {#4b59 .graf .graf--pre .graf-after--p name="4b59"}
Tr = function(t) { i = as.integer(t / pi) ifelse(t > 0, ifelse(i %% 2 == 0, t - i*pi, -t + (i+1)*pi), ifelse(i %% 2 == 0, -t + i*pi, t - (i-1)*pi))}t = seq(-3*pi, 3*pi, 0.01)ggplot() +     geom_line(aes(t, Tr(t)), size=1.5, col='red') +     ylab('f(t)') +     scale_x_continuous(breaks=seq(-3*pi, 3*pi, pi),                        labels=c(TeX('$-3\\pi$'), TeX('$-2\\pi$'),                                 TeX('$\\pi$'), 0, TeX('$\\pi$'),                                 TeX('$2\\pi$'), TeX('$3\\pi$'))) +     scale_y_continuous(breaks=c(-pi,0,pi),                      labels=c(TeX('$-\\pi$'), '0', TeX('$\\pi$'))) +     ggtitle(TeX(paste('Triangle wave of period $2\\pi$'))) +     theme(plot.title = element_text(hjust = 0.5))
```

![](https://cdn-images-1.medium.com/max/800/0*Kwfzj5ESlaZlBS1h)

Image by Author

### Computing the constant term {#3421 .graf .graf--h3 .graf-after--figure name="3421"}

``` {#c0ed .graf .graf--pre .graf-after--h3 name="c0ed"}
a_0 = 1/pi*integrate(t, 0, pi)$valueprint(paste(a_0, pi/2))#[1] "1.5707963267949 1.5707963267949"
```

The next animation shows how the superposition of first few terms in the
Fourier series approximates the triangle wave function.

![](https://cdn-images-1.medium.com/max/800/0*Tv4Kc8HxtNxYy6gF)

Image by Author

### Computing the Fourier Coefficients of the Sawtooth function {#142c .graf .graf--h3 .graf-after--figure name="142c"}

Similarly, let’s visualize the sawtooth wave function using the
following R code snippet:

``` {#5cd3 .graf .graf--pre .graf-after--p name="5cd3"}
St = function(t) { i = floor((t + pi) / (2*pi)) t - i*2*pi}t = seq(-3*pi, 3*pi, 0.01)ggplot() +    geom_line(aes(t, St(t)), size=1.5, col='red') +    scale_x_continuous(breaks=seq(-3*pi, 3*pi, pi),                       labels=c(TeX('$-3\\pi$'), TeX('$-2\\pi$'),                                TeX('$\\pi$'), 0, TeX('$\\pi$'),                                TeX('$2\\pi$'), TeX('$3\\pi$'))) +    scale_y_continuous(breaks=c(-pi,0,pi),                        labels=c(TeX('-$\\pi$'), '0',                                 TeX('$\\pi$'))) +     ylab('f(t)') +    ggtitle(TeX(paste('Sawtooth wave of period $2\\pi$'))) +    theme(plot.title = element_text(hjust = 0.5))
```

![](https://cdn-images-1.medium.com/max/800/0*vcL1Ja0xUMTXRKm4)

Image by Author

The following animation shows how the superposition of first few terms
in the Fourier series of the **Sawtooth** wave approximate the function.

![](https://cdn-images-1.medium.com/max/800/0*o39W7Ut6ZxU3M4r5)

Image by Author

### Computing the Fourier series for the even periodic extension of a function {#2b8d .graf .graf--h3 .graf-after--figure name="2b8d"}

Let’s first plot the even periodic extension of the function \
f(t) = t\^2, -1 ≤ t ≤ 1, with the following R code.

``` {#cc6c .graf .graf--pre .graf-after--p name="cc6c"}
Pr = function(t) { i = as.integer(abs(t-ifelse(t > 0,-1,1)) / 2) ifelse(t > 0, (t-2*i)^2, (t+2*i)^2)}
```

``` {#8cdc .graf .graf--pre .graf-after--pre name="8cdc"}
t = seq(-5, 5, 0.01)ggplot() +    geom_line(aes(t, Pr(t)), size=1.5, col='red') +    scale_x_continuous(breaks=-5:5, labels=-5:5) +    ylab('f(t)') +    ggtitle(TeX(paste('Even periodic extension of the function            $f(t) = t^2$, $-1\\leq t \\leq 1$'))) +    theme(plot.title = element_text(hjust = 0.5))
```

![](https://cdn-images-1.medium.com/max/800/0*3iqH6kP3y2bayJfs)

Image by Author

Here is another function which is neither even nor odd, with period 2π

![](https://cdn-images-1.medium.com/max/800/0*ommQUi4ir1q6HmAA)

Image by Author

The following animation shows how the Fourier series of the function
approximates the above function more closely as more and more terms are
added.

![](https://cdn-images-1.medium.com/max/800/0*jQFhHnWMHtFcduYK)

Image by Author

### 1. Solution of ODE with ERF and Resonance {#70ec .graf .graf--h3 .graf-after--figure name="70ec"}

![](https://cdn-images-1.medium.com/max/800/0*YLK629jBNnDYEX91)

Let’s solve the following differential equation (for a system without
damping) and find out when the pure resonance takes place.

![](https://cdn-images-1.medium.com/max/800/0*6V5s-f6djwf5q7Mc)

The next section shows how to find the largest gain corresponding to the
following system with damping.

![](https://cdn-images-1.medium.com/max/800/0*gyvwcVi7EpFh1oFK)

Let’s plot the amplitudes of the terms to see which term has the largest
gain.

``` {#8cd4 .graf .graf--pre .graf-after--p name="8cd4"}
n = seq(1,15,2)f = (1/sqrt((49-n^2)^2 + (0.1*n)^2)/n)ggplot() + geom_point(aes(n, f)) +           geom_line(aes(n, f)) + ylab('c_n') +           scale_x_continuous(breaks=seq(1,15,2),                              labels=as.character(seq(1,15,2)))
```

![](https://cdn-images-1.medium.com/max/800/0*6gUwo3Am61POZ7xl)

Image by Author

### 2. Boundary Value Problems {#e86c .graf .graf--h3 .graf-after--figure name="e86c"}

![](https://cdn-images-1.medium.com/max/800/0*aUbbhJJc95iTSv9t)

As expected, since the right end of the beam is free, it will bend down
when more loads are applied on the beam to the transverse direction of
its length. The next animation shows how the beam bending varies with
the value of q.

![](https://cdn-images-1.medium.com/max/800/0*vm5x6NS56BvkIOHr)

Image by Author

### Numerically solving a linear system to obtain the solution of the beam-bending system represented by the 4th-order differential equation in R {#2978 .graf .graf--h3 .graf-after--figure name="2978"}

![](https://cdn-images-1.medium.com/max/800/0*eOfPi12myM83BktC)

First create a near-**tri-diagonal** matrix **A** that looks like the
following one, it takes care of the differential coefficients of the
beam equation along with all the boundary value conditions.

![](https://cdn-images-1.medium.com/max/800/0*BRrDnQ14fCDjAXFN)

Now, let’s solve the linear system using the following R code.

``` {#7937 .graf .graf--pre .graf-after--p name="7937"}
# Create a vector b that is zero for boundary conditions and         # -0.0000001 in every other entry.b = rep(1,10)*(-1e-7)# Create a vector v that solves Av = b.v = solve(A, b)# Create a column vector x of 10 evenly spaced points between # 0 and 1 (for plotting)x = seq(0,1,1/9)# Plot v on the vertical axis, and x on the horizontal axis.print(ggplot() + geom_line(aes(x,v), col='blue', size=2) +                                  geom_point(aes(x, v), size=2) +ggtitle(paste('Solution of the linear (beam bending) system')) + theme_bw())
```

![](https://cdn-images-1.medium.com/max/800/0*I5BiK_gYm7uFnLZx)

Image by Author

As can be seen from the above figure, the output of the numerical
solution agrees with the one obtained above theoretically.

### 3. Heat / Diffusion Equation {#26d1 .graf .graf--h3 .graf-after--p name="26d1"}

![](https://cdn-images-1.medium.com/max/800/0*WkjOUOaFYGRhr4QO)

![](https://cdn-images-1.medium.com/max/800/0*-F2DFpGLMk9WkBy8)

The following animation shows how the temperature changes on the bar
with time (considering only the first 100 terms for the Fourier series
for the square wave).

![](https://cdn-images-1.medium.com/max/800/0*NK47i9iKu-iwrguQ)

Image by Author

Let’s solve the below diffusion *PDE* with the given ***Neumann BCs***.

![](https://cdn-images-1.medium.com/max/800/0*AlgSTe5fkX-0fAa4)

As can be seen from above, the initial condition can be represented as a
2-periodic triangle wave function (using even periodic extension), i.e.,

![](https://cdn-images-1.medium.com/max/800/0*8-s7g2BbRkCFDrJk)

The next animation shows how the different points on the tube arrive at
the steady-state solution over time.

![](https://cdn-images-1.medium.com/max/800/0*mx2TXRngy_mlMqPk)

Image by Author

The next figure shows the time taken for each point inside the tube, to
approach within 1% the steady-state solution.

![](https://cdn-images-1.medium.com/max/800/0*4-iWx0OOzNmD_s-m)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*vur4s5rLYl3fTf8_)

with the following

-   L=5, the length of the bar is 5 units.
-   ***Dirichlet BC*s**: *u(0, t) =0*, *u(L,t) = sin(2πt/L)*, i.e., the
    left end of the bar is held at a constant temperature 0 degree (at
    ice bath) and the right end changes temperature in a sinusoidal
    manner.
-   **IC**: *u(x,0) = 0*, i.e., the entire bar has temperature 0 degree.

The following R code implements the numerical method:

``` {#8ec0 .graf .graf--pre .graf-after--p name="8ec0"}
#Solve heat equation using forward time, centered space scheme#Define simulation parametersx = seq(0,5,5/19) #Spatial griddt = 0.06 #Time steptMax = 20 #Simulation timenu = 0.5 #Constant of proportionalityt = seq(0, tMax, dt) #Time vectorn = length(x)m = length(t)fLeft = function(t) 0 #Left boundary conditionfRight = function(t) sin(2*pi*t/5) #Right boundary conditionfInitial = function(x) matrix(rep(0,m*n), nrow=m) #Initial condition#Run simulationdx = x[2]-x[1]r = nu*dt/dx^2#Create tri-diagonal matrix A with entries 1-2r and r
```

The tri-diagonal matrix A is shown in the following figure

![](https://cdn-images-1.medium.com/max/800/0*PZgfWL9XaoFLnHF8)

Image by Author

``` {#979c .graf .graf--pre .graf-after--figure name="979c"}
#Impose inital conditionsu = fInitial(x)u[1,1] = fLeft(0)u[1,n] = fRight(0)for (i in 1:(length(t)-1)) { u[i+1,] = A%*%(u[i,]) # Find solution at next time step u[i+1,1] = fLeft(t[i]) # Impose left B.C. u[i+1,n] = fRight(t[i]) # Impose right B.C.}
```

The following animation shows the solution obtained to the heat equation
using the numerical method (in R) described above,

![](https://cdn-images-1.medium.com/max/800/0*P0fK6N9UZWzakEkO)

Image by Author

### 4. Wave PDE {#4a06 .graf .graf--h3 .graf-after--figure name="4a06"}

![](https://cdn-images-1.medium.com/max/800/0*_uqb4wmKY_Plldoz)

![](https://cdn-images-1.medium.com/max/800/0*3RS4YzT5ZEYoK61D)

The next figure shows how can a numerical method be used to solve the
wave PDE

![](https://cdn-images-1.medium.com/max/800/0*nscytUOnluIpIqF4)

### Implementation in R {#dcfe .graf .graf--h3 .graf-after--figure name="dcfe"}

``` {#8c64 .graf .graf--pre .graf-after--h3 name="8c64"}
#Define simulation parametersx = seq(0,1,1/99) #Spatial griddt = 0.5 #Time steptMax = 200 #Simulation timec = 0.015 #Wave speedfLeft = function(t) 0 #Left boundary conditionfRight = f(t) 0.1*sin(t/10) #Right boundary conditionfPosInitial = f(x) exp(-200*(x-0.5)^2) #Initial positionfVelInitial = f(x) 0*x #Initial velocity#Run simulationdx = x[2]-x[1]r = c*dt/dxn = length(x)t = seq(0,tMax,dt) #Time vectorm = length(t) + 1
```

Now iteratively compute *u(x,t)* by imposing the following **boundary
conditions**

​1. *u(0,t) = 0*\
2. *u(L,t) = (1/10).sin(t/10)*

along with the following **initial conditions**

​1. *u(x,0) = exp(-500.(x-1/2)2)*\
2. *∂u(x,0)/∂t = 0.x*

as defined in the above code snippet.

``` {#ec33 .graf .graf--pre .graf-after--p name="ec33"}
#Create tri-diagonal matrix A here, as described in the algorithm#Impose initial conditionu <- matrix(rep(0, n*m), nrow=m)u[1,] = fPosInitial(x)u[1,1] = fLeft(0)u[1,n] = fRight(0)for (i in 1:length(t)) { #Find solution at next time step if (i == 1) {   u[2,] = 1/2*(A%*%u[1,]) + dt*fVelInitial(x) } else {   u[i+1,] = (A%*%u[i,])-u[i-1,] } u[i+1,1] = fLeft(t[i]) #Impose left B.C. u[i+1,n] = fRight(t[i]) #Impose right B.C.}
```

The following animation shows the output of the above implementation of
the solution of wave PDE using R, it shows how the waves propagate,
given a set of BCs and ICs.

![](https://cdn-images-1.medium.com/max/800/0*fO7oXYuUgcHhYOmP)

Image by Author

### 5. Audio processing with Python: De-noising an audio file with Fourier Transform {#23d0 .graf .graf--h3 .graf-after--figure name="23d0"}

Let’s denoise an input noisy audio file (part of the theme music from
Satyajit Ray’s famous movie পথের পাঁচালী, Songs of the road) using
python *scipy.fftpack* module’s *fft()* implementation.

The noisy input file was generated and uploaded
[here](https://github.com/sandipan/edx/blob/courses/pather_panchali_noisy.wav).

``` {#247b .graf .graf--pre .graf-after--p name="247b"}
from scipy.io import wavfileimport scipy.fftpack as fpimport numpy as npimport matplotlib.pylab as plt# first fft to see the pattern then we can edit to see signal and inverse# back to a better soundFs, y = wavfile.read('pather_panchali_noisy.wav') # can be found here: "https://github.com/sandipan/edx/blob/courses/pather_panchali_noisy.wav" data-mce-href="https://github.com/sandipan/edx/blob/courses/pather_panchali_noisy.wav".# you may want to scale y, in this case, y was already scaled in between [-1,1]Y = fp.fftshift(fp.fft(y))  #Take the Fourier series and take a symmetric shiftfshift = np.arange(-n//2, n//2)*(Fs/n)plt.plot(fshift, Y.real) #plot(t, y); #plot the sound signalplt.show()
```

You will obtain a figure like the following:

![](https://cdn-images-1.medium.com/max/800/0*W0S5gHFmqU_lISqN)

Image by Author

``` {#eac4 .graf .graf--pre .graf-after--figure name="eac4"}
L = len(fshift) #Find the length of frequency valuesYfilt = Y# find the frequencies corresponding to the noise hereprint(np.argsort(Y.real)[::-1][:2])# 495296 639296Yfilt[495296] = Yfilt[639296] = 0  # find the frequencies corresponding to the spikesplt.plot(fshift, Yfilt.real)plt.show()soundFiltered = fp.ifft(fp.ifftshift(Yfilt)).realwavfile.write('pather_panchali_denoised.wav', Fs, soundNoisy)
```

![](https://cdn-images-1.medium.com/max/800/0*O3qyZoyrWhb4imxL)

Image by Author

The de-noised output wave file produced with the filtering can be found
[here](https://github.com/sandipan/edx/blob/courses/pather_panchali_denoised.wav).
(use vlc media player to listen to input and output wave files).

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [July 5,
2020](https://medium.com/p/2ae33d5dd983).

[Canonical
link](https://medium.com/@sandipan-dey/in-this-article-a-few-applications-of-fourier-series-in-solving-differential-equations-will-be-2ae33d5dd983)

Exported from [Medium](https://medium.com) on January 8, 2021.

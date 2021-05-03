Euler Method to solve Ordinary differential Equations with R {.p-name}
============================================================

Solving 1st and 2nd order ODEs with Euler method

* * * * *

### Euler Method to solve Ordinary differential Equations with R {#512b .graf .graf--h3 .graf--leading .graf--title name="512b"}

#### Solving 1st and 2nd order ODEs with Euler method {#b4e0 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="b4e0"}

In this article, the simplest numeric method, namely the Euler method to
solve 1st order ODEs will be demonstrated with examples (and then use it
to solve 2nd order ODEs). Also, we shall see how to plot the phase lines
(gradient fields) for an ODE and understand from examples how to
qualitatively find a solution curve with the phaselines. All the
problems are taken from the **edx Course: MITx — 18.03x: Introduction to
Differential Equations**.

![](https://cdn-images-1.medium.com/max/800/0*qLMx2-HdHve3reYx)

Image created from the lecture notes of the edX course MITx:
18.031x ([source](https://courses.edx.org/courses/course-v1:MITx+18.031x+2T2020/course/))

Let’s implement the above algorithm steps with the following R code
snippet.

![](https://cdn-images-1.medium.com/max/800/1*kJB8O7B0yTSRDRvRKaTpHw.png)

Let’s define the function *f* next.

![](https://cdn-images-1.medium.com/max/800/1*dIJ4bTyJoKKbuovjiEiCGw.png)

The below table shows the solution curve computed by the Euler method
for the first few iterations:

![](https://cdn-images-1.medium.com/max/800/0*uWQOf_UFaxp5AE0G)

![](https://cdn-images-1.medium.com/max/800/0*8rS4qoGwGj1e6okl)

Image created from the lecture notes of the edX course MITx:
18.031x ([source](https://courses.edx.org/courses/course-v1:MITx+18.031x+2T2020/course/))

Let’s implement the following R functions to draw the isoclines and the
phase lines for an ODE.

![](https://cdn-images-1.medium.com/max/800/1*PZBD7I-sWl8mx0VlmhfaFg.png)

![](https://cdn-images-1.medium.com/max/800/1*XcUi6B013GxRKqSq3Bo73g.png)

Now, let’s plot the 0 (the nullcline), 1, 2, 3 isoclines for the ODE:
*y’ = y.sin(x)*.

![](https://cdn-images-1.medium.com/max/800/1*TqiP379LZMl0_wIaVsdccQ.png)

![Image by
author](https://cdn-images-1.medium.com/max/800/0*zDdb862bYHxRmQzv)

Image by author

![](https://cdn-images-1.medium.com/max/800/1*P_Ui1CZJfG_XhQEDJREk8A.png)

![](https://cdn-images-1.medium.com/max/800/0*p5XawvfXU2CIYqsO)

Image by author

Now, let’s solve the ODE with Euler’s method, using step size *h = 0.1*
with the initial value *y(-3) = 1*.

``` {#ebcd .graf .graf--pre .graf-after--p name="ebcd"}
l <- 0.2
```

``` {#ef8a .graf .graf--pre .graf-after--pre name="ef8a"}
xgap <- 0.05 #05
```

``` {#04a8 .graf .graf--pre .graf-after--pre name="04a8"}
ygap <- 0.05 #05
```

``` {#286d .graf .graf--pre .graf-after--pre name="286d"}
p <- plot_phaselines(f, rx, ry, xgap, ygap, l, -1:1, TeX('phaselines and m-isoclines for $\\frac{dy}{dx} = y.sin(x)$'))
```

``` {#5ada .graf .graf--pre .graf-after--pre name="5ada"}
df <- Euler(f, -3, 1, 0.1, 100)
```

``` {#2249 .graf .graf--pre .graf-after--pre name="2249"}
p + geom_point(aes(Xn, Yn), data=df) + geom_path(aes(Xn, Yn), arrow = arrow(), data=df)
```

The black curve in the following figure shows the solution path with
Euler.

![](https://cdn-images-1.medium.com/max/800/0*oSF_2JJXLMtCaytZ)

Image by author

Let’s plot the phase lines for the ODE *y’ = y\^2 — x*, solve it with
Euler and plot the solution curve again, using the following code.

![](https://cdn-images-1.medium.com/max/800/1*q8-eQPA7e9aNKg-eXeXHsg.png)

The black solution curve shows the trajectory of the solution obtained
with Euler.

![](https://cdn-images-1.medium.com/max/800/0*RY4_QTC2HXPdS96t)

Image by author

Finally, let’ plot the isoclines for the ODE *y’ = y\^2 — x\^2* and
solve it with Euler. Let’s try starting with different initial values
and observe the solution curves in each case.

``` {#3ffc .graf .graf--pre .graf-after--p name="3ffc"}
f <- function(x, y) {  return(y^2-x^2)}
```

``` {#3b8b .graf .graf--pre .graf-after--pre name="3b8b"}
rx <- c(-4,4)ry <- c(-4,4)l <- 0.2xgap <- 0.1ygap <- 0.1
```

``` {#4e7f .graf .graf--pre .graf-after--pre name="4e7f"}
p <- plot_phaselines(f, rx, ry, xgap, ygap, l, -2:2, TeX('phaselines and m-isoclines for $\\frac{dy}{dx} = y^2-x^2$, solving with Euler with diff. initial values'))
```

``` {#b733 .graf .graf--pre .graf-after--pre name="b733"}
df <- Euler(f, -3.25, 3, 0.1, 100)p <- p + geom_point(aes(Xn, Yn), data=df) + geom_path(aes(Xn, Yn), arrow = arrow(), data=df)
```

As can be seen, the solution curve starting with different initial
values lead to entirely different solution paths, it depends on the
phase lines and isoclines it gets trapped into.

![](https://cdn-images-1.medium.com/max/800/0*gJXpuAu8ZGL3YvEH)

Image by author

Finally, let’s notice the impact of the step size on the solution
obtained by the Euler method, in terms of the error and the number of
iterations taken to obtain the solution curve. As can be seen from the
following animation, with bigger step size h=0.5, the algorithm
converges to the solution path faster, but with much higher error
values, particularly for the initial iterations. On the contrary,
smaller step size h=0.1 taken many more iterations to converge, but the
error rate is much smaller, the solution curve is pretty smooth and
follows the isoclines without any zigzag movement.

![](https://cdn-images-1.medium.com/max/800/1*kTd4Robs6oLJFYRTuwG-eQ.gif)

Image by author

### Spring-mass-dashpot system — the Damped Harmonic Oscillator {#8b1e .graf .graf--h3 .graf-after--figure name="8b1e"}

A spring is attached to a wall on its left and a cart on its right. A
dashpot is attached to the cart on its left and the wall on its right.
The cart has wheels and can roll either left or right.

The spring constant is labeled *k*, the mass is labeled *m*, and the
damping coefficient of the dashpot is labeled b. The center of the cart
is at the rest at the location *x* equals 0, and we take rightwards to
be the positive *x* direction.

Consider the spring-mass-dashpot system in which mass is*m*(kg), and
spring constant is *k* (Newton per meter). The position *x(t)* (meters)
of the mass at (seconds) is governed by the DE

![](https://cdn-images-1.medium.com/max/800/0*_yNb1Zos9cr6oNrP)

where m, b, k \> 0. Let’s investigate the effect of the damping
constant, on the solutions to the DE.

![](https://cdn-images-1.medium.com/max/800/0*90LOOVrJFjwdWpCw)

Image created from the lecture notes of the edX course MITx:
18.031x ([source](https://courses.edx.org/courses/course-v1:MITx+18.031x+2T2020/course/))

The next animation shows how the solution changes for different values
of 𝑏, given 𝑚=1 and 𝑘=16, with couple of different initial values ©.

![](https://cdn-images-1.medium.com/max/800/1*mvhZAPveplCAGTLYNXYkFw.gif)

Image by author

Now, let’s observe how the amplitude of the system response increases
with the increasing frequency of the input sinusoidal force, achieving
the peak at resonance frequency (for *m=1*, *k=16* and *b=√14*, the
resonance occurs at *wr = 3*)

![](https://cdn-images-1.medium.com/max/800/0*xf-weUo3xaOrIAO0)

Image by author

### Solving the 2nd order ODE for the damped Harmonic Oscillator with Euler method {#07eb .graf .graf--h3 .graf-after--figure name="07eb"}

Now, let’s see how the 2nd order ODE corresponding to the damped
Harmonic oscillator system can be numerically solved using the Euler
method. In order to be able to use Euler, we need to represent the 2nd
order ODE as a collection of 1st order ODEs and then solve those 1st
order ODEs simultaneously using Euler, as shown in the following figure:

![](https://cdn-images-1.medium.com/max/800/0*kqMc5br1qQOEsnSH)

Created from the Youtube video by Jeffrey
Chasnov ([source](https://www.youtube.com/watch?v=QuyBVdDHkZY))

Let’s implement a numerical solver with Euler for a 2nd order ODE using
the following R code snippet:

![](https://cdn-images-1.medium.com/max/800/1*T0yXfFDnWKiBKrlBtOGM-w.png)

For the damped Harmonic Oscillator, we have the 2nd order ODE

![](https://cdn-images-1.medium.com/max/800/0*fACWEmX49u4wVV7O)

from which we get the following two 1st order ODEs:

![](https://cdn-images-1.medium.com/max/800/0*o9yC1kl53bDXsy6t)

Let’s solve the damped harmonic oscillator with Euler method:

![](https://cdn-images-1.medium.com/max/800/1*LbPi3INsuc--gQpjZGxIEQ.png)

Now, let’s plot the solution curves using the following R code block:

``` {#ee09 .graf .graf--pre .graf-after--p name="ee09"}
par(mfrow=c(1,2))
```

``` {#2209 .graf .graf--pre .graf-after--pre name="2209"}
plot(df$tn, df$xn, main="x(t) vs. t", xlab='t', ylab='x(t)',       col='red', pch=19, cex=0.7, xaxs="i", yaxs="i")lines(df$tn, df$xn, col='red')grid(10, 10)
```

``` {#e3d1 .graf .graf--pre .graf-after--pre name="e3d1"}
plot(df$tn, df$un, main=TeX('$\\dot{x}(t)$ vs. t'), xlab='t',      ylab=TeX('$\\dot{x}(t)$'), col='blue', pch=19, cex=0.7,      xaxs="i", yaxs="i")lines(df$tn, df$un, col='blue')grid(10, 10)
```

![](https://cdn-images-1.medium.com/max/800/0*U4J0GV2jmmqt8PoT)

Image by author

The following animation shows the solution curves obtained by joining
the points computed with the Euler method for the damped Harmonic
Oscillator:

![](https://cdn-images-1.medium.com/max/800/1*_pI42FftfsYIku-h9FTCVw.gif)

Image by author

As can be seen from the above results, using the Euler method we can
solve ODEs of any order, as long as we can represent the system as a
collection of 1st order ODEs. There are more accurate methods for
solving ODEs (e.g., the RK-4 method) than Euler method (also known an
1st order Runge-Kutta method) that produces much lower error in
approximation (e.g., RK-4 has global error of O(h⁴)), we can use them to
obtain more accurate approximation of the solution curve.

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [September 13,
2020](https://medium.com/p/50ec2993042).

[Canonical
link](https://medium.com/@sandipan-dey/euler-method-to-solve-ordinary-differential-equations-with-r-50ec2993042)

Exported from [Medium](https://medium.com) on January 8, 2021.

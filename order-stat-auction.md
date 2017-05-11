---
title: "Some Statistics Concepts: Order Statistics and Application in Auction"
author: "Sandipan Dey"
date: "11 May 2017"
layout: post
comments: true
---

Editor’s note this is a guest post by Sandipan Dey

In this article, the concepts of the *Order Statistics* along with its applications will be discussed. These concepts were discussed in a lecture video in the *edX course MITx 14.310x Data Analysis for Social Scientists*.

Order Statistics
----------------

### Definition

-   Let's consider a collection of *i.i.d. contiuous random variables* *X*<sub>1</sub>, *X*<sub>2</sub>, …, *X*<sub>*n*</sub>. If *X*<sub>(*j*)</sub> is the *j*<sup>*th*</sup> smallest of *X*<sub>1</sub>, *X*<sub>2</sub>, …, *X*<sub>*n*</sub>, it's called the *j*<sup>*th*</sup> order statistics.

-   The 1<sup>*st*</sup> and the *n*<sup>*th*</sup> order statistics are the minimum and maximum of the variables, respectively.

-   We are interested in the *marginal density* and *expected value* of the *j*<sup>*th*</sup> order statistics.

-   The below figure shows the density of different *order statistics* in general and also for specific distributions (*exponential* and *uniform* distributions).

![](https://sandipanweb.files.wordpress.com/2016/12/order_stat.png)

-   The following figure (generated using the following *R* code) and animation show the density of the order statistics for *n* i.i.d random variables *X*<sub>*i*</sub> ∼ *U*(0, 1), for different values of *n*, along with the *expected values* of the *order statistics* as *dotted vertical lines*, quantities that we shall also be interested in.

``` r
library(ggplot2)

order.stat.density <- function(j, n, f, F) {
  function(x) (factorial(n)/(factorial(j-1)*factorial(n-j)))*f(x)*(F(x))^(j-1)*(1-F(x))^(n-j)
}

x <- seq(0,10,0.01)
df <- NULL
n <- 3
for (i in 1:n) {
  df <- rbind(df, data.frame(x, value=order.stat.density(i, n, function(x) 1/10, function(x) x/10)(x), type=paste0('OrderStat',i)))
}
ggplot(df, aes(x, value, col=type)) + geom_line() + ggtitle(paste('Density of the order statistics for n=', 3, 'iid rvs ~ U(0,10)'))
```

![](https://sandipanweb.files.wordpress.com/2016/12/o1.png)

![](https://sandipanweb.files.wordpress.com/2016/12/animation1.gif)

The following animation shows the density of the order statistics for *n i.i.d.* random variables *X*<sub>*i*</sub>∼*Exp*(5).

![](https://sandipanweb.files.wordpress.com/2016/12/animation2.gif)

### Application

-   Now, let's apply the *order statistics* concepts in the following settings of an *auction*.

    -   Let there be *N* potential buyers of some good.

    -   Their valuations are *i.i.d.* with *U*(0, 1).

    -   The seller can offer the good
        -   at no cost
        -   at a *posted price*
        -   or can *auction* it off.
    -   The seller knows the distribution of valuations, but does not know the individual realizations.

    -   As shown in the following figure, the *expected profit* at the *posted price* depends on the *CDF* of the *N*<sup>*th*</sup> *order statistics* of the valuations. The seller wants to *maximize* his *expected profit* and the *optimal posted price* is *(1/(N+1))^(1/N)*, with the *optimal expected profit* as *(N/(N+1))(1/(N+1))^(1/N)*.

    -   Again. as shown in the next figure, the *profit* at the *2nd price auction* depends on the distribution of the (*N* − 1)<sup>*th*</sup> *order statistics* of the valuations and the *expected profit* is computed to be *(N-1)/(N+1)*.

![](https://sandipanweb.files.wordpress.com/2016/12/auction_posted.png)

-   The following animation shows the distribution of the (*N* − 1)<sup>*th*</sup> *order statistics* for *N* valuations for different *N*. The vertical dotted line shows the *expected value* as before.

![](https://sandipanweb.files.wordpress.com/2016/12/animation3.gif)

As can be seen from the following figure, as there are more and more potential buyers *(N >= 3)*, the *2nd price auction* becomes 
more profitable in expectation than the *optimal posted price*.

![](https://sandipanweb.files.wordpress.com/2016/12/auc_post.png)
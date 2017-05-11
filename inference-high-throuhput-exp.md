---
title: "Statistical Inference and Modeling for High-throughput Experiments"
author: "Sandipan Dey"
date: "11 May 2017"
layout: post
comments: true
---

The following *inference* problems (along with the descriptions) are
taken directly from the exercises of the *edX Course HarvardX: PH525.3x
Statistical Inference and Modeling for High-throughput Experiments*.

Inference in Practice Exercises
-------------------------------

These exercises will help clarify that *p-values* are *random variables*
and some of the properties of these *p-values*. Note that just like the
sample average is a random variable because it is based on a random
sample, the p-values are based on random variables (sample mean and
sample standard deviation for example) and thus it is also a random
variable. The following 2 properties of the *p-values* are
mathematically proved in the following figure:

-   The *p-values* are *random variables*.
-   Under *null-hypothesis* the *p-values* form a *uniform
    distribution*.

![](https://sandipanweb.files.wordpress.com/2017/04/im18.png)

To see this, let's see how *p-values* change when we take different
samples. The next table shows the dataset *Bodyweight* from which

1.  The *control* and *treatment* groups each of size 12 are randomly drawn.

2.  Then *2-sample t-test* is performed with this groups to compute the *p-value*.

3.  Steps 1-2 is replicated 10000 times.

Bodyweight
----------

      27.03
      24.80
      27.02
      28.07
      23.55
      22.72

The next table shows randomly chosen *control* and *treatment* group for
a single replication.

<table>
<thead>
<tr class="header">
<th align="right">control</th>
<th align="right">treatment</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="right">21.51</td>
<td align="right">19.96</td>
</tr>
<tr class="even">
<td align="right">28.14</td>
<td align="right">18.08</td>
</tr>
<tr class="odd">
<td align="right">24.04</td>
<td align="right">20.81</td>
</tr>
<tr class="even">
<td align="right">23.45</td>
<td align="right">22.12</td>
</tr>
<tr class="odd">
<td align="right">23.68</td>
<td align="right">30.45</td>
</tr>
<tr class="even">
<td align="right">19.79</td>
<td align="right">24.96</td>
</tr>
</tbody>
</table>

The following animation shows first 10 replication steps.

![](https://sandipanweb.files.wordpress.com/2017/04/animation.gif)

The following figure shows the distribution of the *p-values* obtained, 
which is nearly *uniform*, as expected.

![](https://sandipanweb.files.wordpress.com/2017/04/pdist.png)

Now, let's assume that we are testing the effectiveness of 20 diets on
mice weight. For each of the 20 diets let's run an experiment with 10
control mice and 10 treated mice. Assume the *null hypothesis* that the
diet has no effect is *true* for all 20 diets and that mice weights
follow a normal distribution with mean 30 grams and a standard deviation
of 2 grams, run a Monte Carlo simulation for one of these studies, to
learn about the distribution of the number of p-values that are less
than 0.05. Let's run these 20 experiments 1,000 times and each time save
the number of *p-values* that are less than 0.05.

The following figures show using *Monte-Carlo* simulations how some of
the *t-tests* reject the *null-hypothesis* (at 5% level of significance)
simply *by chance*, even though it is *true*.

![](https://sandipanweb.files.wordpress.com/2017/04/test.gif)

![](https://sandipanweb.files.wordpress.com/2017/04/fp.png)

The following figure shows how the *FWER* (*Family-wise error rate*,
i.e., *probability of rejecting at least one true null-hypothesis*)
computed with (*Monte-Carlo simulation*) increases with the number of
multiple hypothesis tests.

![](https://sandipanweb.files.wordpress.com/2017/04/f13.png)

The following figures show theoretically how the *FWER* can be computed:

![](https://sandipanweb.files.wordpress.com/2017/04/im25.png)

![](https://sandipanweb.files.wordpress.com/2017/04/f14.png)

Now, let's try to understand the concept of a *error controlling
procedure*. We can think of it as defnining a set of instructions, such
as *"*reject all the null hypothesis for for which p-values &lt;
0.0001"\* or *"reject the null hypothesis* for the 10 features with
smallest p-values". Then, knowing the *p-values* are random variables,
we use statistical theory to compute how many mistakes, on average, will
we make if we follow this procedure. More precisely we commonly bounds
on these rates, meaning that we show that they are smaller than some
predermined value.

We can compute the following different error rates:

1.  The *FWER* (*Family-wise error rate*) tells us the *probability* of
    having *at least one false positive*.
2.  The *FDR* (*False discovery rate*) is the *expected rate* of
    *rejected null hypothesis*.

#### Note 1

The *FWER* and *FDR* are not procedures but error rates. We will review
procedures here and use Monte Carlo simulations to estimate their error
rates.

#### Note 2

We sometimes use the colloquial term "pick genes that" meaning "reject
the null hypothesis for genes that."

Bonferroni Correction Exercises (Bonferonni versus Sidak)
---------------------------------------------------------

Let's consider the following figure:

![](https://sandipanweb.files.wordpress.com/2017/04/im46.png?w=676)

As we have learned about the family wide error rate *FWER*, it is the
probability of incorrectly rejecting the null at least once, i.e., the
probability that *Pr(V&gt;0)*.

What we want to do in practice is choose a *procedure* that guarantees
this probability is smaller than a predetermined value such as
*α* = 0.05.

We have already learned that the procedure "pick all the genes with
p-value &lt;0.05" fails miserably as we have seen that
*P**r*(*V* &gt; 0)∼1. So what else can we do?

The **Bonferroni** procedure assumes we have computed *p-values* for
each test and asks what constant *k* should we pick so that the
procedure "pick all genes with p-value less than *k*" has
*Pr(V&gt;0)=0.05*. And we typically want to be *conservative* rather
than lenient, so we accept a procedure that has *P**r*(*V* &gt; 0)≤0.05.

So the first result we rely on is that this probability is largest when
all the null hypotheses are true:

*P**r*(*V* &gt; 0)≤ *Pr(V&gt;0|all nulls are true)* or
*P**r*(*V* &gt; 0)≤*P**r*(*V* &gt; 0|*m*1 = 0)

If the tests are *independent* then
*P**r*(*V* &gt; 0|*m*1 = 0)=1 − (1 − *k*)<sup>*m*</sup> and we pick *k*
so that 1 − (1 − *k*)<sup>*m*</sup> = *α*⇒
*k* = 1 − (1 − *α*)<sup>1/*m*</sup>, this is called **Sidak** procedure.
Now, this requires the tests to be independent. The **Bonferroni**
procedure does not make this assumption, if we set *k* = *α*/*m* this
procedure has the property that *P**r*(*V* &gt; 0)≤*α*. The following
figure shows the proof.

![](https://sandipanweb.files.wordpress.com/2017/04/f31.png)

Let's plot of *α*/*m* and 1 − (1 − *α*)<sup>1/*m*</sup> for various
values of *m&gt;1*. Which procedure is more conservative (picks less
genes, i.e. rejects less null hypothesis): Bonferroni's or Sidak's? As
can be seen from the next figures, **Bonferroni** is more conservative.

![](https://sandipanweb.files.wordpress.com/2017/04/f41.png)

    ## [1] 0.0467

    ## [1] 0.0473

As explained in
<a href http://genomicsclass.github.io/book/pages/multiple_testing.html></a>, the
*specificity* constraint posed by *FWER* can sometimes be an over-kill.
A widely used alternative to the FWER is the false discover rate
(*FDR*). The idea behind FDR is to focus on the random variable *Q* as
follows:

$$   
Q = 
     \\begin{cases}
       \\frac{V}{R} & R &gt; 0\\\\
       0 & V = R = 0
     \\end{cases}
$$

*Q* is a random variable that can take values between 0 and 1 and we can
define a rate by considering the average (expected value) of *Q* which
is defined as the *FDR*.

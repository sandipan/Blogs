Gaussian Process Regression With Python {.p-name}
=======================================

In this blog, we shall discuss on Gaussian Process Regression, the basic
concepts, how it can be implemented with python from scratch and…

* * * * *

### Gaussian Process Regression With Python {#b7fd .graf .graf--h3 .graf--leading .graf--title name="b7fd"}

#### Gaussian Process Regression and Bayesian Optimization {#8389 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="8389"}

In this blog, we shall discuss on Gaussian Process Regression, the basic
concepts, how it can be implemented with python from scratch and also
using the GPy library. Then we shall demonstrate an application of GPR
in Bayesian optimization with the GPyOpt library. The problems appeared
in this coursera course on [Bayesian methods for Machine
Learning](https://www.coursera.org/learn/bayesian-methods-in-machine-learning)
by UCSanDiego HSE and also in this [Machine learning
course](https://www.cs.ubc.ca/~nando/540-2013/) provided at UBC.

### Gaussian Process {#fdb9 .graf .graf--h3 .graf-after--p name="fdb9"}

A GP is a Gaussian distribution over functions, that takes two
parameters, namely the mean (m) and the kernel function K (to ensure
smoothness). In this article, we shall implement non-linear regression
with GP.

Given training data points (X,y) we want to learn a (non-linear)
function f:*R\^d* -\> R (here X is d-dimensional), s.t., y = f(x).

Then use the function f to predict the value of y for unseen data points
Xtest, along with the confidence of prediction.

The following figure describes the basic concepts of a GP and how it can
be used for regression. We need to use the conditional expectation and
variance formula (given the data) to compute the posterior distribution
for the GP.

![](https://cdn-images-1.medium.com/max/800/1*OqVN2ajQuteMg32Od59KNA.png)

Here, we shall first discuss on Gaussian Process Regression. Let’s
follow the steps below to get some intuition.

-   Generate 10 data points (these points will serve as training
    datapoints) with negligible noise (corresponds to noiseless GP
    regression). Use the following python function with default noise
    variance.

``` {#cc1a .graf .graf--pre .graf-after--li name="cc1a"}
import numpy as npdef generate_noisy_points(n=10, noise_variance=1e-6):    np.random.seed(777)    X = np.random.uniform(-3., 3., (n, 1))    y = np.sin(X) + np.random.randn(n, 1) * noise_variance**0.5    return X, y
```

-   Plot the points with the following code snippet.

``` {#2883 .graf .graf--pre .graf-after--li name="2883"}
import matplotlib.pylab as pltX, y = generate_noisy_points()plt.plot(X, y, 'x')plt.show()
```

![](https://cdn-images-1.medium.com/max/800/1*m31_rrUIaM_MautIQWgAQA.png)

-   Now, let’s implement the algorithm for GP regression, the one shown
    in the above figure. First lets generate 100 test data points.

``` {#4dc7 .graf .graf--pre .graf-after--li name="4dc7"}
Xtest, ytest = generate_noisy_points(100)Xtest.sort(axis=0)
```

-   Draw 10 function samples from the GP prior distribution using the
    following python code.

``` {#17d3 .graf .graf--pre .graf-after--li name="17d3"}
n = len(Xtest)K = kernel(Xtest, Xtest)L = np.linalg.cholesky(K + noise_var*np.eye(n))f_prior = np.dot(L, np.random.normal(size=(n, n_samples)))
```

-   The following animation shows the sample functions drawn from the GP
    prior dritibution.

![](https://cdn-images-1.medium.com/max/800/0*HFLmUNICU4KJFZdZ.gif)

Image by Author

-   Next, let’s compute the GP posterior distribution given the original
    (training) 10 data points, using the following python code snippet.

``` {#0fc2 .graf .graf--pre .graf-after--li name="0fc2"}
N, n = len(X), len(Xtest)K = kernel(X, X)L = np.linalg.cholesky(K + noise_var*np.eye(N))K_ = kernel(Xtest, Xtest)Lk = np.linalg.solve(L, kernel(X, Xtest))mu = np.dot(Lk.T, np.linalg.solve(L, y))L = np.linalg.cholesky(K_ + noise_var*np.eye(n) - np.dot(Lk.T, Lk))f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,n_samples)))
```

-   The following animation shows 10 function samples drawn from the GP
    posterior distribution. As expected, we get nearly zero uncertainty
    in the prediction of the points that are present in the training
    dataset and the variance increase as we move further from the
    points.

![](https://cdn-images-1.medium.com/max/800/0*P_bVG9xOc4bvw8Uj.gif)

Image by Author

-   The kernel function used here is RBF kernel, can be implemented with
    the following python code snippet.

``` {#c60e .graf .graf--pre .graf-after--li name="c60e"}
def kernel(x, y, l2):    sqdist = np.sum(x**2,1).reshape(-1,1) + \             np.sum(y**2,1) - 2*np.dot(x, y.T)    return np.exp(-.5 * (1/l2) * sqdist)
```

-   Now, let’s predict with the Gaussian Process Regression model, using
    the following python function:

``` {#1ca9 .graf .graf--pre .graf-after--li name="1ca9"}
def posterior(X, Xtest, l2=0.1, noise_var=1e-6):    # compute the mean at our test points.    N, n = len(X), len(Xtest)    K = kernel(X, X, l2)    L = np.linalg.cholesky(K + noise_var*np.eye(N))    Lk = np.linalg.solve(L, kernel(X, Xtest, l2))    mu = np.dot(Lk.T, np.linalg.solve(L, y))    # compute the variance at our test points.    K_ = kernel(Xtest, Xtest, l2)    sd = np.sqrt(np.diag(K_) - np.sum(Lk**2, axis=0))    return (mu, sd)
```

-   Use the above function to predict the mean and standard deviation at
    x=1.

``` {#f6a9 .graf .graf--pre .graf-after--li name="f6a9"}
mu, sd = posterior(X, np.array([[1]]))print(mu, sd)# [[0.85437633]] [0.26444361]
```

-   The following figure shows the predicted values along with the
    associated 3 s.d. confidence.

![](https://cdn-images-1.medium.com/max/800/1*9dLp6cr_Ggcrz5dO_nGDPA.png)

Image by Author

-   As can be seen, the highest confidence (corresponds to zero
    confidence interval) is again at the training data points. The blue
    curve represents the original function, the red one being the
    predicted function with GP and the red “+” points are the training
    data points.

**Noisy Gaussian Regression**

-   Now let’s increase the noise variance to implement the noisy version
    of GP.
-   The following animation shows how the predictions and the confidence
    interval change as noise variance is increased: the predictions
    become less and less uncertain, as expected.

![](https://cdn-images-1.medium.com/max/800/0*WuOehVk0TcJUPcNg.gif)

Image by Author

-   Next, let’s see how varying the RBF kernel parameter l changes the
    confidence interval, in the following animation.

![](https://cdn-images-1.medium.com/max/800/0*rMzwiDwQIYKqkRW1.gif)

Image by Author

### **Gaussian processes and Bayesian optimization** {#b9d4 .graf .graf--h3 .graf-after--figure name="b9d4"}

Now, let’s learn how to use [GPy](http://sheffieldml.github.io/GPy/) and
[GPyOpt](http://sheffieldml.github.io/GPyOpt/) libraries to deal with
gaussian processes. These libraries provide quite simple and inuitive
interfaces for training and inference, and we will try to get familiar
with them in a few tasks. The following figure shows the basic concepts
required for GP regression again.

![](https://cdn-images-1.medium.com/max/800/1*uGlhk0fiRnzVp1lU8r0vBQ.png)

#### Gaussian processes Regression with GPy ([*documentation*](http://pythonhosted.org/GPy/)) {#374a .graf .graf--h4 .graf-after--figure name="374a"}

-   Again, let’s start with a simple regression problem, for which we
    will try to fit a Gaussian Process with RBF kernel.
-   Create RBF kernel with variance sigma\_f and length-scale parameter
    l for 1D samples and compute value of the kernel between points,
    using the following code snippet.

``` {#e895 .graf .graf--pre .graf-after--li name="e895"}
import GPyimport GPyOptimport seaborn as snssigma_f, l = 1.5, 2kernel = GPy.kern.RBF(1, sigma_f, l)sns.heatmap(kernel.K(X, X))plt.show()
```

-   The following figure shows how the kernel heatmap looks like (we
    have 10 points in the training data, so the computed kernel is a
    10X10 matrix.

![](https://cdn-images-1.medium.com/max/800/0*lXJrYTiMJHrqSD_i.png)

-   Let’s fit a GP on the training data points. Use kernel from previous
    task. As shown in the code below, use GPy.models.GPRegression class
    to predict mean and vairance at position 𝑥=1, e.g.

``` {#3c73 .graf .graf--pre .graf-after--li name="3c73"}
X, y = generate_noisy_points(noise_variance=0.01)model = GPy.models.GPRegression(X,y,kernel) mean, variance = model.predict(np.array([[1]]))print(mean, variance)# 0.47161301004863576 1.1778512693257484
```

-   Let’s see the parameters of the model and plot the model

``` {#b1c7 .graf .graf--pre .graf-after--li name="b1c7"}
print(model)model.plot()plt.show()# Name : GP regression# Objective : 11.945995014694255# Number of Parameters : 3# Number of Optimization Parameters : 3# Updates : True# Parameters:# GP_regression.           |               value  |  constraints  |  # priors# rbf.variance             |  0.5884024388364221  |      +ve      |        # rbf.lengthscale          |   1.565659066396689  |      +ve      |        # Gaussian_noise.variance  |                 1.0  |      +ve      |
```

![](https://cdn-images-1.medium.com/max/800/1*O5T8MphLyFpIl-JTONeXiw.png)

-   Observe that the model didn’t fit the data quite well. Let’s try to
    fit kernel and noise parameters automatically.

``` {#4999 .graf .graf--pre .graf-after--li name="4999"}
model.optimize()print(model)# Name : GP regression# Objective : 0.691713117288967# Number of Parameters : 3# Number of Optimization Parameters : 3# Updates : True# Parameters:# GP_regression.           |                  value  |  constraints  |  # priors# rbf.variance             |     0.5088590289246014  |      +ve      |        # rbf.lengthscale          |     1.1019439281553658  |      +ve      |        # Gaussian_noise.variance  |  0.0030183424485056066  |      +ve      |
```

-   Now plot the model to obtain a figure like the following one.

![](https://cdn-images-1.medium.com/max/800/1*afDMMf8a9-tGZ3Q0_TqB3A.png)

-   As can be seen from the above figure, the process generates outputs
    just right.

#### Differentiating noise and signal with GP {#b374 .graf .graf--h4 .graf-after--li name="b374"}

-   Generate two datasets: sinusoid wihout noise (with the function
    generate\_points() and noise variance 0) and samples from gaussian
    noise (with the function generate\_noise() define below).

``` {#24e8 .graf .graf--pre .graf-after--li name="24e8"}
def generate_noise(n=10, noise_variance=0.01):    np.random.seed(777)    X = np.random.uniform(-3., 3., (n, 1))    y = np.random.randn(n, 1) * noise_variance**0.5    return X, y    X, y = generate_noise(noise_variance=10)
```

-   Optimize kernel parameters compute the optimal values of noise
    component for the noise.

``` {#5962 .graf .graf--pre .graf-after--li name="5962"}
model = GPy.models.GPRegression(X,y,kernel) model.optimize()print(model)# Name : GP regression# Objective : 26.895319516885678# Number of Parameters : 3# Number of Optimization Parameters : 3# Updates : True# Parameters:# GP_regression.           |              value  |  constraints  |  # priors# rbf.variance             |  4.326712527380182  |      +ve      |        # rbf.lengthscale          |  0.613701590417825  |      +ve      |        # Gaussian_noise.variance  |  9.006031601676087  |      +ve      |
```

-   Now optimize kernel parameters compute the optimal values of noise
    component for the signal without noise.

``` {#cd44 .graf .graf--pre .graf-after--li name="cd44"}
X, y = generate_noisy_points(noise_variance=0)model = GPy.models.GPRegression(X,y,kernel) model.optimize()print(model)# Name : GP regression# Objective : -22.60291145140328# Number of Parameters : 3# Number of Optimization Parameters : 3# Updates : True# Parameters:#  GP_regression.         |                  value  |  constraints |  #  priors#  rbf.variance           |      2.498516381000242  |      +ve     |        #  rbf.lengthscale        |     2.4529513517426444  |      +ve     |        #  Gaussian_noise.variance |  5.634716625480888e-16  |      +ve    |
```

-   As can be seen from above, the GP detects the noise correctly with a
    high value of **Gaussian\_noise.variance** output parameter.

### Sparse GP {#dbfa .graf .graf--h3 .graf-after--li name="dbfa"}

-   Now let’s consider the speed of GP. Let’s generate a dataset of 3000
    points and measure the time that is consumed for prediction of mean
    and variance for each point.
-   Then let’s try to use inducing inputs and find the optimal number of
    points according to quality-time tradeoff.
-   For the sparse model with inducing points, you should use
    GPy.models.SparseGPRegression class.
-   The number of inducing inputs can be set with parameter
    num\_inducing and optimize their positions and values
    with .optimize() call.
-   Let’s first create a dataset of 1000 points and fit GPRegression.
    Measure time for predicting mean and variance at position 𝑥=1.

``` {#505f .graf .graf--pre .graf-after--li name="505f"}
import timeX, y = generate_noisy_points(1000)start = time.time()model = GPy.models.GPRegression(X,y,kernel)mean, variance = model.predict(np.array([[1]]))time_gp = time.time()-startprint(mean, variance, time_gp)# [[0.84157973]] [[1.08298092e-06]] 7.353320360183716
```

-   Then fit SparseGPRegression with 10 inducing inputs and repeat the
    experiment. Let’s find speedup as a ratio between consumed time
    without and with inducing inputs.

``` {#3a72 .graf .graf--pre .graf-after--li name="3a72"}
start = time.time()model = GPy.models.SparseGPRegression(X,y,kernel,num_inducing=10) model.optimize()mean, variance = model.predict(np.array([[1]]))time_sgp = time.time()-startprint(mean, variance, time_sgp)# [[0.84159203]] [[1.11154212e-06]] 0.8615052700042725
```

-   As can be seen, there is a speedup of more than 8 with sparse GP
    using only the inducing points.

``` {#a928 .graf .graf--pre .graf-after--li name="a928"}
time_gp / time_sgp# 8.535432824627119
```

-   The model is shown in the next figure.

![](https://cdn-images-1.medium.com/max/800/1*qB-L5EZNRD3OxglCa-0bqw.png)

### Bayesian optimization {#4726 .graf .graf--h3 .graf-after--figure name="4726"}

-   Bayesian Optimization is used when there is no explicit objective
    function and it’s expensive to evaluate the objective function.
-   As shown in the next figure, a GP is used along with an acquisition
    (utility) function to choose the next point to sample, where it’s
    more likely to find the maximum value in an unknown objective
    function.
-   A GP is constructed from the points already sampled and the next
    point is sampled from the region where the GP posterior has higher
    mean (to exploit) and larger variance (to explore), which is
    determined by the maximum value of the acquisition function (which
    is a function of GP posterior mean and variance).
-   To choose the next point to be sampled, the above process is
    repeated.
-   The next couple of figures show the basic concepts of Bayesian
    optimization using GP, the algorithm, how it works, along with a few
    popular acquisition functions.

![](https://cdn-images-1.medium.com/max/800/1*tS1koXHkx84Baw2OAGENIA.png)

![](https://cdn-images-1.medium.com/max/800/1*DwmhejLfkfQUeob2xLuMoA.png)

#### Bayesian Optimization with GPyOpt ([documentation](http://pythonhosted.org/GPyOpt/)) {#ee71 .graf .graf--h4 .graf-after--figure name="ee71"}

**Paramerter Tuning for XGBoost Regressor**

-   Let’s now try to find optimal hyperparameters to XGBoost model using
    Bayesian optimization with GP, with the diabetes dataset (from
    sklearn) as input. Let’s first load the dataset with the following
    python code snippet:

``` {#19aa .graf .graf--pre .graf-after--li name="19aa"}
from sklearn import datasetsfrom xgboost import XGBRegressorfrom sklearn.model_selection import cross_val_score
```

``` {#3166 .graf .graf--pre .graf-after--pre name="3166"}
dataset = sklearn.datasets.load_diabetes()X = dataset['data']y = dataset['target']
```

-   We will use cross-validation score to estimate accuracy and our goal
    will be to tune: max\_depth, learning\_rate, n\_estimators
    parameters. First, we have to define optimization function and
    domains, as shown in the code below.

``` {#1135 .graf .graf--pre .graf-after--li name="1135"}
# Optimizer will try to find minimum, so let's add a "-" sign.def f(parameters):    parameters = parameters[0]    score = -cross_val_score(        XGBRegressor(learning_rate=parameters[0],                     max_depth=int(parameters[2]),                     n_estimators=int(parameters[3]),                     gamma=int(parameters[1]),                     min_child_weight = parameters[4]),         X, y, scoring='neg_root_mean_squared_error'    ).mean()    score = np.array(score)    return score    # Bounds (define continuous variables first, then discrete!)bounds = [    {'name': 'learning_rate',     'type': 'continuous',     'domain': (0, 1)},    {'name': 'gamma',     'type': 'continuous',     'domain': (0, 5)},    {'name': 'max_depth',     'type': 'discrete',     'domain': (1, 50)},    {'name': 'n_estimators',     'type': 'discrete',     'domain': (1, 300)},    {'name': 'min_child_weight',     'type': 'discrete',     'domain': (1, 10)}]
```

-   Let’s find the baseline RMSE with default XGBoost parameters is .
    Let’s see if we can do better.

``` {#8cd7 .graf .graf--pre .graf-after--li name="8cd7"}
baseline = -cross_val_score(    XGBRegressor(), X, y, scoring='neg_root_mean_squared_error').mean()baseline# 64.90693011829266
```

-   Now, run the Bayesian optimization with GPyOpt and plot convergence,
    as in the next code snippet:

``` {#f807 .graf .graf--pre .graf-after--li name="f807"}
optimizer = GPyOpt.methods.BayesianOptimization(    f=f, domain=bounds,    acquisition_type ='MPI',    acquisition_par = 0.1,    exact_eval=True)max_iter = 50max_time = 60optimizer.run_optimization(max_iter, max_time)optimizer.plot_convergence()
```

![](https://cdn-images-1.medium.com/max/800/1*fojKw5uzzI5A_4zzHyrmqg.png)

-   Extract the best values of the parameters and compute the RMSE /
    gain obtained with Bayesian Optimization, using the following code.

``` {#8bcb .graf .graf--pre .graf-after--li name="8bcb"}
optimizer.X[np.argmin(optimizer.Y)]# array([2.01515532e-01, 1.35401092e+00, 1.00000000e+00, # 3.00000000e+02, 1.00000000e+00])print('RMSE:', np.min(optimizer.Y),      'Gain:', baseline/np.min(optimizer.Y)*100)# RMSE: 57.6844355488563 Gain: 112.52069904249859
```

-   As can be seen, we were able to get 12% boost without tuning
    parameters by hand.

**Paramerter Tuning for SVR**

-   Now, let’s tune a Support Vector Regressor model with Bayesian
    Optimization and find the optimal values for three parameters: C,
    epsilon and gamma.
-   Let’s use range (1e-5, 1000) for C, (1e-5, 10) for epsilon and
    gamma.
-   Let’s use MPI as an acquisition function with weight 0.1.

``` {#0c28 .graf .graf--pre .graf-after--li name="0c28"}
from sklearn.svm import SVR
```

``` {#7569 .graf .graf--pre .graf-after--pre name="7569"}
# Bounds (define continuous variables first, then discrete!)bounds = [    {'name': 'C',     'type': 'continuous',     'domain': (1e-5, 1000)},
```

``` {#0d26 .graf .graf--pre .graf-after--pre name="0d26"}
    {'name': 'epsilon',     'type': 'continuous',     'domain': (1e-5, 10)},
```

``` {#0307 .graf .graf--pre .graf-after--pre name="0307"}
    {'name': 'gamma',     'type': 'continuous',     'domain': (1e-5, 10)}]
```

``` {#a49a .graf .graf--pre .graf-after--pre name="a49a"}
# Score. Optimizer will try to find minimum, so we will add a "-" sign.def f(parameters):    parameters = parameters[0]    score = -cross_val_score(        SVR(C = parameters[0],            epsilon = parameters[1],            gamma = parameters[2]),         X, y, scoring='neg_root_mean_squared_error'    ).mean()    score = np.array(score)    return score
```

``` {#f98c .graf .graf--pre .graf-after--pre name="f98c"}
optimizer = GPyOpt.methods.BayesianOptimization(    f=f, domain=bounds,    acquisition_type ='MPI',    acquisition_par = 0.1,    exact_eval=True)
```

``` {#15c8 .graf .graf--pre .graf-after--pre name="15c8"}
max_iter = 50*4max_time = 60*4optimizer.run_optimization(max_iter, max_time)
```

``` {#9722 .graf .graf--pre .graf-after--pre name="9722"}
baseline = -cross_val_score(    SVR(), X, y, scoring='neg_root_mean_squared_error').mean()print(baseline)# 70.44352670586173
```

``` {#e1dd .graf .graf--pre .graf-after--pre name="e1dd"}
print(optimizer.X[np.argmin(optimizer.Y)])# [126.64337652   8.49323372   8.59189135]
```

``` {#0858 .graf .graf--pre .graf-after--pre name="0858"}
print('RMSE:', np.min(optimizer.Y),      'Gain:', baseline/np.min(optimizer.Y)*100)# RMSE: 54.02576574389976 Gain: 130.38876124364006     
```

``` {#be18 .graf .graf--pre .graf-after--pre name="be18"}
best_epsilon = optimizer.X[np.argmin(optimizer.Y)][1] optimizer.plot_convergence()
```

![](https://cdn-images-1.medium.com/max/800/1*ZbSrg6DYtfFSi19M94H2QQ.png)

-   For the model above the boost in performance that was obtained after
    tuning hyperparameters was 30%.

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [November 2,
2020](https://medium.com/p/edd11bebdce8).

[Canonical
link](https://medium.com/@sandipan-dey/gaussian-process-regression-with-python-edd11bebdce8)

Exported from [Medium](https://medium.com) on January 8, 2021.

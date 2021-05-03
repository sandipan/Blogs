Named-Entity-Recognition (NER) on Twitter using Bi-directional LSTM with tensorflow in python {.p-name}
=============================================================================================

In this article, we shall discuss on how to use a recurrent neural
network to solve Named Entity Recognition (NER) problem. NER is a…

* * * * *

### Named-Entity-Recognition (NER) on Twitter Texts using Bi-directional LSTM with tensorflow in python {#23f3 .graf .graf--h3 .graf--leading .graf--title name="23f3"}

#### Named-Entity-Recognition with Bi-LSTM {#5503 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="5503"}

In this article, we shall discuss on how to use a recurrent neural
network to solve **Named Entity Recognition** (**NER**) problem. NER is
a common task in NLP systems. Given a text document, a NER system aims
at extracting the entities (e.g., persons, organizations, locations,
etc.) from the text. Here, a **BiLSTM** (bi-directional long short term
memory) will be used to recognize the named entities from Twitter texts.
This problem appears as an assignment in the Coursera course **Natural
Language Processing** by **National Research University Higher School of
Economics**, it’s a part of **Advanced Machine Learning**
Specialization. The problem is taken from the assignment.

### Named Entity Recognition Problem {#1c75 .graf .graf--h3 .graf-after--p name="1c75"}

Let’s try to understand by a few examples. The following figure shows
three examples of Twitter texts from the training corpus that we are
going to use, along with the *NER tags* corresponding to each of the
tokens from the texts.

Let’s say we want to extract

-   the *person names*
-   the *company names*
-   the *location names*
-   the *music artist names*
-   the *tv show names*

from the texts. Then a perfect *NER* model needs to generate the
following sequence of tags, as shown in the next figure.

![](https://cdn-images-1.medium.com/max/800/0*B08RHZMD4Az6uqkS)

Where *B-* and *I-* prefixes stand for the beginning and inside of the
entity, while *O* stands for out of tag or no tag. Markup with the
prefix scheme is called *BIO markup*. This markup is introduced for
distinguishing of consequent entities with similar types.

In this article we shall use a recurrent neural network (*RNN*),
particularly, a Bi-Directional Long Short-Term Memory Networks
(*Bi-LSTMs*), to predict the *NER* tags given the input text tokens. The
BiLSTM model is needed to b e trained first, so that it can be used for
prediction. The RNN architecture that is used for NER is shown below.

![](https://cdn-images-1.medium.com/max/800/0*3jXQCkLTgPKkaFwn)

The above figure is taken from
[https://stanford.edu/\~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks).

### Bi-directional LSTM {#a3b6 .graf .graf--h3 .graf-after--p name="a3b6"}

-   provides a universal approach for sequence tagging
-   several layers can be stacked + linear layers can be added on top
-   is trained by cross-entropy loss coming from each position

![](https://cdn-images-1.medium.com/max/800/0*qtQbLivgOWPPbWSl)

### Load the Twitter Named Entity Recognition corpus {#ed1d .graf .graf--h3 .graf-after--figure name="ed1d"}

The corpus to be used here contains tweets with NE tags. Every line of a
file contains a pair of a token (word/punctuation symbol) and a tag,
separated by a whitespace. Different tweets are separated by an empty
line.

The function *read\_data* reads a corpus from the *file\_path* and
returns two lists: one with tokens and one with the corresponding tags.
A user’s nickname in a tweet needs to be replaced by the
`<USR>`{.markup--code .markup--p-code} token and any URL with
the`<URL>`{.markup--code .markup--p-code} token (assuming that a URL and
a nickname are just strings which start with *http://* or *https://* in
case of URLs and a *@* symbol for nicknames).

![](https://cdn-images-1.medium.com/max/800/1*dfzWBSrJWBunRHX-GXvusg.png)

And now we can load three separate parts of the dataset:

-   *train* data for training the model;
-   *validation* data for evaluation and hyperparameters tuning;
-   *test* data for final evaluation of the model.

``` {#95ed .graf .graf--pre .graf-after--li name="95ed"}
train_tokens, train_tags = read_data('data/train.txt') validation_tokens, validation_tags =                         read_data('data/validation.txt') test_tokens, test_tags = read_data('data/test.txt')
```

We should always understand what kind of data you deal with. For this
purpose, let’s print the data running the following cell:

![](https://cdn-images-1.medium.com/max/800/1*u7Z7zTB-mTkp74aBmWbVZw.png)

### Prepare dictionaries {#2192 .graf .graf--h3 .graf-after--figure name="2192"}

To train a neural network, we will use two mappings:

-   {token} → {token id}: address the row in *embeddings* matrix for the
    current token;
-   {tag} → {tag id}: *one-hot-encoding* ground truth probability
    distribution vectors for computing the loss at the output of the
    network.

Now let’s implement the function *build\_dict* which will return {token
or tag} → {index} and vice versa.

![](https://cdn-images-1.medium.com/max/800/1*pT_9FmTNsJ5UCuHYaN2FcQ.png)

After implementing the function *build\_dict* we can create dictionaries
for tokens and tags. Special tokens in our case will be:

-   `<UNK>`{.markup--code .markup--li-code} token for out of vocabulary
    tokens;
-   `<PAD>`{.markup--code .markup--li-code} token for padding sentence
    to the same length when we create batches of sentences.

``` {#aefd .graf .graf--pre .graf-after--li name="aefd"}
special_tokens = ['<UNK>', '<PAD>']special_tags = ['O']# Create dictionariestoken2idx, idx2token = build_dict(train_tokens + validation_tokens,                                                     special_tokens)tag2idx, idx2tag = build_dict(train_tags, special_tags)
```

We can see from the below output that there are 21 tags for the named
entities in the corpus.

![](https://cdn-images-1.medium.com/max/800/1*J8NWN5vfFis6lzAECn5SIg.png)

The next additional functions will be helpful for creating the mapping
between tokens and ids for a sentence.

![](https://cdn-images-1.medium.com/max/800/1*otln6M1mEK0Frsa_m2TBRg.png)

### Generate batches {#9461 .graf .graf--h3 .graf-after--figure name="9461"}

Neural Networks are usually trained with batches. It means that weight
updates of the network are based on several sequences at every single
time. The tricky part is that all sequences within a batch need to have
the same length. So we will pad them with a special
`<PAD>`{.markup--code .markup--p-code} token. It is also a good practice
to provide RNN with sequence lengths, so it can skip computations for
padding parts. The following python generator function
*batches\_generator* can be used to generate batches.

![](https://cdn-images-1.medium.com/max/800/1*ihPaVdsthfAuo7RdrZ_9-g.png)

### Build a recurrent neural network {#ab0c .graf .graf--h3 .graf-after--figure name="ab0c"}

This is the most important part where we shall specify the network
architecture based on TensorFlow building blocks. We shall create an
LSTM network which will produce probability distribution over tags for
each token in a sentence. To take into account both right and left
contexts of the token, we will use Bi-Directional LSTM (Bi-LSTM). Dense
layer will be used on top to perform tag classification.

``` {#a9bb .graf .graf--pre .graf-after--p name="a9bb"}
import tensorflow as tfimport numpy as nptf.compat.v1.disable_eager_execution()
```

``` {#a762 .graf .graf--pre .graf-after--pre name="a762"}
class BiLSTMModel(): pass
```

First, we need to create *placeholders* to specify what data we are
going to feed into the network during the execution time. For this task
we will need the following placeholders:

-   *input\_batch* — sequences of words (the shape equals to
    [batch\_size, sequence\_len]);
-   *ground\_truth\_tags* — sequences of tags (the shape equals to
    [batch\_size, sequence\_len]);
-   *lengths* — lengths of not padded sequences (the shape equals to
    [batch\_size]);
-   *dropout\_ph* — dropout keep probability; this placeholder has a
    predefined value 1;
-   *learning\_rate\_ph* — learning rate; we need this placeholder
    because we want to change the value during training.

It could be noticed that we use *None* in the shapes in the declaration,
which means that data of any size can be fed.

Now, let us specify the layers of the neural network. First, we need to
perform some preparatory steps:

-   Create embeddings matrix with
    [tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable).
    Specify its name (*embeddings\_matrix*), type (*tf.float32*), and
    initialize with random values.
-   Create forward and backward LSTM cells. TensorFlow provides a number
    of RNN cells ready for you. We suggest that you use *LSTMCell*, but
    we can also experiment with other types, e.g. GRU cells.
    [This](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    blogpost could be interesting if you want to learn more about the
    differences.
-   Wrap cells with
    [DropoutWrapper](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper).
    Dropout is an important regularization technique for neural
    networks. Specify all keep probabilities using the dropout
    placeholder that we created before.

After that, we can build the computation graph that transforms an
input\_batch:

-   [Look
    up](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)
    embeddings for an *input\_batch* in the prepared
    *embedding\_matrix*.
-   Pass the embeddings through [Bidirectional Dynamic
    RNN](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn)
    with the specified forward and backward cells. Use the lengths
    placeholder here to avoid computations for padding tokens inside the
    RNN.
-   Create a dense layer on top. Its output will be used directly in
    loss function.

![](https://cdn-images-1.medium.com/max/800/1*kM7M5WiIv9bI3ygvlWKDog.png)

To compute the actual predictions of the neural network, we need to
apply
[softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) to
the last layer and find the most probable tags with
[argmax](https://www.tensorflow.org/api_docs/python/tf/argmax).

![](https://cdn-images-1.medium.com/max/800/1*Wyc8mUproZhBAygsOX6EQA.png)

During training we do not need predictions of the network, but we need a
loss function. We will use [cross-entropy
loss](http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy),
efficiently implemented in TF as [cross entropy with
logits](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2).
Note that it should be applied to logits of the model (not to softmax
probabilities!). Also note, that we do not want to take into account
loss terms coming from `<PAD>`{.markup--code .markup--p-code} tokens. So
we need to mask them out, before computing
[mean](https://www.tensorflow.org/api_docs/python/tf/reduce_mean).

![](https://cdn-images-1.medium.com/max/800/1*kHnb1ixPrW_22XCQJNAYbw.png)

The last thing to specify is how we want to optimize the loss. Let’s use
[Adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
optimizer with a learning rate from the corresponding placeholder. We
shall also need to apply clipping to eliminate exploding gradients. It
can be easily done with the function
[clip\_by\_norm](https://www.tensorflow.org/api_docs/python/tf/clip_by_norm).

![](https://cdn-images-1.medium.com/max/800/1*0djsvHG3LWRMRJMWi3Dnog.png)

Now we have specified all the parts of your network. It can be noticed,
that we didn’t deal with any real data yet, so what we have written is
just recipes on how the network should function. Now we will put them to
the constructor of our Bi-LSTM class to use it in the next section.

![](https://cdn-images-1.medium.com/max/800/1*xICzDfZHFbTT0x6SM_VGzg.png)

### Train the network and predict tags {#3ced .graf .graf--h3 .graf-after--figure name="3ced"}

[Session.run](https://www.tensorflow.org/api_docs/python/tf/Session#run)
is a point which initiates computations in the graph that we have
defined. To train the network, we need to compute *self.train\_op*,
which was declared in *perform\_optimization*. To predict tags, we just
need to compute *self.predictions*. Anyway, we need to feed actual data
through the placeholders that we defined before.

![](https://cdn-images-1.medium.com/max/800/1*DgGS6eAxZW45L4COw12eXg.png)

Let’s implement the function *predict\_for\_batch* by initializing
*feed\_dict* with input *x\_batch* and *lengths* and running the
*session* for *self.predictions*.

![](https://cdn-images-1.medium.com/max/800/1*Pcx2dyTwU-Cm1ZRy5yEOow.png)

We finished with necessary methods of our BiLSTMModel model and almost
ready to start experimenting.

### Evaluation {#745f .graf .graf--h3 .graf-after--p name="745f"}

To simplify the evaluation process let’s use the two functions:

-   *predict\_tags*: uses a model to get predictions and transforms
    indices to tokens and tags;
-   *eval\_conll*: calculates precision, recall and F1 for the results.
-   The function precision\_recall\_f1() is implemented / used to
    compute these metrics with training and validation data.

![](https://cdn-images-1.medium.com/max/800/1*7iL4EfN-gz3BGDXD1gKBKw.png)

### Run the experiment {#28dd .graf .graf--h3 .graf-after--figure name="28dd"}

Create *BiLSTMModel* model with the following parameters:

-   *vocabulary\_size* — number of tokens;
-   *n\_tags* — number of tags;
-   *embedding\_dim* — dimension of embeddings, recommended value: 200;
-   *n\_hidden\_rnn* — size of hidden layers for RNN, recommended value:
    200;
-   *PAD\_index* — an index of the padding token (`<PAD>`{.markup--code
    .markup--li-code}).

Set hyperparameters. We might want to start with the following
recommended values:

-   *batch\_size*: 32;
-   4 epochs;
-   starting value of *learning\_rate*: 0.005
-   *learning\_rate\_decay*: a square root of 2;
-   *dropout\_keep\_probability*: try several values: 0.1, 0.5, 0.9.

However, we can conduct more experiments to tune hyperparameters, to
obtain higher accuracy on the held-out validation dataset.

![](https://cdn-images-1.medium.com/max/800/1*WVyZlMgiY07gOVBRrQ7rxw.png)

Finally, we are ready to run the training!

![](https://cdn-images-1.medium.com/max/800/1*eKwNW2cU6FkUMmCM1deSWQ.png)

The following figure shows how the precision / recall and F1 score
changes on the training and validation datasets while training, for each
of the entities.

![](https://cdn-images-1.medium.com/max/800/0*VOHWZzFEwos29YwY)

Now let us see full quality reports for the final model on train,
validation, and test sets.

![](https://cdn-images-1.medium.com/max/800/1*0OxeaD_QvSfyKjJ2pANU7Q.png)

The following animation shows the softmax probabilities corresponding to
the NER tags predicted by the BiLSTM model on the tweets from the test
corpus.

![](https://cdn-images-1.medium.com/max/800/0*Tyn9gyXkUkNudFIV)

Also, visualized in another way, the following two animations show the
tweet text, the probabilities for different NE tags, predicted by the
BiLSTM NER model and the ground-truth tag.

![](https://cdn-images-1.medium.com/max/800/1*yX0mDbwagyVN6sxKw6NRGA.gif)

### Conclusions {#3137 .graf .graf--h3 .graf-after--figure name="3137"}

Nowadays, Bi-LSTM is one of the state of the art approaches for solving
NER problem and it outperforms other classical methods. Despite the fact
that we used small training corpora (in comparison with usual sizes of
corpora in Deep Learning), our results are quite good. In addition, in
this task there are many possible named entities and for some of them we
have only several dozens of trainig examples, which is definately small.
However, the implemented model outperforms classical CRFs for this task.
Even better results could be obtained by some combinations of several
types of methods, e.g. (refer to
[this](https://arxiv.org/abs/1603.01354) paper).

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [September 8,
2020](https://medium.com/p/8b85cccc649c).

[Canonical
link](https://medium.com/@sandipan-dey/named-entity-recognition-ner-on-twitter-using-bi-directional-lstm-with-tensorflow-in-python-8b85cccc649c)

Exported from [Medium](https://medium.com) on January 8, 2021.

Solving a few AI problems with Python: Part 2 {.p-name}
=============================================

This blog is the second part of this blog, here we shall continue
discussing on a few more problems in artificial intelligence and their…

* * * * *

### Solving a few AI problems with Python: Part 2 {#ad94 .graf .graf--h3 .graf--leading .graf--title name="ad94"}

#### AI Problems on Uncertainty, Learning and Language {#ede3 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="ede3"}

This blog is the second part of [this
blog](https://sandipan-dey.medium.com/solving-a-few-ai-problems-with-python-9c5c75f65f3d),
here we shall continue discussing on a few more problems in artificial
intelligence and their python implementations. The problems discussed
here appeared as programming assignments in the edX course****[**CS50’s
Introduction to Artificial Intelligence with Python (HarvardX:CS50
AI)**](https://cs50.harvard.edu/ai/2020/). The problem statements are
taken from the course itself.

### Heredity {#7c0e .graf .graf--h3 .graf-after--p name="7c0e"}

Write an AI agent to assess the likelihood that a person will have a
particular genetic trait.

-   Mutated versions of the [GJB2
    gene](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1285178/) are one
    of the leading causes of hearing impairment in newborns.
-   Each person carries two versions of the gene, so each person has the
    potential to possess either 0, 1, or 2 copies of the hearing
    impairment version GJB2.
-   Unless a person undergoes genetic testing, though, it’s not so easy
    to know how many copies of mutated GJB2 a person has.
-   This is some “hidden state”: information that has an effect that we
    can observe (hearing impairment), but that we don’t necessarily
    directly know.
-   After all, some people might have 1 or 2 copies of mutated GJB2 but
    not exhibit hearing impairment, while others might have no copies of
    mutated GJB2 yet still exhibit hearing impairment.
-   Every child inherits one copy of the GJB2 gene from each of their
    parents. If a parent has two copies of the mutated gene, then they
    will pass the mutated gene on to the child; if a parent has no
    copies of the mutated gene, then they will not pass the mutated gene
    on to the child; and if a parent has one copy of the mutated gene,
    then the gene is passed on to the child with probability 0.5.
-   After a gene is passed on, though, it has some probability of
    undergoing additional mutation: changing from a version of the gene
    that causes hearing impairment to a version that doesn’t, or vice
    versa.
-   We can attempt to model all of these relationships by forming a
    Bayesian Network of all the relevant variables, as in the one below,
    which considers a family of two parents and a single child.

![](https://cdn-images-1.medium.com/max/800/0*IMAx__LhnAfJF7Z1.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/2/heredity/)

-   Each person in the family has a Gene random variable representing
    how many copies of a particular gene (e.g., the hearing impairment
    version of GJB2) a person has: a value that is 0, 1, or 2.
-   Each person in the family also has a Trait random variable, which is
    yes or no depending on whether that person expresses a trait (e.g.,
    hearing impairment) based on that gene.
-   There’s an arrow from each person’s Gene variable to their Trait
    variable to encode the idea that a person’s genes affect the
    probability that they have a particular trait.
-   Meanwhile, there’s also an arrow from both the mother and father’s
    Gene random variable to their child’s Gene random variable: the
    child’s genes are dependent on the genes of their parents.
-   The task in this problem is to use this model to make inferences
    about a population. Given information about people, who their
    parents are, and whether they have a particular observable trait
    (e.g. hearing loss) caused by a given gene, the AI agent will infer
    the probability distribution for each person’s genes, as well as the
    probability distribution for whether any person will exhibit the
    trait in question.
-   The next figure shows the concepts required for running inference on
    a Bayes Network

![](https://cdn-images-1.medium.com/max/800/1*4TNFZfCSji7aXxqQ5Ol9wg.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

-   Recall that to compute a joint probability of multiple events, you
    can do so by multiplying those probabilities together. But remember
    that for any child, the probability of them having a certain number
    of genes is conditional on what genes their parents have.
-   The following figure shows the given data files

![](https://cdn-images-1.medium.com/max/800/0*NlJrYHqiH-kQYBpw.png)

Image created by Author

-   Implement a function joint\_probability() that should take as input
    a dictionary of people, along with data about who has how many
    copies of each of the genes, and who exhibits the trait. The
    function should return the joint probability of all of those events
    taking place.
-   Implement a function update() that adds a new joint distribution
    probability to the existing probability distributions in
    probabilities .
-   Implement the function normalize() that updates a dictionary of
    probabilities such that each probability distribution is normalized.
-   PROBS is a dictionary containing a number of constants representing
    probabilities of various different events. All of these events have
    to do with how many copies of a particular gene a person has , and
    whether a person exhibits a particular trait based on that gene.
-   PROBS[“gene”] represents the unconditional probability distribution
    over the gene (i.e., the probability if we know nothing about that
    person’s parents).
-   PROBS[“trait”] represents the conditional probability that a person
    exhibits a trait.
-   PROBS[“mutation”] is the probability that a gene mutates from being
    the gene in question to not being that gene, and vice versa.

The following python code snippet shows how to compute the joint
probabilities:

``` {#7aba .graf .graf--pre .graf-after--p name="7aba"}
def joint_probability(people, one_gene, two_genes, have_trait):    """    Compute and return a joint probability.    The probability returned should be the probability that    * everyone in set `one_gene` has one copy of the gene, and    * everyone in set `two_genes` has two copies of the gene, and    * everyone not in `one_gene` / `two_gene` doesn't have the gene    * everyone in set `have_trait` has the trait, and    * everyone not in set` have_trait` does not have the trait.    """    prob_gene = PROBS['gene']    prob_trait_gene = PROBS['trait']    prob_mutation = PROBS['mutation']    prob_gene_pgenes = { # CPT        2: {            0: {                 0: (1-prob_mutation)*prob_mutation,                 1: (1-prob_mutation)*(1-prob_mutation) + \                                   prob_mutation*prob_mutation,                 2: prob_mutation*(1-prob_mutation)            },            1: {                 0: prob_mutation/2,                 1: 0.5,                 2: (1-prob_mutation)/2            },            2: {                 0: prob_mutation*prob_mutation,                 1: 2*(1-prob_mutation)*prob_mutation,                  2: (1-prob_mutation)*(1-prob_mutation)            }        },   # compute the probabilities for the other cases...  }       all_probs = {}    for name in people:        has_trait = name in have_trait        num_gene = 1 if name in one_gene else 2 if name in two_genes                                                                    else 0        prob = prob_trait_gene[num_gene][has_trait]* \                                        prob_gene[num_gene]         if people[name]['mother'] != None:           mother, father = people[name]['mother'], \                            people[name]['father']           num_gene_mother = 1 if mother in one_gene else \                             2 if mother in two_genes else 0           num_gene_father = 1 if father in one_gene else 2                                if father in two_genes else 0           prob = prob_trait_gene[num_gene][has_trait] * \                prob_gene_pgenes[num_gene_mother][num_gene_father] \                                            [num_gene]                      all_probs[name] = prob    probability = np.prod(list(all_probs.values()))    return probability
```

The following animations show how the CPTs (conditional probability
tables) are computed for the Bayesian Networks corresponding to the
first couple of families, respectively.

![](https://cdn-images-1.medium.com/max/800/1*MlqrZx6_3TIJZ0UIv_bOYw.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/1*3e2HsbL7-pqZYyaQ0CkXxQ.gif)

Image by Author

The following figure shows the conditional probabilities (computed with
simulation) that children does / doesn’t have the gene given the parents
does / doesn’t have the gene.

![](https://cdn-images-1.medium.com/max/800/1*wZ5_-QhoXsibsN_KbPYJMA.png)

Image by Author

### Crossword {#149d .graf .graf--h3 .graf-after--figure name="149d"}

Write an AI agent to generate crossword puzzles.

-   Given the structure of a crossword puzzle (i.e., which squares of
    the grid are meant to be filled in with a letter), and a list of
    words to use, the problem becomes one of choosing which words should
    go in each vertical or horizontal sequence of squares.
-   We can model this sort of problem as a constraint satisfaction
    problem.
-   Each sequence of squares is one variable, for which we need to
    decide on its value(which word in the domain of possible words will
    􀃥ll in that sequence). Consider the following crossword puzzle
    structure.

![](https://cdn-images-1.medium.com/max/800/0*Umm3YKcgf7vIOFKW.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/3/crossword/)

-   In this structure, we have four variables, representing the four
    words we need to 􀃥ll into this crossword puzzle (each indicated by a
    number in the above image). Each variable is defined by four values:
    the row it begins on (its i value), the column it begins on (its j
    value), the direction of the word (either down or across), and the
    length of the word.
-   Variable 1, for example, would be a variable represented by a row of
    1 (assuming 0 indexed counting from the top), a column of 1 (also
    assuming 0 indexed counting from the left), a direction of across,
    and a length of 4.
-   As with many constraint satisfaction problems, these variables have
    both unary and binary constraints. The unary constraint on a
    variable is given by its length.
-   For Variable 1, for instance, the value BYTE would satisfy the unary
    constraint, but the value BIT would not (it has the wrong number of
    letters).
-   Any values that don’t satisfy a variable’s unary constraints can
    therefore be removed from the variable’s domain immediately.
-   The binary constraints on a variable are given by its overlap with
    neighboring variables. Variable 1 has a single neighbor: Variable 2.
    Variable 2 has two neighbors: Variable 1 and Variable 3.
-   For each pair of neighboring variables, those variables share an
    overlap: a single square that is common to them both.
-   We can represent that overlap as the character index in each
    variable’s word that must be the same character.
-   For example, the overlap between Variable 1 and Variable 2 might be
    represented as the pair (1, 0), meaning that Variable 1’s character
    at index 1 necessarily must be the same as Variable 2’s character at
    index 0 (assuming 0-indexing, again).
-   The overlap between Variable 2 and Variable 3 would therefore be
    represented as the pair (3, 1): character 3 of Variable 2’s value
    must be the same as character 1 of Variable 3’s value.
-   For this problem, we’ll add the additional constraint that all words
    must be different: the same word should not be repeated multiple
    times in the puzzle.
-   The challenge ahead, then, is write a program to 􀃥find a satisfying
    assignment: a different word (from a given vocabulary list) for each
    variable such that all of the unary and binary constraints are met.
-   The following figure shows the theory required to solve the problem:

![](https://cdn-images-1.medium.com/max/800/1*GE1XxMoVGoxTrjp_C6hlqA.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

-   Implement a function enforce\_node\_consistency() that should update
    the domains such that each variable is node consistent.
-   Implement a function ac3() that should, using the AC3 algorithm,
    enforce arc consistency on the problem. Recall that arc consistency
    is achieved when all the values in each variable’s domain satisfy
    that variable’s binary constraints.
-   Recall that the AC3 algorithm maintains a queue of arcs to process.
    This function takes an optional argument called arcs, representing
    an initial list of arcs to process. If arcs is None, the function
    should start with an initial queue of all of the arcs in the
    problem. Otherwise, the algorithm should begin with an initial queue
    of only the arcs that are in the list arcs.
-   Implement a function consistent() that should check to see if a
    given assignment is consistent and a function assignment\_complete()
    that should check if a given assignment is complete.
-   Implement a function select\_unassigned\_variable() that should
    return a single variable in the crossword puzzle that is not yet
    assigned by assignment, according to the minimum remaining value
    heuristic and then the degree heuristic. Also, it should return the
    variable with the fewest number of remaining values in its domain.
-   Implement a function backtrack() that should accept a partial
    assignment as input and, using backtracking search, return a
    complete satisfactory assignment of variables to values if it is
    possible to do so.

The following python code snippet shows couple of the above function
implementations:

``` {#e70f .graf .graf--pre .graf-after--p name="e70f"}
def select_unassigned_variable(self, assignment):    """    Return an unassigned variable not already part of `assignment`.    Choose the variable with the minimum number of remaining values    in its domain. If there is a tie, choose the variable with the     highest degree. If there is a tie, any of the tied variables     are acceptable return values.    """    unassigned_vars = list(self.crossword.variables - \                               set(assignment.keys()))    mrv, svar = float('inf'), None    for var in unassigned_vars:        if len(self.domains[var]) < mrv:           mrv, svar =  len(self.domains[var]), var        elif len(self.domains[var]) == mrv:           if self.crossword.neighbors(var) >               self.crossword.neighbors(svar):              svar = var     return svar    
```

``` {#c913 .graf .graf--pre .graf-after--pre name="c913"}
def backtrack(self, assignment):    """    Using Backtracking Search, take as input a partial assignment     for the crossword and return a complete assignment if possible     to do so.    `assignment` is a mapping from variables (keys) to words     (values).    If no assignment is possible, return None.    """    # Check if assignment is complete    if len(assignment) == len(self.crossword.variables):    return assignment
```

``` {#b42e .graf .graf--pre .graf-after--pre name="b42e"}
    # Try a new variable    var = self.select_unassigned_variable(assignment)    for value in self.domains[var]:        new_assignment = assignment.copy()        new_assignment[var] = value        if self.consistent(new_assignment):           result = self.backtrack(new_assignment)           if result is not None:              return result    return None
```

The following animation shows how the backtracking algorithm with AC3
generates the crossword with the given structure.

``` {#b5bc .graf .graf--pre .graf-after--p name="b5bc"}
######_____##__##_____##_##__##_##_#___##_
```

![](https://cdn-images-1.medium.com/max/800/1*eYGNhI_IIZadH73Bw5Aykw.gif)

— — — — — — — — Image by
Author — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —

Now let’s compute the speedup obtained with backtracking with and
without AC3 (on average), i.e., compare the time to generate a crossword
puzzle (with the above structure), averaged over 1000 runs (each time
the puzzle generated may be different though). The next figure shows the
result of the runtime experiment and that backtracking with AC3 is way
faster on average over backtracking implementation without AC3.

![](https://cdn-images-1.medium.com/max/800/1*bqStwqOs-ZL0sBJAPQok3Q.png)

Image by Author

### Shopping {#c0b2 .graf .graf--h3 .graf-after--figure name="c0b2"}

Write an AI agent to predict whether online shopping customers will
complete a purchase.

-   When users are shopping online, not all will end up purchasing
    something. Most visitors to an online shopping website, in fact,
    likely don’t end up going through with a purchase during that web
    browsing session.
-   It might be useful, though, for a shopping website to be able to
    predict whether a user intends to make a purchase or not: perhaps
    displaying different content to the user, like showing the user a
    discount offer if the website believes the user isn’t planning to
    complete the purchase.
-   How could a website determine a user’s purchasing intent? That’s
    where machine learning will come in.
-   The task in this problem is to build a nearest-neighbor classifier
    to solve this problem. Given information about a user — how many
    pages they’ve visited, whether they’re shopping on a weekend, what
    web browser they’re using, etc. — the classifier will predict
    whether or not the user will make a purchase.
-   The classifier won’t be perfectly accurate — but it should be better
    than guessing randomly.
-   To train the classifier a dataset from a shopping website from about
    12,000 users sessions is provided.
-   How do we measure the accuracy of a system like this? If we have a
    testing data set, we could run our classifier and compute what
    proportion of the time we correctly classify the user’s intent.
-   This would give us a single accuracy percentage. But that number
    might be a little misleading. Imagine, for example, if about 15% of
    all users end up going through with a purchase. A classifier that
    always predicted that the user would not go through with a purchase,
    then, we would measure as being 85% accurate: the only users it
    classifies incorrectly are the 15% of users who do go through with a
    purchase. And while 85% accuracy sounds pretty good, that doesn’t
    seem like a very useful classifier.
-   Instead, we’ll measure two values: sensitivity (also known as the
    “true positive rate”) and specificity (also known as the “true
    negative rate”).
-   Sensitivity refers to the proportion of positive examples that were
    correctly identified: in other words, the proportion of users who
    did go through with a purchase who were correctly identified.
-   Specificity refers to the proportion of negative examples that were
    correctly identified: in this case, the proportion of users who did
    not go through with a purchase who were correctly identified.
-   So our “always guess no” classifier from before would have perfect
    specificity (1.0) but no sensitivity (0.0). Our goal is to build a
    classifier that performs reasonably on both metrics.
-   First few rows from the csv dataset are shown in the following
    figure:

![](https://cdn-images-1.medium.com/max/800/1*FPAZTLlnoKay-RBE267eRA.png)

Image created by Author

-   There are about 12,000 user sessions represented in this
    spreadsheet: represented as one row for each user session.
-   The first six columns measure the different types of pages users
    have visited in the session: the Administrative , Informational ,
    and ProductRelated columns measure how many of those types of pages
    the user visited, and their corresponding \_Duration columns measure
    how much time the user spent on any of those pages.
-   The BounceRates , ExitRates , and PageValues columns measure
    information from Google Analytics about the page the user visited.
    SpecialDay is a value that measures how closer the date of the
    user’s session is to a special day (like Valentine’s Day or Mother’s
    Day).
-   Month is an abbreviation of the month the user visited.
    OperatingSystems , Browser , Region, and TrafficType are all
    integers describing information about the user themself.
-   VisitorType will take on the value Returning\_Visitor for returning
    visitors and New\_Visitor for new visitors to the site.
-   Weekend is TRUE or FALSE depending on whether or not the user is
    visiting on a weekend.
-   Perhaps the most important column, though, is the last one: the
    Revenue column. This is the column that indicates whether the user
    ultimately made a purchase or not: TRUE if they did, FALSE if they
    didn’t. This is the column that we’d like to learn to predict (the
    “label”), based on the values for all of the other columns (the
    “evidence”).
-   Implement a function load\_data() that should accept a CSV filename
    as its argument, open that file, and return a tuple (evidence,
    labels) . evidence should be a list of all of the evidence for each
    of the data points, and labels should be a list of all of the labels
    for each data point.
-   Implement a function train\_model() function should accept a list of
    evidence and a list of labels, and return a scikit-learn
    nearest-neighbor classifier (a k-nearest-neighbor classifier where k
    = 1) 􀃥fitted on that training data.
-   Relevant concepts in **supervised machine learning**
    (classification) needed for this problem is described in the
    following figure.

![](https://cdn-images-1.medium.com/max/800/1*l8EsymqBjugRIkTPeCiqKQ.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

Use the following python code snippet to divide the dataset into two
parts: the first one to train the 1-NN model on and the second one being
a held-out test dataset to evaluate the model using the performance
metrics.

``` {#b0bc .graf .graf--pre .graf-after--p name="b0bc"}
from sklearn.model_selection import train_test_splitfrom sklearn.neighbors import KNeighborsClassifierTEST_SIZE = 0.4
```

``` {#1889 .graf .graf--pre .graf-after--pre name="1889"}
# Load data from spreadsheet and split into train and test setsevidence, labels = load_data('shopping.csv')X_train, X_test, y_train, y_test = train_test_split(   evidence, labels, test_size=TEST_SIZE)# Train model and make predictionsmodel = train_model(X_train, y_train)predictions = model.predict(X_test)sensitivity, specificity = evaluate(y_test, predictions)
```

``` {#0210 .graf .graf--pre .graf-after--pre name="0210"}
def train_model(evidence, labels):    """    Given a list of evidence lists and a list of labels, return a    fitted k-nearest neighbor model (k=1) trained on the data.    """    model = KNeighborsClassifier(n_neighbors=1)    model.fit(evidence, labels)    return model
```

With 60–40 validation with a sample run following result is obtained, by
training the model on 60% training data and use the model to predict on
the 40% held-out test dataset, and then comparing the test predictions
with the ground truth:

``` {#f26d .graf .graf--pre .graf-after--p name="f26d"}
Correct: 3881Incorrect: 1051True Positive Rate: 31.14%True Negative Rate: 87.37%
```

Now, let’s repeat the experiment for 25 times (with different random
seeds for train\_test\_split()) and let’s plot the sensitivity against
specificity for every random partition, along with color-coded accuracy,
to obtain a plot like the following one.

![](https://cdn-images-1.medium.com/max/800/1*XtLH7tnmEC1KXexwswzXzg.png)

Image by Author

with the mean performance metrics reported as follows (to reduce the
variability in the result):

``` {#c77a .graf .graf--pre .graf-after--p name="c77a"}
Sensitivity: 0.3044799546457174 Specificity: 0.8747533289505658 Accuracy: 0.7867234387672343
```

As can be seen from the above result, the specificity if quite good, but
sensitivity obtained is quite poor.

Let’s now plot the decision boundary obtained with 1-NN classifier for
different 60–40 train-test validation datasets created, with the data
projected across two principal components, the output is shown in the
following animation (here the red points represent the cases where
purchase happened and the blue points represent the cases where it did
not).

![](https://cdn-images-1.medium.com/max/800/0*B1XV4_ApmRRa0eCs.gif)

Image by Author

### Nim {#93ec .graf .graf--h3 .graf-after--figure name="93ec"}

Implement an AI agent that teaches itself to play Nim through
**reinforcement learning**.

-   Recall that in the game Nim, we begin with some number of piles,
    each with some number of objects. Players take turns: on a player’s
    turn, the player removes any non-negative number of objects from any
    one non-empty pile. Whoever removes the last object loses.
-   There’s some simple strategy you might imagine for this game: if
    there’s only one pile and three objects left in it, and it’s your
    turn, your best bet is to remove two of those objects, leaving your
    opponent with the third and final object to remove.
-   But if there are more piles, the strategy gets considerably more
    complicated. In this problem, we’ll build an AI agent to learn the
    strategy for this game through reinforcement learning. By playing
    against itself repeatedly and learning from experience, eventually
    our AI agent will learn which actions to take and which actions to
    avoid.
-   In particular, we’ll use Q-learning for this project. Recall that in
    Q-learning, we try to learn a reward value (a number) for every
    (state,action) pair. An action that loses the game will have a
    reward of -1, an action that results in the other player losing the
    game will have a reward of 1, and an action that results in the game
    continuing has an immediate reward of 0, but will also have some
    future reward.
-   How will we represent the states and actions inside of a Python
    program? A “state” of the Nim game is just the current size of all
    of the piles. A state, for example, might be [1, 1, 3, 5] ,
    representing the state with 1 object in pile 0, 1 object in pile 1,
    3 objects in pile 2, and 5 objects in pile 3. An “action” in the Nim
    game will be a pair of integers (i, j) , representing the action of
    taking j objects from pile i . So the action (3, 5) represents the
    action “from pile 3, take away 5 objects.” Applying that action to
    the state [1, 1, 3, 5] would result in the new state [1, 1, 3, 0]
    (the same state, but with pile 3 now empty).
-   Recall that the key formula for Q-learning is below. Every time we
    are in a state s and take an action a , we can update the Q-value
    Q(s, a) according to: Q(s, a) \<- Q(s, a) + alpha \* (new value
    estimate — old value estimate)
-   In the above formula, alpha is the learning rate (how much we value
    new information compared to information we already have).
-   The new value estimate represents the sum of the reward received for
    the current action and the estimate of all the future rewards that
    the player will receive.
-   The old value estimate is just the existing value for Q(s, a) . By
    applying this formula every time our AI agent takes a new action,
    over time our AI agent will start to learn which actions are better
    in any state.
-   The concepts required for reinforcement learning and Q-learning is
    shown in the following figure.

![](https://cdn-images-1.medium.com/max/800/1*xI9G_8wv5IMx9OkUMas9og.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

Next python code snippet shows how to implement q-values update for
Q-learning with the update() method from the NimAI class.

``` {#df69 .graf .graf--pre .graf-after--p name="df69"}
class NimAI():
```

``` {#9685 .graf .graf--pre .graf-after--pre name="9685"}
    def update(self, old_state, action, new_state, reward):        """        Update Q-learning model, given an old state, an action taken        in that state, a new resulting state, and the reward         received from taking that action.        """        old = self.get_q_value(old_state, action)        best_future = self.best_future_reward(new_state)        self.update_q_value(old_state, action, old, reward,                                                     best_future)
```

``` {#d48e .graf .graf--pre .graf-after--pre name="d48e"}
    def update_q_value(self, state, action, old_q, reward,                                                    future_rewards):        """        Update the Q-value for the state `state` and the action         `action` given the previous Q-value `old_q`, a current         reward `reward`, and an estimate of future rewards           `future_rewards`.         Use the formula:
```

``` {#5bd1 .graf .graf--pre .graf-after--pre name="5bd1"}
        Q(s, a) <- old value estimate                 + alpha * (new value estimate - old value estimate)
```

``` {#8cc2 .graf .graf--pre .graf-after--pre name="8cc2"}
        where `old value estimate` is the previous Q-value,        `alpha` is the learning rate, and `new value estimate`        is the sum of the current reward and estimated future         rewards.        """        self.q[(tuple(state), action)] = old_q + self.alpha * (                              (reward + future_rewards) - old_q)
```

The AI agent plays 10000 training games to update the q-values, the
following animation shows how the q-values are updated in last few games
(corresponding to a few selected states out of
(1+1)\*(3+1)\*(5+1)\*(7+1) = 384 states and action pairs):

![](https://cdn-images-1.medium.com/max/800/1*OmhBUZF-yXqVjZ1oaqOavw.gif)

Image by Author

The following animation shows an example game played between the AI
agent (the computer) and the human (the green, red and white boxes in
each row of the first subplot represent the non-empty, to be removed and
empty piles for the corresponding row, respectively, for the AI agent
and the human player).

![](https://cdn-images-1.medium.com/max/800/1*aP51tWxCbzHtOCfTcIJL8Q.gif)

Image by Author

As can be seen from the above figure, one of actions with the highest
q-value for the AI agent’s first turn is (3,6) , that’s why it removes 6
piles from the 3rd row. Similarly, in the second tun of the AI agent,
one of the best actions is (2,5), that’s why it removes 5 piles from row
2 and likewise.

The following animation shows another example game AI agent plays with
human:

![](https://cdn-images-1.medium.com/max/800/1*tJEO55BlmAmhickTCAVCrA.gif)

Image by Author

### Traffic {#b9af .graf .graf--h3 .graf-after--figure name="b9af"}

Write an AI agent to identify which traffic sign appears in a
photograph.

-   As research continues in the development of **self-driving cars**,
    one of the key challenges is [computer
    vision](https://en.wikipedia.org/wiki/Computer_vision), allowing
    these cars to develop an understanding of their environment from
    digital images.
-   In particular, this involves the ability to recognize and
    distinguish road signs — stop signs, speed limit signs, yield signs,
    and more.
-   The following figure contains the theory required to perform the
    multi-class classification task using a deep neural network.

![](https://cdn-images-1.medium.com/max/800/1*uAMEK91JTQA03ExJfMdegQ.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

-   In this problem, we shall use TensorFlow
    ([https://www.tensor􀃦ow.org/](https://www.tensor%F4%80%83%A6ow.org/))
    to build a neural network to classify road signs based on an image
    of those signs.
-   To do so, we shall need a labeled dataset: a collection of images
    that have already been categorized by the road sign represented in
    them.
-   Several such data sets exist, but for this project, we’ll use the
    German Traffic Sign Recognition Benchmark
    ([http://benchmark.ini.rub.de/?section=gtsrb&subsection=news](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news))
    (GTSRB) dataset, which contains thousands of images of 43 different
    kinds of road signs.
-   The dataset contains 43 sub-directories, numbered 0 through 42. Each
    numbered sub-directory represents a different category (a different
    type of road sign). Within each traffic sign’s directory is a
    collection of images of that type of traffic sign.
-   Implement the load\_data() function which should accept as an
    argument data\_dir, representing the path to a directory where the
    data is stored, and return image arrays and labels for each image in
    the data set.
-   While loading the dataset, the images have to be resized to a fixed
    size. If the images are resized correctly, each of them will have
    shape as (30, 30, 3).
-   Also, implement the function get\_model() which should return a
    compiled neural network model.
-   The following python code snippet shows how to implement the
    classification model with keras, along with the output produced when
    the model is run.

``` {#fc32 .graf .graf--pre .graf-after--li name="fc32"}
import numpy as npimport tensorflow as tf
```

``` {#0ca1 .graf .graf--pre .graf-after--pre name="0ca1"}
EPOCHS = 10IMG_WIDTH = 30IMG_HEIGHT = 30NUM_CATEGORIES = 43TEST_SIZE = 0.4
```

``` {#30a1 .graf .graf--pre .graf-after--pre name="30a1"}
# Get image arrays and labels for all image filesimages, labels = load_data('gtsrb')
```

``` {#fdb2 .graf .graf--pre .graf-after--pre name="fdb2"}
# Split data into training and testing setslabels = tf.keras.utils.to_categorical(labels)x_train, x_test, y_train, y_test = train_test_split(     np.array(images), np.array(labels), test_size=TEST_SIZE)
```

``` {#594e .graf .graf--pre .graf-after--pre name="594e"}
# Get a compiled neural networkmodel = get_model()
```

``` {#8554 .graf .graf--pre .graf-after--pre name="8554"}
# Fit model on training datamodel.fit(x_train, y_train, epochs=EPOCHS)
```

``` {#e2ce .graf .graf--pre .graf-after--pre name="e2ce"}
# Evaluate neural network performancemodel.evaluate(x_test,  y_test, verbose=2)
```

``` {#af4b .graf .graf--pre .graf-after--pre name="af4b"}
def get_model():    """    Returns a compiled convolutional neural network model.     Assume that the `input_shape` of the first layer is     `(IMG_WIDTH, IMG_HEIGHT, 3)`.    The output layer should have `NUM_CATEGORIES` units, one for     each category.    """    # Create a convolutional neural network    model = tf.keras.models.Sequential([
```

``` {#f1e6 .graf .graf--pre .graf-after--pre name="f1e6"}
       # Convolutional layer. Learn 32 filters using a 3x3 kernel       tf.keras.layers.Conv2D(         32, (3, 3), activation="relu", input_shape=(30, 30, 3)       ),
```

``` {#b366 .graf .graf--pre .graf-after--pre name="b366"}
       # Max-pooling layer, using 2x2 pool size       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
```

``` {#0a69 .graf .graf--pre .graf-after--pre name="0a69"}
       # Convolutional layer. Learn 32 filters using a 3x3 kernel       tf.keras.layers.Conv2D(         64, (3, 3), activation="relu"       ),
```

``` {#592b .graf .graf--pre .graf-after--pre name="592b"}
       # Max-pooling layer, using 2x2 pool size       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),        tf.keras.layers.Conv2D(         128, (3, 3), activation="relu"       ),
```

``` {#b83b .graf .graf--pre .graf-after--pre name="b83b"}
       # Max-pooling layer, using 2x2 pool size       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
```

``` {#98eb .graf .graf--pre .graf-after--pre name="98eb"}
       # Flatten units       tf.keras.layers.Flatten(),
```

``` {#250f .graf .graf--pre .graf-after--pre name="250f"}
       # Add a hidden layer with dropout       tf.keras.layers.Dense(256, activation="relu"),       tf.keras.layers.Dropout(0.5),
```

``` {#3110 .graf .graf--pre .graf-after--pre name="3110"}
       # Add an output layer with output units for all 10 digits       tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")    ])    print(model.summary)
```

``` {#bb7e .graf .graf--pre .graf-after--pre name="bb7e"}
    # Train neural network    model.compile(       optimizer="adam",       loss="categorical_crossentropy",       metrics=["accuracy"]    )    return model    # Model: "sequential"# _________________________________________________________________# Layer (type)                 Output Shape              Param ## =================================================================# conv2d (Conv2D)              (None, 28, 28, 32)        896# _________________________________________________________________# max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0# _________________________________________________________________# conv2d_1 (Conv2D)            (None, 12, 12, 64)        18496# _________________________________________________________________# max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0# _________________________________________________________________# conv2d_2 (Conv2D)            (None, 4, 4, 128)         73856# _________________________________________________________________# max_pooling2d_2 (MaxPooling2 (None, 2, 2, 128)         0# _________________________________________________________________# flatten (Flatten)            (None, 512)               0# _________________________________________________________________# dense (Dense)                (None, 256)               131328# _________________________________________________________________# dropout (Dropout)            (None, 256)               0# _________________________________________________________________# dense_1 (Dense)              (None, 43)                11051# =================================================================# Total params: 235,627# Trainable params: 235,627# Non-trainable params: 0# _________________________________________________________________
```

-   The architecture of the deep network is shown in the next figure:

![](https://cdn-images-1.medium.com/max/800/1*C6C1lGulq2cwEgkHNlalow.png)

Image by Author

-   The following output shows how the cross-entropy loss on the
    training dataset decreases and the accuracy of classification
    increases over epochs on the training dataset.

``` {#dcf1 .graf .graf--pre .graf-after--li name="dcf1"}
Epoch 1/1015864/15864 [==============================] - 17s 1ms/sample - loss: 2.3697 - acc: 0.4399Epoch 2/1015864/15864 [==============================] - 15s 972us/sample - loss: 0.6557 - acc: 0.8133Epoch 3/1015864/15864 [==============================] - 16s 1ms/sample - loss: 0.3390 - acc: 0.9038Epoch 4/1015864/15864 [==============================] - 16s 995us/sample - loss: 0.2526 - acc: 0.9285Epoch 5/1015864/15864 [==============================] - 16s 1ms/sample - loss: 0.1992 - acc: 0.9440Epoch 6/1015864/15864 [==============================] - 16s 985us/sample - loss: 0.1705 - acc: 0.9512Epoch 7/1015864/15864 [==============================] - 15s 956us/sample - loss: 0.1748 - acc: 0.9536Epoch 8/1015864/15864 [==============================] - 15s 957us/sample - loss: 0.1551 - acc: 0.9578Epoch 9/1015864/15864 [==============================] - 16s 988us/sample - loss: 0.1410 - acc: 0.9630Epoch 10/1015864/15864 [==============================] - 14s 878us/sample - loss: 0.1303 - acc: 0.966710577/10577 - 3s - loss: 0.1214 - acc: 0.9707
```

-   The following animation again shows how the cross-entropy loss on
    the training dataset decreases and the accuracy of classification
    increases over epochs on the training dataset.

![](https://cdn-images-1.medium.com/max/800/0*9hpVWCjTJxYv44jY.gif)

Image by Author

-   The following figure shows a few sample images from the traffic
    dataset with different labels.

![](https://cdn-images-1.medium.com/max/800/1*pyFZLkzQslKEz9bUXDyUjA.png)

Image by Author

-   The next animations show the features learnt at various convolution
    and pooling layers.

![](https://cdn-images-1.medium.com/max/800/0*aB4P6gzr8171oPp2.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*q-7duEcXOTDGTONM.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*bu67Y68ufVv_3Oig.gif)

Image by Author

-   The next figure shows the evaluation result on test dataset — the
    corresponding confusion matrix.

![](https://cdn-images-1.medium.com/max/800/1*7zMjwXTMEPGmZPBhtF0qcA.png)

Image by Author

### Parser {#d743 .graf .graf--h3 .graf-after--figure name="d743"}

Write an AI agent to parse sentences and extract noun phrases.

-   A common task in natural language processing is parsing, the process
    of determining the structure of a sentence.
-   This is useful for a number of reasons: knowing the structure of a
    sentence can help a computer to better understand the meaning of the
    sentence, and it can also help the computer extract information out
    of a sentence.
-   In particular, it’s often useful to extract noun phrases out of a
    sentence to get an understanding for what the sentence is about.
-   In this problem, we’ll use the context-free grammar formalism to
    parse English sentences to determine their structure.
-   Recall that in a context-free grammar, we repeatedly apply rewriting
    rules to transform symbols into other symbols. The objective is to
    start with a non-terminal symbol S (representing a sentence) and
    repeatedly apply context-free grammar rules until we generate a
    complete sentence of terminal symbols (i.e., words).
-   The rule S -\> N V , for example, means that the S symbol can be
    rewritten as N V (a noun followed by a verb). If we also have the
    rule N -\> “Holmes” and the rule V -\> “sat” , we can generate the
    complete sentence “Holmes sat.” .
-   Of course, noun phrases might not always be as simple as a single
    word like “Holmes” . We might have noun phrases like “my companion”
    or “a country walk” or “the day before Thursday” , which require
    more complex rules to account for.
-   To account for the phrase “my companion” , for example, we might
    imagine a rule like: NP -\> N | Det N. In this rule, we say that an
    NP (a “noun phrase”) could be either just a noun ( N ) or a
    determiner ( Det ) followed by a noun, where determiners include
    words like “a” , “the” , and “my” . The vertical bar ( | ) just
    indicates that there are multiple possible ways to rewrite an NP ,
    with each possible rewrite separated by a bar.
-   To incorporate this rule into how we parse a sentence ( S ), we’ll
    also need to modify our S -\> N V rule to allow for noun phrases (
    NP s) as the subject of our sentence. See how? And to account for
    more complex types of noun phrases, we may need to modify our
    grammar even further.
-   Given sentences that are needed to be parsed by the CFG (to be
    constructed) are shown below:

``` {#35ea .graf .graf--pre .graf-after--li name="35ea"}
Holmes sat.Holmes lit a pipe.We arrived the day before Thursday.Holmes sat in the red armchair and he chuckled.My companion smiled an enigmatical smile. Holmes chuckled to himself.She never said a word until we were at the door here.Holmes sat down and lit his pipe.I had a country walk on Thursday and came home in a dreadful mess.I had a little moist red paint in the palm of my hand.
```

**Implementation tips**

-   The NONTERMINALS global variable should be replaced with a set of
    context-free grammar rules that, when combined with the rules in
    TERMINALS , allow the parsing of all sentences in the sentences/
    directory.
-   Each rules must be on its own line. Each rule must include the -\>
    characters to denote which symbol is being replaced, and may
    optionally include | symbols if there are multiple ways to rewrite a
    symbol.
-   Do not need to keep the existing rule S -\> N V in your solution,
    but the first rule must begin with S -\> since S (representing a
    sentence) is the starting symbol.
-   May add as many non-terminal symbols as needed.
-   Use the non-terminal symbol NP to represent a “noun phrase”, such as
    the subject of a sentence.
-   A few rules needed to parse the given sentences are shown below (we
    need to add a few more rules):

``` {#78d0 .graf .graf--pre .graf-after--li name="78d0"}
NONTERMINALS = """S -> NP VP | S Conj VP | S Conj S | S P SNP -> N | Det N | AP NP | Det AP NP | N PP | Det N PP"""
```

-   Few rules for the TERMINALS from the given grammar are shown below:

``` {#02ae .graf .graf--pre .graf-after--li name="02ae"}
TERMINALS = """Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"Conj -> "and"Det -> "a" | "an" | "his" | "my" | "the"N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"V -> "smiled" | "tell" | "were""""
```

-   The next python code snippet implements the function np\_chunk()
    that should accept a nltk tree representing the syntax of a
    sentence, and return a list of all of the noun phrase chunks in that
    sentence.
-   For this problem, a “noun phrase chunk” is defined as a noun phrase
    that doesn’t contain other noun phrases within it. Put more
    formally, a noun phrase chunk is a subtree of the original tree
    whose label is NP and that does not itself contain other noun
    phrases as subtrees.
-   Assume that the input will be a nltk.tree object whose label is S
    (that is to say, the input will be a tree representing a sentence).
-   The function should return a list of nltk.tree objects, where each
    element  has the label NP.
-   The documentation for
    [nltk.tree](https://www.nltk.org/_modules/nltk/tree.html) will be
    helpful for identifying how to manipulate a nltk.tree object.

``` {#c3f8 .graf .graf--pre .graf-after--li name="c3f8"}
import nltk
```

``` {#6f09 .graf .graf--pre .graf-after--pre name="6f09"}
def np_chunk(tree):    """    Return a list of all noun phrase chunks in the sentence tree.    A noun phrase chunk is defined as any subtree of the sentence    whose label is "NP" that does not itself contain any other    noun phrases as subtrees.    """    chunks = []    for s in tree.subtrees(lambda t: t.label() == 'NP'):       if len(list(s.subtrees(lambda t: t.label() == 'NP' and                                         t != s))) == 0:        chunks.append(s)    return chunks
```

The following animation shows how the given sentences are parsed with
the grammar rules constructed.

![](https://cdn-images-1.medium.com/max/800/1*eMWUeENzoQfnNmmcxiV30g.gif)

Image by Author

### Questions {#b32e .graf .graf--h3 .graf-after--figure name="b32e"}

Write an AI agent to answer questions from a given corpus

-   **Question Answering** (QA) is a field within natural language
    processing focused on designing systems that can answer questions.
-   Among the more famous question answering systems is
    [Watson ](https://en.wikipedia.org/wiki/Watson_%28computer%29), the
    IBM computer that competed (and won) on Jeopardy!.
-   A question answering system of Watson’s accuracy requires enormous
    complexity and vast amounts of data, but in this problem, we’ll
    design a very simple question answering system based on inverse
    document frequency.
-   Our question answering system will perform two tasks: document
    retrieval and passage retrieval.
-   Our system will have access to a corpus of text documents.
-   When presented with a query (a question in English asked by the
    user), document retrieval will first identify which document(s) are
    most relevant to the query.
-   Once the top documents are found, the top document(s) will be
    subdivided into passages (in this case, sentences) so that the most
    relevant passage to the question can be determined.
-   How do we find the most relevant documents and passages? To find the
    most relevant documents, we’ll use tf-idf to rank documents based
    both on term frequency for words in the query as well as inverse
    document frequency for words in the query. The required concepts are
    summarized in the following figure:

![](https://cdn-images-1.medium.com/max/800/1*26ffZI34dyJXBDKeu7HhGw.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

-   Once we’ve found the most relevant documents, there are [many
    possible
    metrics](https://groups.csail.mit.edu/infolab/publications/Tellex-etal-SIGIR03.pdf)
    for scoring passages, but we’ll use a combination of inverse
    document frequency and a query term density measure.
-   More sophisticated question answering systems might employ other
    strategies (analyzing the type of question word used, looking for
    synonyms of query words,
    [lemmatizing](https://en.wikipedia.org/wiki/Lemmatisation) to handle
    different forms of the same word, etc.), but that comes only after
    the baseline version is implemented.

**Implementation tips**

-   Implement a function compute\_idfs() that should accept a dictionary
    of documents and return a new dictionary mapping words to their IDF
    (inverse document frequency) values.
-   Assume that documents will be a dictionary mapping names of
    documents to a list of words in that document.
-   The returned dictionary should map every word that appears in at
    least one of the documents to its inverse document frequency value.
-   Recall that the inverse document frequency of a word is defined by
    taking the natural logarithm of the number of documents divided by
    the number of documents in which the word appears.
-   Implement the top\_files() function that should, given a query (a
    set of words), files (a dictionary mapping names of files to a list
    of their words), and idfs (a dictionary mapping words to their IDF
    values), return a list of the filenames the n top files that that
    match the query, ranked according to tf-idf.
-   The returned list of should be of length n and should be ordered
    with the best match first.
-   Files should be ranked according to the sum of tf-idf values for any
    word in the query that also appears in the file. Words in the query
    that do not appear in the file should not contribute to the file’s
    score.
-   Recall that tf-idf for a term is computed by multiplying the number
    of times the term appears in the document by the IDF value for that
    term.
-   Implement the function top\_sentences() that should, given a query
    (a set of words), sentences (a dictionary mapping sentences to a
    list of their words), and idfs (a dictionary mapping words to their
    IDF values), return a list of the n top sentences that match the
    query, ranked according to IDF.
-   The returned list of sentences should be of length n and should be
    ordered with the best match first.
-   Sentences should be ranked according to “matching word measure”:
    namely, the sum of IDF values for any word in the query that also
    appears in the sentence. Note that term frequency should not be
    taken into account here, only inverse document frequency.
-   If two sentences have the same value according to the matching word
    measure, then sentences with a higher “query term density” should be
    preferred. Query term density is defined as the proportion of words
    in the sentence that are also words in the query.
-   The next python code snippet shows how the couple of functions
    mentioned above are implemented:

``` {#1e6f .graf .graf--pre .graf-after--li name="1e6f"}
def compute_idfs(documents):    """    Given a dictionary of `documents` that maps names of documents     to a list of words, return a dictionary that maps words to their     IDF values.    Any word that appears in at least one of the documents should     be in the resulting dictionary.    """    words = set()    for document in documents:        words.update(documents[document])    idfs = dict()    for word in words:        f = sum(word in documents[document] for document in                                                  documents)        idfs[word] = math.log(len(documents) / f)    return idfs
```

``` {#7bbf .graf .graf--pre .graf-after--pre name="7bbf"}
def top_files(query, files, idfs, n):    """    Given a `query` (a set of words), `files` (a dictionary mapping     names of files to a list of their words), and `idfs` (a     dictionary mapping words to their IDF values), return a list of     the filenames of the the `n` top files that match the query,     ranked according to tf-idf.    """    scores = []    for filename in files:        score = 0        for word in query:            tf = len(list(filter(lambda w: w == word,                                            files[filename])))            score += tf * idfs.get(word, 0)        scores.append((score, filename))    scores.sort(key = lambda x: -x[0])    return [item[1] for item in scores][:n]
```

The following animation shows the answers to the questions found from
the given corpus: for each question, the best candidate text (document)
likely containing the answer is first found inside the corpus and then
the top 5 candidate answer sentences is found inside that document.

![](https://cdn-images-1.medium.com/max/800/1*pgl1RWNpENMVRuusTn-DTQ.gif)

Image by Author

This blog is a continuation of my [previous
blog](https://medium.com/r?url=https%3A%2F%2Fsandipan-dey.medium.com%2Fsolving-a-few-ai-problems-with-python-9c5c75f65f3d).
In this blog, we discussed some AI problems related to / to be solved
with **Uncertainty**,**Optimization, Learning, Neural
Network**and**Language**. We discussed how to implement a
(reinforcement-) learning-based game (Nim). We have also seen how to
generate crossword puzzles by representing the problem as a CSP problem.
Finally, we have seen application of AI in Computer Vision (traffic sign
classification needed for self driving cars) and Natural Language
Processing (question-answering system with information retrieval).

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [October 24,
2020](https://medium.com/p/dafd73860d1f).

[Canonical
link](https://medium.com/@sandipan-dey/solving-a-few-ai-problems-with-python-part-2-dafd73860d1f)

Exported from [Medium](https://medium.com) on January 8, 2021.

Solving a few AI problems with Python: Part 1 {.p-name}
=============================================

In this blog we shall discuss about a few problems in artificial
intelligence and their python implementations. The problems discussed…

* * * * *

### Solving a few AI problems with Python: Part 1 {#2688 .graf .graf--h3 .graf--leading .graf--title name="2688"}

#### AI Problems on Search, Logic and Uncertainty {#2bad .graf .graf--h4 .graf-after--h3 .graf--subtitle name="2bad"}

In this blog we shall discuss about a few problems in artificial
intelligence and their python implementations. The problems discussed
here appeared as programming assignments in the edX course****[**CS50’s
Introduction to Artificial Intelligence with Python (HarvardX:CS50
AI)**](https://cs50.harvard.edu/ai/2020/). The problem statements are
taken from the course itself.

### Degrees {#048b .graf .graf--h3 .graf-after--p name="048b"}

Write a program that determines how many “degrees of separation” apart
two actors are.

According to the [Six Degrees of Kevin
Bacon](https://en.wikipedia.org/wiki/Six_Degrees_of_Kevin_Bacon) game,
anyone in the Hollywood film industry can be connected to Kevin Bacon
within six steps, where each step consists of finding a film that two
actors both starred in.

In this problem, we’re interested in finding the shortest path between
any two actors by choosing a sequence of movies that connects them. For
example, the shortest path between Jennifer Lawrence and Tom Hanks is 2:
Jennifer Lawrence is connected to Kevin Bacon by both starring in
“X-Men: First Class,” and Kevin Bacon is connected to Tom Hanks by both
starring in “Apollo 13.”

We can frame this as a search problem: our states are people. Our
actions are movies, which take us from one actor to another (it’s true
that a movie could take us to multiple different actors, but that’s okay
for this problem). Our initial state and goal state are defined by the
two people we’re trying to connect. By using breadth-first search, we
can find the shortest path from one actor to another.

The following figure shows how small samples from the dataset look like:

![](https://cdn-images-1.medium.com/max/800/1*KoCfC9QZ_QjTVHjzFJEdBA.png)

Image created by Author from the [csv files
provided here](https://cs50.harvard.edu/ai/2020/projects/0/degrees/)

-   In the csv file for the people, each person has a unique id,
    corresponding with their id in [IMDb](https://www.imdb.com/)’s
    database. They also have a name, and a birth year.
-   In the csv file for the movies, each movie also has a unique id, in
    addition to a title and the year in which the movie was released.
-   The csv file for the stars establishes a relationship between the
    people and the movies from the corresponding files. Each row is a
    pair of a person\_id value and movie\_id value.
-   Given two actors nodes in the graph we need to find the distance
    (shortest path) between the nodes. We shall use BFS to find the
    distance or the degree of separation. The next figure shows the
    basic concepts required to define a search problem in AI.

![](https://cdn-images-1.medium.com/max/800/1*5ol7zprtW7cQca2zUG4hDA.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

The next python code snippet implements BFS and returns the path between
two nodes source and target in the given input graph.

``` {#a40f .graf .graf--pre .graf-after--p name="a40f"}
def shortest_path(source, target):    """    Returns the shortest list of (movie_id, person_id) pairs    that connect the source to the target.    If no possible path, returns None.    """     explored = set([])    frontier = [source]    parents = {}    while len(frontier) > 0:        person = frontier.pop(0)        if person == target:            break        explored.add(person)        for (m, p) in neighbors_for_person(person):            if not p in frontier and not p in explored:                frontier.append(p)                parents[p] = (m, person)    if not target in parents:        return None    path = []    person = target    while person != source:        m, p = parents[person]        path.append((m, person))        person = p    path = path[::-1]    return path
```

The following animation shows how BFS finds the minimum degree(s) of
separation (distance) between the query source and target actor nodes
(green colored). The shortest path found between the actor nodes (formed
of the edges corresponding to the movies a pair of actors acted together
in) is colored in red.

![](https://cdn-images-1.medium.com/max/800/0*NlISzC4qgKOMXyg6.gif)

Image by Author

The following shows the distribution of the degrees of separation
between the actors.

![](https://cdn-images-1.medium.com/max/800/1*cBbphXn5Dlmf4-q_memrJw.png)

Image by Author

### Tic-Tac-Toe {#0c75 .graf .graf--h3 .graf-after--figure name="0c75"}

Using Minimax, implement an AI to play Tic-Tac-Toe optimally.

-   For this problem, the board is represented as a list of three lists
    (representing the three rows of the board), where each internal list
    contains three values that are either X, O, or EMPTY.
-   Implement a player function that should take a board state as input,
    and return which player’s turn it is (either X or O).
-   In the initial game state, X gets the first move. Subsequently, the
    player alternates with each additional move.
-   Any return value is acceptable if a terminal board is provided as
    input (i.e., the game is already over).
-   Implement an action function that should return a set of all of the
    possible actions that can be taken on a given board.
-   Each action should be represented as a tuple (i, j) where i
    corresponds to the row of the move (0, 1, or 2) and j corresponds to
    which cell in the row corresponds to the move (also 0, 1, or 2).
-   Possible moves are any cells on the board that do not already have
    an X or an O in them.
-   Any return value is acceptable if a terminal board is provided as
    input.
-   Implement a result function that takes a board and an action as
    input, and should return a new board state, without modifying the
    original board.
-   If action is not a valid action for the board, your program should
    raise an exception.
-   The returned board state should be the board that would result from
    taking the original input board, and letting the player whose turn
    it is make their move at the cell indicated by the input action.
-   Importantly, the original board should be left unmodified: since
    Minimax will ultimately require considering many different board
    states during its computation. This means that simply updating a
    cell in board itself is not a correct implementation of the result
    function. You’ll likely want to make a deep copy of the board first
    before making any changes.
-   Implement a winner function that should accept a board as input, and
    return the winner of the board if there is one.
-   If the X player has won the game, your function should return X. If
    the O player has won the game, your function should return O.
-   One can win the game with three of their moves in a row
    horizontally, vertically, or diagonally.
-   You may assume that there will be at most one winner (that is, no
    board will ever have both players with three-in-a-row, since that
    would be an invalid board state).
-   If there is no winner of the game (either because the game is in
    progress, or because it ended in a tie), the function should return
    None.
-   Implement a terminal function that should accept a board as input,
    and return a boolean value indicating whether the game is over.
-   If the game is over, either because someone has won the game or
    because all cells have been filled without anyone winning, the
    function should return True.
-   Otherwise, the function should return False if the game is still in
    progress.
-   Implement an utility function that should accept a terminal board as
    input and output the utility of the board.
-   If X has won the game, the utility is 1. If O has won the game, the
    utility is -1. If the game has ended in a tie, the utility is 0.
-   You may assume utility will only be called on a board if
    terminal(board) is True.
-   Implement a minimax function that should take a board as input, and
    return the optimal move for the player to move on that board.
-   The move returned should be the optimal action (i, j) that is one of
    the allowable actions on the board. If multiple moves are equally
    optimal, any of those moves is acceptable.
-   If the board is a terminal board, the minimax function should return
    None.
-   For all functions that accept a board as input, you may assume that
    it is a valid board (namely, that it is a list that contains three
    rows, each with three values of either X, O, or EMPTY).
-   Since Tic-Tac-Toe is a tie given optimal play by both sides, you
    should never be able to beat the AI (though if you don’t play
    optimally as well, it may beat you!)
-   The following figure demonstrates the basic concepts of adversarial
    search and Game and defines the different functions for tic-tac-toe.
-   Here we shall use Minimax with alpha-beta pruning to speedup the
    game when it is computer’s turn.

![](https://cdn-images-1.medium.com/max/800/1*UdflQeTyI19HT5KT7ycM_w.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

The following python code fragment shows how to implement the minimax
algorithm with alpha-beta pruning:

``` {#db3a .graf .graf--pre .graf-after--p name="db3a"}
def max_value_alpha_beta(board, alpha, beta):    if terminal(board):       return utility(board), None    v, a = -np.inf, None    for action in actions(board):        m, _ = min_value_alpha_beta(result(board, action),                                     alpha, beta)        if m > v:           v, a = m, action           alpha = max(alpha, v)        if alpha >= beta:           break         return (v, a)        def min_value_alpha_beta(board, alpha, beta):    if terminal(board):       return utility(board), None    v, a = np.inf, None    for action in actions(board):        m, _ = max_value_alpha_beta(result(board, action),                                     alpha, beta)        if m < v:           v, a = m, action         beta = min(beta, v)        if alpha >= beta:           break    return (v, a)        def minimax(board):    """    Returns the optimal action for the current player on the board.    """    if terminal(board):       return None    cur_player = player(board)    if cur_player == X:       _, a = max_value_alpha_beta(board, -np.inf, np.inf)      elif cur_player == O:       _, a = min_value_alpha_beta(board, -np.inf, np.inf)    return a
```

The following sequence of actions by the human player (plays as X, the
Max player) and the computer (plays as O, the Min player) shows what the
computer thinks to come up with optimal position of O and the game tree
it produces using Minimax with alpha-beta pruning algorithm shown in the
above figure.

![](https://cdn-images-1.medium.com/max/800/1*Sl9PbYu4IlVaq-o_tL4Bkw.png)

Image by Author

The following animations show how the game tree is created when the
computer thinks for the above turn. The row above the game board denotes
the value of the utility function and it’s color-coded: when the value
of the corresponding state is +1 (in favor of the Max player) it’s
colored as green, when the value is -1 (in favor of the Min player) it’s
colored as red, otherwise it’s colored gray if the value is 0.

![](https://cdn-images-1.medium.com/max/800/0*qNhjGy61bHA_UFb8.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/1*OaixCe0yfd5a_QlNrZ4Riw.gif)

Image by Author

Notice the game tree is full-grown to the terminal leaves. The following
figure shows a path (colored in blue) in the game tree that leads to a
decision taken by the computer to choose the position of O.

![](https://cdn-images-1.medium.com/max/800/1*qBDJqCZSA6DVaV40hyMwRg.png)

Image by Author

Now let’s consider the next sequence of actions by the human player and
the computer. As can be seen, the computer again finds the optimal
position.

![](https://cdn-images-1.medium.com/max/800/1*TY2B7jAbANY8vjrOeFEQ_A.png)

Image by Author

The following animation shows how the game tree is created when the
computer thinks for the above turn.

![](https://cdn-images-1.medium.com/max/800/0*4IcjyIVv4LsEMtXl.gif)

Image by Author

The following figure shows a path (colored in blue) in the game tree
that leads to a decision taken by the computer to choose the position of
O (note that the optimal path corresponds to a tie in a terminal node).

![](https://cdn-images-1.medium.com/max/800/1*3BDLfcpmuD1YGxb5m1o1kA.png)

Image by Author

Finally, let’s consider the next sequence of actions by the human player
and the computer. As can be seen, the computer again finds the optimal
position.

![](https://cdn-images-1.medium.com/max/800/1*EQf5eNuBIkuFN4jz-eiNnw.png)

Image by Author

The following animations show how the game tree is created when the
computer thinks for the above turn.

![](https://cdn-images-1.medium.com/max/800/0*ZS4kSWQ1iTWGkMWv.gif)

Image by Author

The next figure shows the path (colored in blue) in the game tree that
leads to a decision taken by the computer to choose the position of O
(note that the optimal path corresponds to a tie in a terminal node).

![](https://cdn-images-1.medium.com/max/800/1*f4Zhv-l7jKrvi3a8na4eUw.png)

Image by Author

Now let’s consider another example game. The following sequence of
actions by the computer player (AI agent now plays as X, the Max player)
and the human player (plays as O, the Min player) shows what the
computer thinks to come up with optimal position of O and the game tree
it produces using Minimax with alpha-beta pruning algorithm.

![](https://cdn-images-1.medium.com/max/800/1*cUKFr2gggXyw6n9D7qQLhw.png)

Image by Author

The following animations show how the game tree is created when the
computer thinks for the above turn.

![](https://cdn-images-1.medium.com/max/800/0*kM3ggKU5-6uvF-LH.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*iCTBE2ENuXJwcutV.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*Co6oPEeiIax7A222.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*hWJ0y25qGQUOuNqb.gif)

Image by Author

The next figure shows the path (colored in blue) in the game tree that
leads to a decision taken by the computer to choose the next position of
X (note that the optimal path corresponds to a win in a terminal node).

![](https://cdn-images-1.medium.com/max/800/1*nZtdNRb8D0B-30K-0D_16A.png)

Image by Author

Note from the above tree that no matter what action the human player
chooses next, AI agent (the computer) can always win the game. An
example final outcome of the game be seen from the next game tree.

![](https://cdn-images-1.medium.com/max/800/1*-bUwZ11liYB-Aimtz5d2eQ.jpeg)

Image by Author

### Knights {#f38b .graf .graf--h3 .graf-after--figure name="f38b"}

Write a program to solve logic puzzles.

In 1978, logician Raymond Smullyan published “What is the name of this
book?”, a book of logical puzzles. Among the puzzles in the book were a
class of puzzles that Smullyan called “Knights and Knaves” puzzles.

-   In a Knights and Knaves puzzle, the following information is given:
    Each character is either a knight or a knave.
-   A knight will always tell the truth: if knight states a sentence,
    then that sentence is true.
-   Conversely, a knave will always lie: if a knave states a sentence,
    then that sentence is false.
-   The objective of the puzzle is, given a set of sentences spoken by
    each of the characters, determine, for each character, whether that
    character is a knight or a knave.

The task in this problem is to determine how to represent these puzzles
using propositional logic, such that an AI running a model-checking
algorithm could solve these puzzles.

**Puzzles**

1.  Contains a single character: A.\
    - A says “I am both a knight and a knave.”
2.  Has two characters: A and B.\
    - A says “We are both knaves.”\
    - B says nothing.
3.  Has two characters: A and B.\
    - A says “We are the same kind.”\
    - B says “We are of different kinds.”
4.  Has three characters: A, B, and C.\
    - A says either “I am a knight.” or “I am a knave.”, but you don’t
    know which.\
    - B says “A said ‘I am a knave.’”\
    - B then says “C is a knave.”\
    - C says “A is a knight.”

-   In this problem, we shall use model checking to find the solutions
    to the above puzzles. The following figure shows the theory that are
    relevant.

![](https://cdn-images-1.medium.com/max/800/1*5gEaaFniCxUqq1Xmzm1QOQ.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

-   The following python code snippet shows the base class corresponding
    to a propositional logic sentence (taken from the code provided for
    the assignment).

``` {#b74c .graf .graf--pre .graf-after--li name="b74c"}
import itertools
```

``` {#80a9 .graf .graf--pre .graf-after--pre name="80a9"}
class Sentence():
```

``` {#e9ea .graf .graf--pre .graf-after--pre name="e9ea"}
    def evaluate(self, model):        """Evaluates the logical sentence."""        raise Exception("nothing to evaluate")
```

``` {#bea0 .graf .graf--pre .graf-after--pre name="bea0"}
    def formula(self):        """Returns string formula representing logical sentence."""        return ""
```

``` {#5dc8 .graf .graf--pre .graf-after--pre name="5dc8"}
    def symbols(self):        """Returns a set of all symbols in the logical sentence."""        return set()
```

``` {#3fd5 .graf .graf--pre .graf-after--pre name="3fd5"}
    @classmethod    def validate(cls, sentence):        if not isinstance(sentence, Sentence):            raise TypeError("must be a logical sentence")
```

-   Now, a propositional symbol and an operator (unary / binary) can
    inherit this class and implement the methods specific to the symbol
    as shown for the boolean operator AND (the code taken from the
    resources provided for the assignment).

``` {#8f52 .graf .graf--pre .graf-after--li name="8f52"}
class And(Sentence):    def __init__(self, *conjuncts):        for conjunct in conjuncts:            Sentence.validate(conjunct)        self.conjuncts = list(conjuncts)
```

``` {#50b7 .graf .graf--pre .graf-after--pre name="50b7"}
    def __eq__(self, other):        return isinstance(other, And) and \               self.conjuncts == other.conjuncts
```

``` {#5c08 .graf .graf--pre .graf-after--pre name="5c08"}
    def __hash__(self):        return hash(            ("and", tuple(hash(conjunct) for conjunct in                          self.conjuncts))        )
```

``` {#c611 .graf .graf--pre .graf-after--pre name="c611"}
    def __repr__(self):        conjunctions = ", ".join(            [str(conjunct) for conjunct in self.conjuncts]        )        return f"And({conjunctions})"
```

``` {#16c3 .graf .graf--pre .graf-after--pre name="16c3"}
    def add(self, conjunct):        Sentence.validate(conjunct)        self.conjuncts.append(conjunct)
```

``` {#2711 .graf .graf--pre .graf-after--pre name="2711"}
    def evaluate(self, model):        return all(conjunct.evaluate(model) for conjunct in                    self.conjuncts)
```

``` {#8a6e .graf .graf--pre .graf-after--pre name="8a6e"}
    def formula(self):        if len(self.conjuncts) == 1:            return self.conjuncts[0].formula()        return " ∧ ".join([Sentence.parenthesize(conjunct.formula())                           for conjunct in self.conjuncts])
```

``` {#6559 .graf .graf--pre .graf-after--pre name="6559"}
    def symbols(self):        return set.union(*[conjunct.symbols() \               for conjunct in self.conjuncts])
```

-   Finally, we need to represent the KB for every puzzle, as the one
    shown in the code below:

``` {#6702 .graf .graf--pre .graf-after--li name="6702"}
AKnight = Symbol("A is a Knight")AKnave = Symbol("A is a Knave")
```

``` {#b0be .graf .graf--pre .graf-after--pre name="b0be"}
BKnight = Symbol("B is a Knight")BKnave = Symbol("B is a Knave")
```

``` {#8ea5 .graf .graf--pre .graf-after--pre name="8ea5"}
CKnight = Symbol("C is a Knight")CKnave = Symbol("C is a Knave")
```

``` {#29ea .graf .graf--pre .graf-after--pre name="29ea"}
# Puzzle 1# A says "I am both a knight and a knave."knowledge1 = And(    Or(And(AKnight, Not(AKnave)), And(Not(AKnight), AKnave)),    Implication(AKnight, And(AKnight, AKnave)),    Implication(AKnave, Not(And(AKnight, AKnave))))
```

The following section demonstrates the outputs obtained with **model
checking** implemented using the above code (by iteratively assigning
all possible models / truth values and checking if the KB corresponding
to a puzzle evaluates to true for a model, and then outputting all the
models for which the KB evaluates to true). To start solving a puzzle we
must define the KB for the puzzle as shown above.

-   The following figure shows the KB (presented in terms of a
    propositional logic sentence and) shown as an expression tree for
    the Puzzle 1

![](https://cdn-images-1.medium.com/max/800/1*xxNrl1W9GTNT-4rWOwhhoQ.jpeg)

Image by Author

-   The following animation shows how model-checking solves Puzzle 1
    (the solution is marked in green).

![](https://cdn-images-1.medium.com/max/800/0*L9DS8SKHc49nGD6p.gif)

Image by Author

-   The following figure shows the KB in terms of an expression tree for
    the Puzzle 2

![](https://cdn-images-1.medium.com/max/800/1*R-3Youq8dgOWSfF24yDmtA.jpeg)

Image by Author

-   The following animation shows how model-checking solves Puzzle 2
    (the solution is marked in green).

![](https://cdn-images-1.medium.com/max/800/0*AgN5XIJ9keDuQemo.gif)

Image by Author

-   The following figure shows the KB in terms of an expression tree for
    the Puzzle 3

![](https://cdn-images-1.medium.com/max/800/1*qfZVNMaDclUiys45AKFGIQ.jpeg)

Image by Author

-   The following animation shows how model-checking solves Puzzle 3
    (the solution is marked in green).

![](https://cdn-images-1.medium.com/max/800/0*lC-b57RB2GcnL3kV.gif)

Image by Author

-   The following figure shows the KB in terms of an expression tree for
    the Puzzle 4

![](https://cdn-images-1.medium.com/max/800/1*RPFXmQjTaoPYHRyohzR3UA.jpeg)

Image by Author

-   The following animation shows how model-checking solves Puzzle 4
    (the solution is marked in green).

![](https://cdn-images-1.medium.com/max/800/0*wIDhikqpQ3jstSOU.gif)

Image by Author

### Minesweeper {#4cc8 .graf .graf--h3 .graf-after--figure name="4cc8"}

Write an AI agent to play Minesweeper.

![](https://cdn-images-1.medium.com/max/800/0*OL5UlEAk2F9sToVZ.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/1/minesweeper/)

-   Minesweeper is a puzzle game that consists of a grid of cells, where
    some of the cells contain hidden “mines.”
-   Clicking on a cell that contains a mine detonates the mine, and
    causes the user to lose the game.
-   Clicking on a “safe” cell (i.e., a cell that does not contain a
    mine) reveals a number that indicates how many neighboring
    cells — where a neighbor is a cell that is one square to the left,
    right, up, down, or diagonal from the given cell — contain a mine.
-   The following figure shows an example Minesweeper game. In this 3 X
    3 Minesweeper game, for example, the three 1 values indicate that
    each of those cells has one neighboring cell that is a mine. The
    four 0 values indicate that each of those cells has no neighboring
    mine.

![](https://cdn-images-1.medium.com/max/800/0*vW98E5LIWtVokhFj.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/1/minesweeper/)

-   Given this information, a logical player could conclude that there
    must be a mine in the lower-right cell and that there is no mine in
    the upper-left cell, for only in that case would the numerical
    labels on each of the other cells be accurate.
-   The goal of the game is to flag (i.e., identify) each of the mines.
-   In many implementations of the game, the player can flag a mine by
    right-clicking on a cell (or two-finger clicking, depending on the
    computer).

**Propositional Logic**

-   Your goal in this project will be to build an AI that can play
    Minesweeper. Recall that knowledge-based agents make decisions by
    considering their knowledge base, and making inferences based on
    that knowledge.
-   One way we could represent an AI’s knowledge about a Minesweeper
    game is by making each cell a propositional variable that is true if
    the cell contains a mine, and false otherwise.
-   What information does the AI agent have access to? Well, the AI
    would know every time a safe cell is clicked on and would get to see
    the number for that cell.
-   Consider the following Minesweeper board, where the middle cell has
    been revealed, and the other cells have been labeled with an
    identifying letter for the sake of discussion.

![](https://cdn-images-1.medium.com/max/800/0*M7LH0dvaoUji10H9.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/1/minesweeper/)

-   What information do we have now? It appears we now know that one of
    the eight neighboring cells is a mine. Therefore, we could write a
    logical expression like the following one to indicate that one of
    the neighboring cells is a mine: Or(A, B, C, D, E, F, G, H)
-   But we actually know more than what this expression says. The above
    logical sentence expresses the idea that at least one of those eight
    variables is true. But we can make a stronger statement than that:
    we know that exactly one of the eight variables is true. This gives
    us a propositional logic sentence like the below.

``` {#7e49 .graf .graf--pre .graf-after--li name="7e49"}
Or(And(A, Not(B), Not(C), Not(D), Not(E), Not(F), Not(G), Not(H)),And(Not(A), B, Not(C), Not(D), Not(E), Not(F), Not(G), Not(H)),And(Not(A), Not(B), C, Not(D), Not(E), Not(F), Not(G), Not(H)),And(Not(A), Not(B), Not(C), D, Not(E), Not(F), Not(G), Not(H)),And(Not(A), Not(B), Not(C), Not(D), E, Not(F), Not(G), Not(H)),And(Not(A), Not(B), Not(C), Not(D), Not(E), F, Not(G), Not(H)),And(Not(A), Not(B), Not(C), Not(D), Not(E), Not(F), G, Not(H)),And(Not(A), Not(B), Not(C), Not(D), Not(E), Not(F), Not(G), H))
```

-   That’s quite a complicated expression! And that’s just to express
    what it means for a cell to have a 1 in it. If a cell has a 2 or 3
    or some other value, the expression could be even longer.
-   Trying to perform model checking on this type of problem, too, would
    quickly become intractable: on an 8 X 8 grid, the size Microsoft
    uses for its Beginner level, we’d have 64 variables, and therefore
    2⁶⁴ possible models to check — far too many for a computer to
    compute in any reasonable amount of time. We need a better
    representation of knowledge for this problem.

**Knowledge Representation**

-   Instead, we’ll represent each sentence of our AI’s knowledge like
    the below. {A, B, C, D, E, F, G, H} = 1
-   Every logical sentence in this representation has two parts: a set
    of cells on the board that are involved in the sentence, and a
    number count , representing the count of how many of those cells are
    mines. The above logical sentence says that out of cells A, B, C, D,
    E, F, G, and H, exactly 1 of them is a mine.
-   Why is this a useful representation? In part, it lends itself well
    to certain types of inference. Consider the game below:

![](https://cdn-images-1.medium.com/max/800/0*boeBD-_TksR8jDNj.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/1/minesweeper/)

-   Using the knowledge from the lower-left number, we could construct
    the sentence {D, E, G} = 0 to mean that out of cells D, E, and G,
    exactly 0 of them are mines.
-   Intuitively, we can infer from that sentence that all of the cells
    must be safe. By extension, any time we have a sentence whose count
    is 0 , we know that all of that sentence’s cells must be safe.
-   Similarly, consider the game below.

![](https://cdn-images-1.medium.com/max/800/0*fNpA9Phrec-SPteN.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/1/minesweeper/)

-   Our AI agent would construct the sentence {E, F, H} = 3 .
    Intuitively, we can infer that all of E, F, and H are mines.
-   More generally, any time the number of cells is equal to the count ,
    we know that all of that sentence’s cells must be mines.
-   In general, we’ll only want our sentences to be about cells that are
    not yet known to be either safe or mines. This means that, once we
    know whether a cell is a mine or not, we can update our sentences to
    simplify them and potentially draw new conclusions.
-   For example, if our AI agent knew the sentence {A, B, C} = 2 , we
    don’t yet have enough information to conclude anything. But if we
    were told that C were safe, we could remove C from the sentence
    altogether, leaving us with the sentence {A, B} = 2 (which,
    incidentally, does let us draw some new conclusions.)
-   Likewise, if our AI agent knew the sentence {A, B, C} = 2 , and we
    were told that C is a mine, we could remove C from the sentence and
    decrease the value of count (since C was a mine that contributed to
    that count), giving us the sentence {A, B} = 1 . This is logical: if
    two out of A, B, and C are mines, and we know that C is a mine, then
    it must be the case that out of A and B, exactly one of them is a
    mine.
-   If we’re being even more clever, there’s one final type of inference
    we can do

![](https://cdn-images-1.medium.com/max/800/0*7ScdGG3nIwxXwzhd.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/1/minesweeper/)

-   Consider just the two sentences our AI agent would know based on the
    top middle cell and the bottom middle cell. From the top middle
    cell, we have {A, B, C} = 1 . From the bottom middle cell, we have
    {A, B, C, D, E} = 2 . Logically, we could then infer a new piece of
    knowledge, that {D, E} = 1 . After all, if two of A, B, C, D, and E
    are mines, and only one of A, B, and C are mines, then it stands to
    reason that exactly one of D and E must be the other mine.
-   More generally, any time we have two sentences set1 = count1 and
    set2 = count2 where set1 is a subset of set2 , then we can construct
    the new sentence set2 — set1 = count2 — count1 . Consider the
    example above to ensure you understand why that’s true.
-   So using this method of representing knowledge, we can write an AI
    agent that can gather knowledge about the Minesweeper board, and
    hopefully select cells it knows to be safe!
-   The following figure shows examples of inference rules and
    resolution by inference relevant for this problem:

![](https://cdn-images-1.medium.com/max/800/1*X1Wn4ufBJtL2uNW0teX7zg.png)

Image created from the lecture notes from
[this course](https://cs50.harvard.edu/ai/2020/)

**A few implementation tips**

-   Represent a logical sentence by a python class Sentence again.
-   Each sentence has a set of cells within it and a count of how many
    of those cells are mines.
-   Let the class also contain functions known\_mines and known\_safes
    for determining if any of the cells in the sentence are known to be
    mines or known to be safe.
-   It should also have member functions mark\_mine() and mark\_safe()
    to update a sentence in response to new information about a cell.
-   Each cell is a pair (i, j) where i is the row number (ranging from 0
    to height — 1 ) and j is the column number (ranging from 0 to
    width — 1 ).
-   Implement a known\_mines() method that should return a set of all of
    the cells in self.cells that are known to be mines.
-   Implement a known\_safes() method that should return a set of all
    the cells in self.cells that are known to be safe.
-   Implement a mark\_mine() method that should first check if cell is
    one of the cells included in the sentence. If cell is in the
    sentence, the function should update the sentence so that cell is no
    longer in the sentence, but still represents a logically correct
    sentence given that cell is known to be a mine.
-   Likewise mark\_safe() method should first check if cell is one of
    the cells included in the sentence. If yes, then it should update
    the sentence so that cell is no longer in the sentence, but still
    represents a logically correct sentence given that cell is known to
    be safe.
-   Implement a method add\_knowledge() that should accept a cell and
-   The function should mark the cell as one of the moves made in the
    game.
-   The function should mark the cell as a safe cell, updating any
    sentences that contain the cell as well.
-   The function should add a new sentence to the AI’s knowledge base,
    based on the value of cell and count , to indicate that count of the
    cell ’s neighbors are mines. Be sure to only include cells whose
    state is still undetermined in the sentence.
-   If, based on any of the sentences in self.knowledge , new cells can
    be marked as safe or as mines, then the function should do so.
-   If, based on any of the sentences in self.knowledge , new sentences
    can be inferred , then those sentences should be added to the
    knowledge base as well.
-   Implement a method make\_safe\_move() that should return a move (i,
    j) that is known to be safe.
-   The move returned must be known to be safe, and not a move already
    made.
-   If no safe move can be guaranteed, the function should return None .
-   Implement a method make\_random\_move() that should return a random
    move (i, j) .
-   This function will be called if a safe move is not possible: if the
    AI agent doesn’t know where to move, it will choose to move randomly
    instead.
-   The move must not be a move that has already been made.
-   he move must not be a move that is known to be a mine.
-   If no such moves are possible, the function should return None .

The following python code snippet shows implementation of few of the
above functions:

``` {#7cca .graf .graf--pre .graf-after--p name="7cca"}
class MinesweeperAI():    """    Minesweeper game player    """
```

``` {#ca26 .graf .graf--pre .graf-after--pre name="ca26"}
    def mark_mine(self, cell):        """        Marks a cell as a mine, and updates all knowledge        to mark that cell as a mine as well.        """        self.mines.add(cell)        for sentence in self.knowledge:            sentence.mark_mine(cell)
```

``` {#2599 .graf .graf--pre .graf-after--pre name="2599"}
    # ensure that no duplicate sentences are added             def knowledge_contains(self, sentence):        for s in self.knowledge:            if s == sentence:               return True        return False
```

``` {#d64f .graf .graf--pre .graf-after--pre name="d64f"}
    def add_knowledge(self, cell, count):        """        Called when the Minesweeper board tells us, for a given        safe cell, how many neighboring cells have mines in them.
```

``` {#a833 .graf .graf--pre .graf-after--pre name="a833"}
        This function should:            1) mark the cell as a move that has been made            2) mark the cell as safe            3) add a new sentence to the AI's knowledge base               based on the value of `cell` and `count`            4) mark any additional cells as safe or as mines               if it can be concluded based on the AI's KB            5) add any new sentences to the AI's knowledge base               if they can be inferred from existing knowledge        """        # mark the cell as a move that has been made        self.moves_made.add(cell)          # mark the cell as safe        self.mark_safe(cell)        # add a new sentence to the AI's knowledge base,         # based on the value of `cell` and `count        i, j = cell        cells = []        for row in range(max(i-1,0), min(i+2,self.height)):            for col in range(max(j-1,0), min(j+2,self.width)):                # if some mines in the neighbors are already known,                 # make sure to decrement the count                if (row, col) in self.mines:                   count -= 1                 if (not (row, col) in self.safes) and \                   (not (row, col) in self.mines):                   cells.append((row, col))        sentence = Sentence(cells, count)        # add few more inference rules here            def make_safe_move(self):        """        Returns a safe cell to choose on the Minesweeper board.        The move must be known to be safe, and not already a move        that has been made.        This function may use the knowledge in self.mines,          self.safes and self.moves_made, but should not modify any of         those values.        """        safe_moves = self.safes - self.moves_made        if len(safe_moves) > 0:           return safe_moves.pop()        else:           return None
```

The following animations show how the Minesweeper AI agent updates safe
/ mine cells and updates its knowledge-base, for a few example games
(the light green and red cells represent known safe and mine cells,
respectively, in the 2nd subplot):

![](https://cdn-images-1.medium.com/max/800/0*Z5EHTjqn0JM8m34R.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*rkI9JOXHYIzsrxAC.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*566rbRuqvTmd6HEc.gif)

Image by Author

### PageRank {#ab09 .graf .graf--h3 .graf-after--figure name="ab09"}

Write an AI agent to rank web pages by importance.

-   When search engines like Google display search results, they do so
    by placing more “important” and higher-quality pages higher in the
    search results than less important pages. But how does the search
    engine know which pages are more important than other pages?
-   One heuristic might be that an “important” page is one that many
    other pages link to, since it’s reasonable to imagine that more
    sites will link to a higher-quality webpage than a lower-quality
    webpage.
-   We could therefore imagine a system where each page is given a rank
    according to the number of incoming links it has from other pages,
    and higher ranks would signal higher importance.
-   But this definition isn’t perfect: if someone wants to make their
    page seem more important, then under this system, they could simply
    create many other pages that link to their desired page to
    arti􀃥cially in􀃦ate its rank.
-   For that reason, the PageRank algorithm was created by Google’s
    co-founders (including Larry Page, for whom the algorithm was
    named).
-   In PageRank’s algorithm, a website is more important if it is linked
    to by other important websites, and links from less important
    websites have their links weighted less.
-   This definition seems a bit circular, but it turns out that there
    are multiple strategies for calculating these rankings.

**Random Surfer Model**

One way to think about PageRank is with the random surfer model, which
considers the behavior of a hypothetical surfer on the internet who
clicks on links at random.

-   The random surfer model imagines a surfer who starts with a web page
    at random, and then randomly chooses links to follow.
-   A page’s PageRank, then, can be described as the probability that a
    random surfer is on that page at any given time. After all, if there
    are more links to a particular page, then it’s more likely that a
    random surfer will end up on that page.
-   Moreover, a link from a more important site is more likely to be
    clicked on than a link from a less important site that fewer pages
    link to, so this model handles weighting links by their importance
    as well.
-   One way to interpret this model is as a Markov Chain, where each
    page represents a state, and each page has a transition model that
    chooses among its links at random. At each time step, the state
    switches to one of the pages linked to by the current state.
-   By sampling states randomly from the Markov Chain, we can get an
    estimate for each page’s PageRank. We can start by choosing a page
    at random, then keep following links at random, keeping track of how
    many times we’ve visited each page.
-   After we’ve gathered all of our samples (based on a number we choose
    in advance), the proportion of the time we were on each page might
    be an estimate for that page’s rank.
-   To ensure we can always get to somewhere else in the corpus of web
    pages, we’ll introduce to our model a damping factor d .
-   With probability d (where d is usually set around 0.85 ), the random
    surfer will choose from one of the links on the current page at
    random. But otherwise (with probability 1 — d ), the random surfer
    chooses one out of all of the pages in the corpus at random
    (including the one they are currently on).
-   Our random surfer now starts by choosing a page at random, and then,
    for each additional sample we’d like to generate, chooses a link
    from the current page at random with probability d , and chooses any
    page at random with probability 1 — d .
-   If we keep track of how many times each page has shown up as a
    sample, we can treat the proportion of states that were on a given
    page as its PageRank.

**Iterative Algorithm**

-   We can also define a page’s PageRank using a recursive mathematical
    expression. Let PR(p) be the PageRank of a given page p : the
    probability that a random surfer ends up on that page.
-   How do we define PR(p)? Well, we know there are two ways that a
    random surfer could end up on the page: 1. With probability 1 — d ,
    the surfer chose a page at random and ended up on page p . 2. With
    probability d , the surfer followed a link from a page i to page p .
-   The first condition is fairly straightforward to express
    mathematically: it’s 1 — d divided by N , where N is the total
    number of pages across the entire corpus. This is because the 1 — d
    probability of choosing a page at random is split evenly among all N
    possible pages.
-   For the second condition, we need to consider each possible page i
    that links to page p . For each of those incoming pages, let
    NumLinks(i) be the number of links on page i .
-   Each page i that links to p has its own PageRank, PR(i) ,
    representing the probability that we are on page i at any given
    time.
-   And since from page i we travel to any of that page’s links with
    equal probability, we divide PR(i) by the number of links
    NumLinks(i) to get the probability that we were on page i and chose
    the link to page p.
-   This gives us the following definition for the PageRank for a page
    p .

![](https://cdn-images-1.medium.com/max/800/0*Cp-FhMCIWlUpfxLG.png)

Image taken
from [here](https://cs50.harvard.edu/ai/2020/projects/2/pagerank/)

-   In this formula, d is the damping factor, N is the total number of
    pages in the corpus, i ranges over all pages that link to page p ,
    and NumLinks(i) is the number of links present on page i .
-   How would we go about calculating PageRank values for each page,
    then? We can do so via iteration: start by assuming the PageRank of
    every page is 1 / N (i.e., equally likely to be on any page).
-   Then, use the above formula to calculate new PageRank values for
    each page, based on the previous PageRank values.
-   If we keep repeating this process, calculating a new set of PageRank
    values for each page based on the previous set of PageRank values,
    eventually the PageRank values will converge (i.e., not change by
    more than a small threshold with each iteration).
-   Now, let’s implement both such approaches for calculating
    PageRank — calculating both by sampling pages from a Markov Chain
    random surfer and by iteratively applying the PageRank formula.

The following python code snippet represents the implementation of
PageRank computation with Monte-Carlo sampling.

``` {#4290 .graf .graf--pre .graf-after--p name="4290"}
def sample_pagerank(corpus, damping_factor, n):    """    Return PageRank values for each page by sampling `n` pages    according to transition model, starting with a page at random.
```

``` {#76ae .graf .graf--pre .graf-after--pre name="76ae"}
    Return a dictionary where keys are page names, and values are    their estimated PageRank value (a value between 0 and 1). All    PageRank values should sum to 1.    """    pages = [page for page in corpus]    pageranks = {page:0 for page in pages}    page = random.choice(pages)    pageranks[page] = 1    for i in range(n):        probs = transition_model(corpus, page, damping_factor)        probs = probs.values() #[probs[p] for p in pages]        page = random.choices(pages, weights=probs, k=1)[0]        pageranks[page] += 1    return {page:pageranks[page]/n for page in pageranks}
```

The following animation shows how PageRank changes with sampling-based
implementation.

![](https://cdn-images-1.medium.com/max/800/0*fFj3yWMohAbjhqjH.gif)

Image by Author

Again, the following python code snippet represents the iterative
PageRank implementation.

``` {#8d06 .graf .graf--pre .graf-after--p name="8d06"}
def iterate_pagerank(corpus, damping_factor):    """    Return PageRank values for each page by iteratively updating    PageRank values until convergence.
```

``` {#4dd7 .graf .graf--pre .graf-after--pre name="4dd7"}
    Return a dictionary where keys are page names, and values are    their estimated PageRank value (a value between 0 and 1). All    PageRank values should sum to 1.    """    incoming = {p:set() for p in corpus}    for p in corpus:        for q in corpus[p]:           incoming[q].add(p)    pages = [page for page in corpus]    n = len(pages)    pageranks = {page:1/n for page in pages}    diff = float('inf')    while diff > 0.001:        diff = 0        for page in pages:            p = (1-damping_factor)/n            for q in incoming[page]:               p += damping_factor * \                 (sum([pageranks[q]/len(corpus[q])]) \                if len(corpus[q]) > 0 else 1/n)            diff = max(diff, abs(pageranks[page]-p))            pageranks[page] = p       return {p:pageranks[p]/sum(pageranks.values())             for p in pageranks}
```

The following animation shows how the iterative PageRank formula is
computed for a sample very small web graph:

![](https://cdn-images-1.medium.com/max/800/0*9l6NYvJxH_kPvbLs.gif)

Image by Author

Artificial Intelligence is a vast area of research and it contains many
different sub-areas those themselves are huge. In this blog, we
discussed some problems from **Search**, **Knowledge**(Logic) and
**Uncertainty**aspects of AI. We discussed how to implement AI agents
for an adversarial search game (tic-tac-toe) and a logic-based game
(minesweeper). In the next part of this blog we shall discuss on some
more problems on AI from a few more areas of AI research.

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [October 21,
2020](https://medium.com/p/9c5c75f65f3d).

[Canonical
link](https://medium.com/@sandipan-dey/solving-a-few-ai-problems-with-python-9c5c75f65f3d)

Exported from [Medium](https://medium.com) on January 8, 2021.

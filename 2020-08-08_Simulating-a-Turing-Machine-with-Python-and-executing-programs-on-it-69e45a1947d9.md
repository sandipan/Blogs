Simulating a Turing Machine with Python and executing programs on it {.p-name}
====================================================================

Use python code to make your computer behave like a Turing Machine

* * * * *

### Simulating a Turing Machine with Python and executing programs on it {#c947 .graf .graf--h3 .graf--leading .graf--title name="c947"}

#### Use python code to make your computer behave like a Turing Machine {#8dce .graf .graf--h4 .graf-after--h3 .graf--subtitle name="8dce"}

In this article, we shall implement a basic version of a Turing Machine
in python and write a few simple programs to execute them on the Turing
machine. This article is inspired by the **edX / MITx course Paradox and
Infinity**and few of the programs to be executed on the (simulated)
Turing machine are taken from the course. Also, some programs are from
this [Cambridge
tutorial](https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/turing-machine/four.html).

### A few Definitions {#ac9f .graf .graf--h3 .graf-after--p name="ac9f"}

Let’s start by defining a Turing machine (originally invented by ***Alan
Turing***) formally, as is shown in the following figure:

![](https://cdn-images-1.medium.com/max/800/0*ZwVHFy5YtQ0m2YXF)

Image taken from from a lecture
note ([source](https://ict.iitk.ac.in/wp-content/uploads/CS340-Theory-of-Computation-14-toc.pdf))

In this article, we shall assume that the tape alphabet can contain only
the **symbols {0,1,⊔}**, the head on the Turing machine moves in left
(L) / right (R) direction (D) or doesn’t move (\*), upon reading a
symbol on the tape (i.,e., **D** can be one of **{L,R,\*}**). Also,
given a **program** along with a valid **input**, the Turing machine
just halts (goes to the state **halt**) after writing the desired
**output** on its tape.

### The python implementation for simulating a Turing Machine {#74ec .graf .graf--h3 .graf-after--p name="74ec"}

The following python code shows a very simple implementation of a Turing
machine. It accepts a program as a string, with each transition function
defined on a new line. The state **halt** is denoted by **H**.

![](https://cdn-images-1.medium.com/max/800/0*Q7Oi725p8M0XrEcK.png)

In the next section we shall show how a program on this Turing Machine
will look like and how to run such a program, by instantiating the above
class. We shall demonstrate with a few programs.

### 1. Binary Addition with Turing Machine {#35da .graf .graf--h3 .graf-after--p name="35da"}

The following figure shows how to perform binary addition with a Turing
machine, where the binary numbers to be added are input to the Turing
machine and are separated by a single blank. The TM header is assumed to
be positioned at the leftmost position of the first number, in the very
beginning.

![](https://cdn-images-1.medium.com/max/800/0*cw4DFLU0ShPsh8bV)

Image created from the lecture notes of the course
MITx — 24.118x ([source](https://courses.edx.org/courses/course-v1:MITx+24.118x+2T2020/course/))

The program is loaded as a text file (**program.txt**) that looks like
the following (here each line represents a transition function of the
form **𝛿(𝑝,𝑋)=(𝑞,𝑌,D)**, where the 5 tuples are strictly in the order
**p, X, Y, D, q**(the character **\_** represents a blank symbol on the
tape):

> *0 0 0 r 0\
> 0 1 1 r 0\
> 0 \_ \_ r 1\
> 1 0 0 r 1\
> 1 1 1 r 1\
> 1 \_ \_ l 2\
> 2 0 1 l 2\
> 2 1 0 l 3\
> 2 \_ \_ r 5\
> 3 0 0 l 3\
> 3 1 1 l 3\
> 3 \_ \_ l 4\
> 4 0 1 r 0\
> 4 1 0 l 4\
> 4 \_ 1 r 0\
> 5 1 \_ r 5\
> 5 \_ \_ \* H*

Given the two binary numbers, the Turing Machine

-   uses the second number as a counter
-   decrements the second number by one
-   increments the first number by one

till the second number becomes 0.

The following code shows how the above program can be run to add two
input binary numbers **1101**(decimal 13) and **101** (decimal 5) to
output the binary number **10010** (decimal 18). The final state of the
machine is **H** (**halt**), as expected.

`input`{.markup--code .markup--p-code} `=`{.markup--code
.markup--p-code} `'1101_101'program =`{.markup--code .markup--p-code
.u-paddingRight0 .u-marginRight0}
`open('program.txt').read()tm =`{.markup--code .markup--p-code
.u-paddingRight0 .u-marginRight0}
`TuringMachine(program, input)tm.run()# 10010 H`{.markup--code
.markup--p-code}

The following animation shows how the binary numbers are added using the
TM simulator.

![](https://cdn-images-1.medium.com/max/800/0*dSPF2TsNEIeEizSm.gif)

Image by author

### 2. Converting a Binary to a Unary number with TM {#6927 .graf .graf--h3 .graf-after--figure name="6927"}

Let’s assume the TM tape contains a binary representation of a number n
(n\> 0) as a sequence of zeros and ones, and is otherwise blank. The
reader positioned at the left-most member of the sequence. The following
figure shows the program that replaces the original sequence with a
sequence of n ones, where the original sequence names n in binary
notation:

![](https://cdn-images-1.medium.com/max/800/0*xOszmBRv2nF87F37)

Image created from the lecture notes of the course
MITx — 24.118x ([source](https://courses.edx.org/courses/course-v1:MITx+24.118x+2T2020/course/))

The following animation shows the result of running the program on the
TM simulator, with the input **1010**, the output obtained is a sequence
of 10 ones.

![](https://cdn-images-1.medium.com/max/800/0*H7gL9DbVmGJs65rE.gif)

Image by author

### 3. Converting a Unary to a Binary number with TM {#376f .graf .graf--h3 .graf-after--figure name="376f"}

Let’s assume the TM tape contains a sequence of n ones (n\> 0) and is
otherwise blank. The reader positioned at the left-most member of the
sequence. The following TM program replaces the sequence of n ones, with
a sequence of zeroes and ones that names n in binary notation:

![](https://cdn-images-1.medium.com/max/800/0*yARJ4qdqGzuUlhka)

Image created from the lecture notes of the course
MITx — 24.118x ([source](https://courses.edx.org/courses/course-v1:MITx+24.118x+2T2020/course/))

The following animation shows the result of running the program on the
TM simulator, with the input **11111111111**(eleven ones), the output
obtained is the binary representation of 11.

![](https://cdn-images-1.medium.com/max/800/0*1W886cPaeByssxHf.gif)

Image by author

### 4. Doubling the length of a sequence with TM {#21d9 .graf .graf--h3 .graf-after--figure name="21d9"}

The following figure shows the program to be run to double the length of
a sequence of ones, input to the TM

![](https://cdn-images-1.medium.com/max/800/0*QEKAEnx0AZpnP8S0)

Image created from the lecture notes of the course
MITx — 24.118x ([source](https://courses.edx.org/courses/course-v1:MITx+24.118x+2T2020/course/))

The following animation shows the result of running the program on the
TM simulator, starting with five ones as input, obtaining ten ones on
the tape as output.

![](https://cdn-images-1.medium.com/max/800/0*UGzV7eX_yS9pn1nT)

Image by author

### 5. Simulating the 4-state Busy Beaver with TM {#a1cf .graf .graf--h3 .graf-after--figure name="a1cf"}

The following figure shows the program to be run

![](https://cdn-images-1.medium.com/max/800/0*5wuK5H-XiUZFINtl)

Image by author

The following animation shows the result of running the program on the
TM simulator, as expected it outputs 13 ones on the tapes and halts in
107 steps.

![](https://cdn-images-1.medium.com/max/800/0*3OftOn6r3YWkw2z3)

Image by author

### 6. Detecting a Palindrome with TM {#23e9 .graf .graf--h3 .graf-after--figure name="23e9"}

The following program checks a string of symbols on the tape and returns
a **1** if it is a palindrome and a **0** if it is not.

![](https://cdn-images-1.medium.com/max/800/0*_2DNvOQE6Lkknumh)

Image by author

The following animation shows the result of running the program on the
TM simulator with a palindrome.

![](https://cdn-images-1.medium.com/max/800/0*Z62Z_0ycUK4dsclA)

Image by author

The following animation shows the result of running the program on the
TM simulator with a non-palindrome.

![](https://cdn-images-1.medium.com/max/800/0*Ywjwc5LHh2u-lPtf)

Image by author

As per the famous [Church-Turing
thesis](https://mathworld.wolfram.com/Church-TuringThesis.html), any
real-world computation can be translated into an equivalent computation
involving a Turing machine. It will be an interesting exercise to
translate an arbitrary complex program written in a high-level
programming language (to be solved by a computer) to a corresponding
program on a Turing machine, and use the simulator to run the program to
get the desired result written on its tape when it finishes.

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [August 8,
2020](https://medium.com/p/69e45a1947d9).

[Canonical
link](https://medium.com/@sandipan-dey/simulating-a-turing-machine-with-python-and-executing-programs-on-it-69e45a1947d9)

Exported from [Medium](https://medium.com) on January 8, 2021.

when A is square and invertible, the solution to the linear system is unique and it's simply x =… {.p-name}
=================================================================================================

In case the system is over-determined and A is not square / not
invertible, the problem reduces to linear regression and can be solved
with…

* * * * *

when A is square and invertible, the solution to the linear system is
unique and it's simply x = inv(A)\*b.

In case the system is over-determined and A is not square / not
invertible, the problem reduces to linear regression and can be solved
with OLS using pseudo-inverse, as x = inv(A'A)\*A'\*b. We can also use
Moore-Penrose inverse, refer to this:
[https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose\_inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [January 1,
2021](https://medium.com/p/c201e1d9730e).

[Canonical
link](https://medium.com/@sandipan-dey/when-a-is-square-and-invertible-the-solution-to-the-linear-system-is-simply-x-inv-a-b-c201e1d9730e)

Exported from [Medium](https://medium.com) on January 8, 2021.

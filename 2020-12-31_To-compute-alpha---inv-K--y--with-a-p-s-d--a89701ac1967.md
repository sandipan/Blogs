To compute alpha = inv(K)\*y, with a p.s.d. {.p-name}
===========================================

Refer to:
https://mathoverflow.net/questions/9001/inverting-a-covariance-matrix-numerically-stable

* * * * *

To compute alpha = inv(K)\*y, with a p.s.d. matrix K, for numerical
stability it's better to use the cholesky decomposition K=LL', with L
lower triangular, then the equation reduces to alpha = L' \\ L \\ y
(here x = A \\ b denotes the solution to the linear system Ax = b). With
lower triangular L it's easier to solve the linear system of equations.

Refer to:
[https://mathoverflow.net/questions/9001/inverting-a-covariance-matrix-numerically-stable](https://mathoverflow.net/questions/9001/inverting-a-covariance-matrix-numerically-stable)

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [December 31,
2020](https://medium.com/p/a89701ac1967).

[Canonical
link](https://medium.com/@sandipan-dey/to-compute-alpha-inv-k-y-with-a-p-s-d-a89701ac1967)

Exported from [Medium](https://medium.com) on January 8, 2021.

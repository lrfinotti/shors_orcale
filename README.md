# Erdos Quantum Computing Mini Project 1

## Implementation of Oracle for Shor's Algorithm

This is a mini project for the [Erdős Institute](https://www.erdosinstitute.org/)'s [Fall 2025 Quantum Computing Boot Camp](https://www.erdosinstitute.org/programs/fall-2025/quantum-computing-boot-camp).

It contains Jupyter notebook (along side the corresponding Markdown and Python files) that implements an oracle we can use with [Shor's Algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm) to compute orders of units in $\mathbb{Z}/N\mathbb{Z}$.

More precisely, given non-negative integers $a$ and $N$, with $N>0$, we need to implement $U_a$ given by

\[
U_a \left| c \right\rangle_1 \left| x \right\rangle_n = \begin{cases}
  \left| c \right\rangle_1 \left| ax \; \mathrm{mod} N \right\rangle_n, & \text{if $c=1$ and $x <N$}, \\
  \left| c \right\rangle_1 \left| x \right\rangle_n, & \text{otherwise,}
\end{cases}
\]

where $n = \lceil \log_2(N) \rceil$.

The implementation follows Beauregard's [Circuit for Shor’s algorithm using $2n+3$ qubits](https://arxiv.org/pdf/quant-ph/0205095) construction.

## Requirements

We use [Qiskit](https://www.ibm.com/quantum/qiskit), [NumPy](https://numpy.org/), and [pylatexenc](https://github.com/phfaist/pylatexenc).  These can be installed with

```
pip install qiskit numpy pylatexenc
```

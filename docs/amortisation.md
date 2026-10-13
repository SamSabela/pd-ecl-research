# Amortisation: derivation and implementation notes

Reference documentation for `src/ecl/amortisation.py` and the balance schedule in
`src/ecl/provision.py`. It records the derivation of the formulae used, the
reason one algebraic form is preferred over its equivalent, and the vectorisation
conventions the two modules follow.

---

## Part 1: the derivation

### Symbols

| Symbol | Code name | Meaning |
|---|---|---|
| $P$ | `principal` | original loan amount |
| $r$ | `annual_rate` | nominal annual rate, as a decimal (0.12 = 12%) |
| $i$ | `rate_m` | monthly rate, $r/12$ |
| $T$ | `term_months` | contractual term in months |
| $t$ | `periods_elapsed` | payments made to date |
| $A$ | `payment` | level monthly instalment |
| $B(t)$ | `balance` | principal still owed after $t$ payments |

### The principle underlying every formula

An amount received today is worth more than the same amount received next month,
because the amount received today can earn interest in the interim. A payment of
$A$ made $t$ months from now is therefore worth

$$\frac{A}{(1+i)^t}$$

in present terms. Every expression below follows from that single proposition.

### 1. The monthly rate

$$i = \frac{r}{12}$$

This is simple division rather than the compounding conversion
$(1+r)^{1/12}-1$. The compounding form is the more defensible of the two in a
pure finance sense, but consumer loans are quoted on the basis of simple
division, and adopting the alternative would cause the computed instalment to
disagree with Lending Club's own `installment` field. That agreement is retained
deliberately, because it provides an independent cross-check on the schedule.

### 2. The instalment

The lender advances $P$ today and the borrower repays $A$ per month for $T$
months. The two sides must be of equal present value:

$$P = \sum_{k=1}^{T} \frac{A}{(1+i)^k}$$

The right-hand side is a geometric series, and solving it for $A$ gives

$$A = \frac{P \cdot i}{1 - (1+i)^{-T}}$$

**The zero-rate case.** Where $i = 0$ the denominator evaluates to $1 - 1 = 0$.
No interest accrues, so repayment is straight-line:

$$A = \frac{P}{T}$$

This is implemented as a separate branch, since the general formula cannot
express it.

### 3. The outstanding balance, in two forms

The balance after $t$ payments may be written in either of two algebraically
identical ways.

The **retrospective** form grows the debt forward and subtracts what has been
paid:

$$B(t) = P(1+i)^t - A\,\frac{(1+i)^t - 1}{i}$$

The **prospective** form takes the present value of the $T-t$ payments that
remain outstanding:

$$B(t) = A \cdot \frac{1 - (1+i)^{-(T-t)}}{i}$$

**The prospective form is used throughout**, for reasons of numerical precision
rather than of economics. As $t$ approaches $T$ the two terms of the
retrospective form become large and nearly equal, and subtracting them destroys
significant figures, which is catastrophic cancellation. The prospective form
evaluates $(1 - (1+i)^0)/i = 0$ exactly at maturity.

The practical difference was measured at approximately 3.6e-12 and is
economically negligible. The prospective form is preferred because it is exact by
construction, not because it changes any reported number.

**The zero-rate case** is $B(t) = \dfrac{P}{T}\,(T-t)$.

**Boundary behaviour:**

| $t$ | $B(t)$ |
|---|---|
| $0$ | $P$ |
| $T$ | $0$, exactly |
| $> T$ | $0$, which must be imposed by clipping, as the formula alone returns a meaningless value |

### 4. Exposure at default

Exposure at default in month $t$ is the balance at the start of that month, which
is the balance after $t-1$ payments:

$$\text{EAD}(t) = B(t-1)$$

A default occurring during month $t$ crystallises against the amount owed when
that month began, not against the amount owed once the month's payment has been
made.

---

## Part 2: vectorisation conventions

The amortisation functions are called both on single loans, where readability
matters, and across cohorts of several hundred thousand rows, where a Python loop
is not viable. Four conventions reconcile the two cases, and they are applied
consistently across both modules.

### Scalar and array arguments are unified at entry

```python
t = np.asarray(periods_elapsed, dtype=np.float64)
```

A scalar such as `12` becomes a zero-dimensional array, after which every
subsequent operation behaves identically whether the caller supplied one month or
a vector of months. The arithmetic is therefore written once, rather than
maintained as two code paths that may diverge over time.

### Bounds are imposed by clipping

```python
remaining = np.clip(term_months - t, 0.0, None)
```

`None` denotes the absence of a bound on that side. Here the number of remaining
months is floored at zero, so that `36 - 40 = -4` becomes `0` and the schedule
does not continue past maturity.

### Conditional logic is expressed elementwise

```python
balance = np.where(remaining <= 0.0, 0.0, balance)
```

The condition is evaluated for each element, with the second argument taken where
it holds and the third where it does not. A plain Python `if` cannot serve this
purpose, since `if remaining <= 0` raises an ambiguous-truth-value error on an
array.

### The caller's type is preserved on return

```python
return float(balance) if np.isscalar(periods_elapsed) else balance
```

The test is applied to the original argument rather than to the converted array,
because after `np.asarray` every input is an array and the caller's intent is no
longer recoverable. One caveat applies: `np.isscalar(5)` returns `True` whereas
`np.isscalar(np.int64(5))` returns `False`. This is immaterial for the inputs
arising in this project, but it is recorded here because it would not be obvious
to a later reader.

### Broadcasting

```python
payment * (1.0 - (1.0 + rate_m) ** (-remaining)) / rate_m
```

Where `payment` and `rate_m` are scalars and `remaining` is an array, numpy
stretches the scalars across the array and returns one result per element. The
vectorised path is roughly two orders of magnitude faster than the equivalent
loop, which is what makes valuing a cohort of 280,000 loans practical.

The correctness of that fast path is not taken on trust.
`tests/test_provision.py` pins the vectorised schedule against the scalar
`outstanding_balance` reference in `src/ecl/amortisation.py`, and where the two
disagree the fast path is taken to be wrong.

---

## Diagnostics

| Message or symptom | Cause |
|---|---|
| `truth value of an array is ambiguous` | a plain `if` applied to an array, where `np.where` is required |
| `unsupported operand type(s)` | a missing `np.asarray`, leaving a list mixed with a float |
| function returns `None` | the body consists only of a docstring, with no `return` statement |
| `ZeroDivisionError` | the zero-rate branch has not been handled |
| balance continues past maturity | `remaining` has not been clipped |

# Cheat sheet: amortisation maths and the numpy patterns

Reference for `src/ecl/amortisation.py`. Keep it in `docs/`.

---

## Part 1 — the maths

### Symbols

| symbol | code name | meaning |
|---|---|---|
| $P$ | `principal` | original loan amount |
| $r$ | `annual_rate` | nominal annual rate, decimal (0.12 = 12%) |
| $i$ | `rate_m` | monthly rate = $r/12$ |
| $T$ | `term_months` | contractual term, months |
| $t$ | `periods_elapsed` | payments made so far |
| $A$ | `payment` | level monthly instalment |
| $B(t)$ | `balance` | principal still owed after $t$ payments |

### The one idea everything rests on

A rand today is worth more than a rand next month, because today's rand earns
interest. So a payment of $A$ made $t$ months from now is worth only

$$\frac{A}{(1+i)^t}$$

today. That single fact generates every formula below.

### 1. Monthly rate

$$i = \frac{r}{12}$$

Simple division, *not* $(1+r)^{1/12}-1$. The compounding conversion is more
"correct" in a pure finance sense, but consumer loans are quoted with simple
division, and using anything else would make your computed instalment disagree
with Lending Club's own `installment` field — losing you a free cross-check.

### 2. The instalment

The lender hands over $P$ today. The borrower repays $A$ per month for $T$
months. Fairness means those two are worth the same today:

$$P = \sum_{k=1}^{T} \frac{A}{(1+i)^k}$$

That sum is a geometric series. Solving for $A$:

$$\boxed{A = \frac{P \cdot i}{1 - (1+i)^{-T}}}$$

**Zero-rate case.** With $i = 0$ the denominator is $1 - 1 = 0$. But there's no
interest, so repayment is straight-line:

$$A = \frac{P}{T}$$

Handle it as a separate branch — the general formula cannot express it.

### 3. Outstanding balance — the two forms

**(a) Retrospective** — grow the debt, subtract what's been paid:

$$B(t) = P(1+i)^t - A\,\frac{(1+i)^t - 1}{i}$$

**(b) Prospective** — present value of the payments still to come. There are
$T-t$ of them:

$$\boxed{B(t) = A \cdot \frac{1 - (1+i)^{-(T-t)}}{i}}$$

Algebraically identical. **Use (b).** At $t = T$ the two terms in (a) are large
and nearly equal, so subtracting them destroys precision — catastrophic
cancellation. Form (b) evaluates $(1 - (1+i)^0)/i = 0$ exactly.

(Measured, the difference is ~3.6e-12. Negligible economically. Prefer (b)
because it's exact by construction, not because it changes any number.)

**Zero-rate case:** $B(t) = \dfrac{P}{T}\,(T-t)$

**Sanity checks:**

| $t$ | $B(t)$ |
|---|---|
| $0$ | $P$ |
| $T$ | $0$ exactly |
| $> T$ | $0$ — the formula alone gives nonsense, you must clip |

### 4. Exposure at default

EAD in month $t$ is the balance at the **start** of that month — i.e. after
$t-1$ payments:

$$\text{EAD}(t) = B(t-1)$$

A default during month $t$ crystallises against what was owed when the month
began.

---

## Part 2 — the numpy patterns

These four appear in nearly every function you'll write in this project.

### `np.asarray` — one function, scalar or array

```python
t = np.asarray(periods_elapsed, dtype=np.float64)
```

A plain `12` becomes a **0-dimensional array**. Every operation below then
behaves identically whether the caller passed one month or a whole vector. You
write the maths once instead of maintaining two code paths that drift apart.

### `np.clip` — bound a value

```python
remaining = np.clip(term_months - t, 0.0, None)
```

`np.clip(x, low, high)`. `None` means no bound on that side. Here it floors
remaining months at zero, so `36 - 40 = -4` becomes `0`.

### `np.where` — if/else across a whole array

```python
balance = np.where(remaining <= 0.0, 0.0, balance)
```

Reads as: *where the condition holds take the second argument, otherwise the
third.* It is `if/else` applied element-by-element. Plain Python `if` cannot do
this — `if remaining <= 0` on an array raises "truth value of an array is
ambiguous", which you will hit sooner or later.

### `np.isscalar` — return the type you were given

```python
return float(balance) if np.isscalar(periods_elapsed) else balance
```

Test the **original argument**, never the converted array. After `asarray`
everything is an array and the caller's intent is gone.

Caveat: `np.isscalar(5)` is `True`, but `np.isscalar(np.int64(5))` is `False`.
For this project's inputs that's fine; be aware of it.

### Broadcasting

```python
payment * (1.0 - (1.0 + rate_m) ** (-remaining)) / rate_m
```

`payment` and `rate_m` are scalars; `remaining` may be a 37-element array. numpy
stretches the scalars across the array automatically and returns 37 results.
No loop, and roughly a hundred times faster than one.

---

## Common errors and what they mean

| message | cause |
|---|---|
| `truth value of an array is ambiguous` | plain `if` on an array — use `np.where` |
| `unsupported operand type(s)` | forgot `np.asarray`, mixing list and float |
| function returns `None` | body is only a docstring; no `return` |
| `ZeroDivisionError` | unhandled zero-rate branch |
| result grows past maturity | forgot to clip `remaining` |

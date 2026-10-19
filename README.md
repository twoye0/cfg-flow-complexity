# Mapping Complexity over the CFG Family

Toy experiments for studying how classifier-free guidance (CFG) changes the geometry of flow maps, and whether these changes can help explain reductions in mapping complexity.

## Motivation

We consider an invertible flow map

$begin:math:display$
x \= f\(z\)\, \\qquad z \\sim p\_Z\,
$end:math:display$

such that

$begin:math:display$
f\_\{\\\#\}p\_Z \= p\_X\.
$end:math:display$

Now consider a family of distributions indexed by a guidance parameter $begin:math:text$\\omega$end:math:text$,

$begin:math:display$
\\\{p\_\\omega\\\}\_\\omega\,
$end:math:display$

with corresponding invertible flow maps

$begin:math:display$
p\_\\omega \= \(f\^\{\(\\omega\)\}\)\_\{\\\#\}p\_Z\,
$end:math:display$

where the original map is recovered at the reference scale,

$begin:math:display$
f\^\{\(1\)\} \= f\.
$end:math:display$

This defines a family of flow maps

$begin:math:display$
\\mathcal\{F\} \= \\\{f\^\{\(\\omega\)\}\\\}\_\\omega\.
$end:math:display$

The main question of this project is:

> **How does the geometry of the flow map change across the CFG family, and can these geometric changes explain mapping complexity?**

---

## CFG-induced re-mapping

For a data point

$begin:math:display$
x \\sim p\_X\,
$end:math:display$

invertibility of $begin:math:text$f$end:math:text$ allows us to associate it with its latent variable

$begin:math:display$
z \= f\^\{\-1\}\(x\)\.
$end:math:display$

Using the same latent variable, we can generate the corresponding point under another member of the family:

$begin:math:display$
f\^\{\(\\omega\)\}\(z\)\.
$end:math:display$

This defines the re-mapping

$begin:math:display$
\\boxed\{
g\_\\omega\(x\)
\=
f\^\{\(\\omega\)\}\\left\(f\^\{\-1\}\(x\)\\right\)
\}
$end:math:display$

or equivalently,

$begin:math:display$
g\_\\omega
\=
f\^\{\(\\omega\)\} \\circ f\^\{\-1\}\.
$end:math:display$

At the reference scale,

$begin:math:display$
g\_1
\=
f \\circ f\^\{\-1\}
\=
I\.
$end:math:display$

Thus, $begin:math:text$g\_\\omega$end:math:text$ describes how changing the guidance scale acts directly on samples from the original data distribution while preserving their underlying latent coordinates.

---

## Hypothesis: contraction

One candidate explanation for reduced mapping complexity is that $begin:math:text$g\_\\omega$end:math:text$ becomes contractive.

For each data point $begin:math:text$x$end:math:text$, consider the Jacobian

$begin:math:display$
J\_\{g\_\\omega\}\(x\)
\=
\\nabla\_x g\_\\omega\(x\)\.
$end:math:display$

Its operator norm is

$begin:math:display$
\\\|J\_\{g\_\\omega\}\(x\)\\\|\_\{\\mathrm\{op\}\}
\=
\\sigma\_\{\\max\}\(J\_\{g\_\\omega\}\(x\)\)\.
$end:math:display$

We measure the average local expansion/contraction over the data distribution using

$begin:math:display$
\\boxed\{
A\(\\omega\)
\=
\\mathbb\{E\}\_\{x\\sim p\_X\}
\\left\[
\\\|\\nabla\_x g\_\\omega\(x\)\\\|\_\{\\mathrm\{op\}\}
\\right\]\.
\}
$end:math:display$

If $begin:math:text$g\_\\omega$end:math:text$ is contractive, this may provide a geometric mechanism through which the corresponding map $begin:math:text$f\^\{\(\\omega\)\}$end:math:text$ becomes simpler.

The first goal of this repository is to test whether this contraction behavior actually appears across the CFG family rather than assuming it.

---

## Experiment 1: Contraction over the CFG family

### 1. Construct the flow maps

For a small toy distribution, estimate the marginal velocity field directly.

Given a velocity field

$begin:math:display$
\\frac\{dx\_t\}\{dt\}
\=
v\_t\^\{\(\\omega\)\}\(x\_t\)\,
$end:math:display$

numerically integrate the dynamics to construct the corresponding flow maps

$begin:math:display$
f\_\{t\,s\}\^\{\(\\omega\)\}\.
$end:math:display$

For experiments involving arbitrary time pairs, instead of exhaustively enumerating every $begin:math:text$s\<t$end:math:text$, sample time pairs from a time distribution,

$begin:math:display$
s\,t \\sim p\_T\,
\\qquad s\<t\.
$end:math:display$

### 2. Construct $begin:math:text$g\_\\omega$end:math:text$

Using the numerically constructed flow maps, build

$begin:math:display$
g\_\\omega
\=
f\^\{\(\\omega\)\}\\circ f\^\{\-1\}\.
$end:math:display$

A basic sanity check is

$begin:math:display$
g\_1\(x\) \\approx x\.
$end:math:display$

### 3. Differentiate $begin:math:text$g\_\\omega$end:math:text$

For each sample $begin:math:text$x$end:math:text$, compute

$begin:math:display$
J\_\{g\_\\omega\}\(x\)
\=
\\nabla\_x g\_\\omega\(x\)
$end:math:display$

and its largest singular value,

$begin:math:display$
\\sigma\_\{\\max\}\(J\_\{g\_\\omega\}\(x\)\)
\=
\\\|J\_\{g\_\\omega\}\(x\)\\\|\_\{\\mathrm\{op\}\}\.
$end:math:display$

Then estimate

$begin:math:display$
A\(\\omega\)
\\approx
\\frac\{1\}\{N\}
\\sum\_\{i\=1\}\^\{N\}
\\\|J\_\{g\_\\omega\}\(x\_i\)\\\|\_\{\\mathrm\{op\}\}\.
$end:math:display$

Importantly, the operator norm is computed for each sample before averaging.

### 4. Sweep the guidance scale

Evaluate the above quantity across guidance scales $begin:math:text$\\omega$end:math:text$.

The primary visualization is

$begin:math:display$
\\boxed\{
\\omega
\\quad \\text\{vs\.\} \\quad
\\mathbb\{E\}\_\{x\\sim p\_X\}
\[
\\\|\\nabla\_x g\_\\omega\(x\)\\\|\_\{\\mathrm\{op\}\}
\]\.
\}
$end:math:display$

Since $begin:math:text$g\_1\=I$end:math:text$, we expect the reference point

$begin:math:display$
A\(1\) \\approx 1
$end:math:display$

up to numerical integration error.

The purpose of this experiment is to determine empirically whether increasing guidance produces contraction in the induced re-mapping $begin:math:text$g\_\\omega$end:math:text$.

---

## Beyond contraction

Contraction is only the first candidate geometric property.

More generally, we would like to identify a property $begin:math:text$P$end:math:text$ of the induced re-mapping such that

$begin:math:display$
P\(g\_\\omega\)
\\quad\\Longrightarrow\\quad
\\text\{lower mapping complexity of \}f\^\{\(\\omega\)\}\.
$end:math:display$

If contraction does not adequately explain the behavior of the CFG family, we will investigate alternative geometric proxies.

This leads to two broader questions:

1. How should we characterize the family

   $begin:math:display$
   \\mathcal F\=\\\{f\^\{\(\\omega\)\}\\\}\_\\omega\?
   $end:math:display$

2. Which measurable properties of $begin:math:text$g\_\\omega$end:math:text$ are predictive of reduced mapping complexity?

Understanding these relationships may provide a route toward the broader problem of identifying **optimal interpolations**: interpolations whose induced flow maps are easier to represent or learn.

---

## Current roadmap

**Phase 1 — Construct and measure $begin:math:text$g\_\\omega$end:math:text$**

- Define a low-dimensional toy distribution.
- Estimate its marginal velocity field.
- Numerically construct the flow maps.
- Construct $begin:math:text$g\_\\omega\=f\^\{\(\\omega\)\}\\circ f\^\{\-1\}$end:math:text$.
- Differentiate $begin:math:text$g\_\\omega$end:math:text$.
- Compute sample-wise Jacobian operator norms.
- Sweep $begin:math:text$\\omega$end:math:text$.
- Plot operator norm statistics against guidance scale.
- Verify the $begin:math:text$g\_1 \\approx I$end:math:text$ sanity check.

**Phase 2 — Mapping complexity**

After observing the geometry of the CFG family, define and measure suitable proxies for the mapping complexity of $begin:math:text$f\^\{\(\\omega\)\}$end:math:text$.

**Phase 3 — Geometry vs. complexity**

Study whether contraction—or another property of $begin:math:text$g\_\\omega$end:math:text$—predicts changes in mapping complexity across the family.

**Long-term direction — Optimal interpolation**

Use the relationship between induced geometry and mapping complexity to investigate what properties make an interpolation easier to learn.

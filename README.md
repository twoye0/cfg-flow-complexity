# Mapping Complexity over the CFG Family

A 2D toy study of how **classifier-free guidance (CFG)** changes flow-map geometry and mapping complexity.

**한국어:** CFG가 flow map의 geometry를 어떻게 변화시키고, 그 변화가 mapping complexity 감소와 어떻게 연결되는지를 분석하는 2D toy experiment 프로젝트입니다.

The project started from a simple hypothesis:

$$
\mathrm{CFG}
\;\longrightarrow\;
\mathrm{contraction}
\;\longrightarrow\;
\mathrm{lower\ mapping\ complexity}.
$$

The experiments show that contraction captures part of the phenomenon, but is not the full mechanism.

---

## Main Question / 핵심 질문

Consider a family of flow maps indexed by guidance scale $\omega$:

$$
\mathcal F=\{f^{(\omega)}\}_{\omega},
\qquad
p_\omega=(f^{(\omega)})_*p_Z.
$$

We measure mapping complexity by

$$
\boxed{
\mathcal C(f^{(\omega)})
=
\mathbb E_z
\left[
\|\nabla f^{(\omega)}(z)\|_F^2
\right].
}
$$

The main question is:

> **How does CFG change the geometry of the flow map, and what mechanism explains the resulting change in mapping complexity?**

**한국어:** CFG scale이 변할 때 flow map의 geometry가 어떻게 달라지고, 그 변화가 mapping complexity 감소를 어떻게 설명하는지가 핵심 질문입니다.

The toy problem uses **analytic marginal velocity fields**, allowing us to study the CFG effect without a neural velocity estimator.

따라서 neural velocity estimation error와 분리하여 **CFG 자체가 만드는 flow-map geometry의 변화**를 관찰할 수 있습니다.

---

## CFG-Induced Re-Mapping

Using the conditional flow at $\omega=1$ as the reference,

$$
f=f^{(1)},
$$

we define the CFG-induced re-mapping

$$
g_\omega
=
f^{(\omega)}\circ f^{-1}.
$$

Equivalently,

$$
g_\omega
=
f_{1,0}^{(\omega)}
\circ
f_{0,1}.
$$

At the reference scale,

$$
g_1=I.
$$

This lets us examine how changing CFG scale deforms reference samples while keeping their latent coordinates fixed.

**한국어:** 동일한 latent coordinate를 유지하면서 CFG scale만 바꾸었을 때 reference data가 어떻게 재배치되는지를 $g_\omega$를 통해 관찰합니다.

---

## Experimental Story / 실험 흐름

### Exp1–2 — Does CFG Induce Contraction?

We first measured the local operator norm

$$
\|\nabla g_\omega(x)\|_{\mathrm{op}}
=
\sigma_{\max}(\nabla g_\omega(x)).
$$

Exp1 showed **directional contraction**. After introducing anisotropic target covariances in Exp2, the sampled local operator norms became smaller than 1 for $\omega>1$.

| $\omega$ | Mean operator norm | Sampled max |
|---:|---:|---:|
| 1.25 | 0.988229 | 0.999015 |
| 1.50 | 0.982124 | 0.998091 |
| 2.00 | 0.975253 | 0.996398 |
| 3.00 | 0.968394 | 0.993514 |
| 4.00 | 0.964710 | 0.991134 |

**한국어:** CFG가 실제로 contraction을 만드는지 먼저 확인했습니다. Exp1에서는 directional contraction을 확인했고, Exp2의 anisotropic toy에서는 sampled local operator norm이 1보다 작아졌습니다.

These are sampled local results, not a proof of global contraction over the entire space.

---

### Exp3 — CFG Reduces Mapping Complexity

We directly measured

$$
\mathcal C(f^{(\omega)})
=
\mathbb E_z
\left[
\|\nabla f^{(\omega)}(z)\|_F^2
\right].
$$

| $\omega$ | Mapping complexity | Ratio to $\omega=1$ |
|---:|---:|---:|
| 1.00 | 1.250000 | 1.000000 |
| 1.25 | 1.019685 | 0.815748 |
| 1.50 | 0.895747 | 0.716597 |
| 2.00 | 0.761131 | 0.608905 |
| 3.00 | 0.637594 | 0.510075 |
| 4.00 | 0.576776 | 0.461420 |

Thus, in this analytic toy,

$$
\boxed{
\text{increasing CFG scale substantially reduces mapping complexity.}
}
$$

**한국어:** neural velocity estimator가 없는 analytic toy에서도 CFG scale이 증가하면서 mapping complexity가 크게 감소했습니다. 따라서 이 toy에서의 CFG 효과는 neural velocity estimation error만으로 설명되지 않습니다.

---

### Exp4 — Global Contraction Is Not Necessary

Let $F$ be the reference flow Jacobian and $G$ the CFG-induced re-mapping Jacobian. Then

$$
J_{f^{(\omega)}}=GF.
$$

The operator norm gives the bound

$$
\|GF\|_F^2
\le
\|G\|_{\mathrm{op}}^2
\|F\|_F^2,
$$

but it ignores how reference Jacobian energy is distributed across directions.

The exact pointwise change can instead be written as

$$
\boxed{
\Delta
=
\operatorname{tr}
\left[
(G^\top G-I)M
\right],
\qquad
M=FF^\top.
}
$$

Exp4 found samples where one direction expanded while total Frobenius mapping complexity still decreased.

**한국어:** 모든 방향이 contraction되어야 complexity가 감소하는 것은 아닙니다. 어떤 방향에 기존 flow의 deformation energy가 얼마나 집중되어 있는지도 중요합니다.

---

## Exact Deformation Dynamics / 핵심 메커니즘

Let

$$
\dot x_t=v_t(x_t),
$$

and define

$$
J_t=\nabla_z f_t(z),
\qquad
A_t=\nabla_xv_t(x_t).
$$

The flow Jacobian satisfies

$$
\dot J_t=A_tJ_t.
$$

Define

$$
S_t
=
\frac{A_t+A_t^\top}{2}
$$

and the accumulated deformation matrix

$$
M_t=J_tJ_t^\top.
$$

Since

$$
\|J_t\|_F^2=\operatorname{tr}(M_t),
$$

the exact mapping-complexity dynamics are

$$
\boxed{
\dot C_t
=
2\mathbb E_z
\left[
\operatorname{tr}(S_tM_t)
\right].
}
$$

Equivalently,

$$
\boxed{
C_1
=
d+
2\int_0^1
\mathbb E_z
\left[
\operatorname{tr}(S_tM_t)
\right]dt.
}
$$

This is an **exact identity**, not a proxy.

- $S_t$: instantaneous signed stretching induced by the velocity field.
- $M_t$: deformation geometry accumulated by the flow up to time $t$.

**한국어:** mapping complexity는 현재 velocity gradient의 크기만으로 결정되지 않습니다. 현재의 **signed stretching** $S_t$와 지금까지 누적된 **deformation geometry** $M_t$의 상호작용에 의해 변화합니다.

---

## Exp5 — Boundary Tests

Exp5 tested which information is actually sufficient to characterize final mapping complexity.

### Exp5-A — Rotating Gaussian Path

Different intermediate covariance rotations produced the same final complexity:

$$
C_1=4.25.
$$

In this linear-Gaussian family,

$$
J_1J_1^\top=\Sigma_1,
$$

so the endpoint covariance fixes the final Frobenius complexity.

**한국어:** 이 결과는 rotation이 중요하지 않다는 일반적인 결론이 아니라, 해당 linear-Gaussian family가 구조적으로 final Frobenius complexity의 path dependence를 보여주기 어렵다는 것을 의미합니다.

### Exp5-B — Same Marginals, Different Transport

We constructed different transport couplings with exactly the same marginal probability path $p_t$.

Gauss-Hermite population calculations gave

$$
C_1(\beta=0)
\approx0.77461145,
$$

$$
C_1(\beta=60^\circ)
\approx0.77553262.
$$

Thus,

$$
\boxed{
\text{the marginal probability path }p_t
\text{ does not uniquely determine final mapping complexity.}
}
$$

### Exp5-C — Low-Information Velocity Statistics

Simple velocity summaries such as

$$
2\int_0^1
\mathbb E[\operatorname{tr}(S_t)]dt
$$

did not reliably reproduce the ordering of true mapping complexity.

The deformation weighting $M_t$ cannot generally be discarded.

### Exp5-D — Different Probability Paths

A genuinely different interpolation family was also tested. Although the intermediate marginal distributions changed, the final population complexity remained essentially invariant in the tested family.

**한국어:** Exp5 전체를 통해 endpoint, marginal path, 또는 단순한 instantaneous velocity statistic만으로 final mapping complexity를 일반적으로 설명하기 어렵다는 경계를 확인했습니다.

---

## Exp6 — Direct CFG Mechanism Test

Finally, we directly compared the baseline flow at

$$
\omega=1
\qquad\text{and}\qquad
\omega=2.
$$

Define the exact instantaneous complexity rate

$$
R_t^{(\omega)}
=
2\mathbb E
\left[
\operatorname{tr}
\left(
S_t^{(\omega)}M_t^{(\omega)}
\right)
\right].
$$

The final complexities are

$$
C_1^{(1)}=1.25,
\qquad
C_1^{(2)}\approx0.761131.
$$

Therefore,

$$
C_1^{(2)}-C_1^{(1)}
\approx-0.488869.
$$

The integrated difference in exact rates is

$$
\int_0^1
\left(
R_t^{(2)}-R_t^{(1)}
\right)dt
\approx-0.488857,
$$

matching the observed final complexity difference up to numerical integration error.

Therefore,

$$
\boxed{
C_1^{(2)}-C_1^{(1)}
=
\int_0^1
2\mathbb E
\left[
\operatorname{tr}
\left(
S_t^{(2)}M_t^{(2)}
-
S_t^{(1)}M_t^{(1)}
\right)
\right]dt.
}
$$

### What Does CFG Actually Change?

CFG is **not** more complexity-decreasing at every instant.

Instead, the baseline toy shows

$$
\boxed{
\text{stronger early complexity decrease}
+
\text{delayed/weaker late re-expansion}.
}
$$

**한국어:** CFG는 모든 시간에서 더 강한 contraction을 만드는 것이 아닙니다. 초기에는 complexity를 더 강하게 감소시키고, 이후 re-expansion을 늦추고 약화시킵니다. 이 전체 deformation history의 차이가 최종 mapping complexity 차이를 만듭니다.

---

## Main Figures / 주요 결과

### Complexity Dynamics

![CFG complexity comparison](results/figures/exp06_complexity_comparison.png)

### Exact Complexity Rate

![CFG rate comparison](results/figures/exp06_rate_comparison.png)

### CFG-Induced Rate Difference

![CFG rate difference](results/figures/exp06_cfg_rate_difference.png)

### Accumulated CFG Effect

![Accumulated CFG effect](results/figures/exp06_accumulated_cfg_effect.png)

---

## Main Takeaways / 핵심 결과

1. **CFG reduces mapping complexity in the analytic toy.**
   CFG는 neural velocity estimation error가 없는 toy에서도 mapping complexity를 감소시킵니다.

2. **Uniform/global contraction is not the full mechanism.**
   모든 방향의 contraction은 complexity 감소를 설명하기 위한 필수조건이 아닙니다.

3. **Deformation history matters.**
   Complexity evolution depends on the interaction between $S_t$ and $M_t$.

4. **A single dominant direction is insufficient.**
   하나의 deformation direction만으로 전체 complexity 변화를 설명할 수 없습니다.

5. **Marginal probability paths do not uniquely determine mapping complexity.**
   같은 $p_t$에서도 transport coupling이 달라지면 다른 flow-map complexity가 나타날 수 있습니다.

---

## Repository Structure

```text
cfg-flow-complexity/
├── experiments/
│   ├── exp01_contraction.py
│   ├── exp02_contraction.py
│   ├── exp03_*.py
│   ├── exp04_*.py
│   ├── exp05_pre_*.py
│   ├── exp05a_rotating_gaussian.py
│   ├── exp05b_*.py
│   ├── exp05c_low_information.py
│   ├── exp05d_interpolation_family.py
│   └── exp06_cfg_mechanism.py
│
├── src/
│   ├── distributions.py
│   ├── velocity.py
│   ├── flow.py
│   └── remapping.py
│
├── results/
│   ├── figures/
│   └── *.npz
│
└── README.md
```

---

## Running the Experiments / 실행 방법

Activate the environment:

```bash
source .venv/bin/activate
```

Run experiments from the repository root:

```bash
python -m experiments.exp01_contraction
python -m experiments.exp02_contraction

python -m experiments.exp03_mapping_complexity
python -m experiments.exp03_contraction_bound
python -m experiments.exp03_directional_decomposition

python -m experiments.exp04_decoupling
python -m experiments.exp04_directional_diagnostic

python -m experiments.exp05_pre_mt_dynamics
python -m experiments.exp05_pre_directional_contributions

python -m experiments.exp05a_rotating_gaussian

python -m experiments.exp05b_velocity_coupling
python -m experiments.exp05b_convergence
python -m experiments.exp05b_quadrature

python -m experiments.exp05c_low_information
python -m experiments.exp05d_interpolation_family

python -m experiments.exp06_cfg_mechanism
```

Generated numerical results and figures are stored under `results/`.

---

## Scope / 해석 범위

This repository is an **exploratory mechanism study**.

It does **not** establish that:

- CFG globally contracts arbitrary flow maps;
- CFG is more contractive at every time;
- rotation itself causes complexity reduction;
- one dominant direction explains the full effect;
- transport is universally more important than the probability path;
- lower Jacobian complexity guarantees lower neural-network approximation error;
- the same mechanism automatically generalizes to high-dimensional generative models.

**한국어:** 현재 결과는 CFG가 왜 neural network 관점에서 더 learnable한 map을 만드는지까지 증명한 것이 아닙니다. 여기서는 **CFG가 exact toy flow의 mapping geometry를 어떻게 변화시키는지**를 분석합니다.

---

## Conclusion / 결론

The initial hypothesis was

$$
\mathrm{CFG}
\rightarrow
\mathrm{contraction}
\rightarrow
\mathrm{lower\ mapping\ complexity}.
$$

The experiments suggest a more precise picture:

$$
\boxed{
\mathrm{CFG}
\rightarrow
\text{changes in }S_t\text{ and }M_t
\rightarrow
\text{different deformation history}
\rightarrow
\mathrm{lower\ final\ mapping\ complexity}.
}
$$

> **EN:** In this analytic 2D toy, CFG lowers mapping complexity not through uniform contraction, but through a time-dependent interaction between instantaneous signed stretching and accumulated deformation geometry.

> **KR:** 이 analytic 2D toy에서 CFG의 mapping complexity 감소는 단순한 uniform contraction보다는 **순간적인 signed stretching과 누적된 deformation geometry의 시간적 상호작용**으로 설명됩니다.

---

**Status:** Core exploratory experiments completed through **Exp6**.

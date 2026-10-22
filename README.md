# Mapping Complexity over the CFG Family

> **Can classifier-free guidance (CFG) make a flow map geometrically simpler, even when there is no neural velocity-estimation error?**

이 프로젝트는 위 질문에서 시작한 2D analytic toy study이다.
처음에는 **CFG가 induced re-mapping을 contraction시키기 때문에 mapping complexity가 감소할 것**이라고 예상했다. 실험을 진행하면서 global contraction만으로는 충분한 설명이 아니라는 것을 확인했고, 최종적으로는 **velocity-gradient가 만들어내는 시간적 deformation history**까지 내려가 CFG의 효과를 분석했다.

---

## Research Flow

**CFG**
→ induced re-mapping의 contraction
→ 실제 flow-map complexity 감소
→ global contraction만으로는 설명이 부족함
→ directional Jacobian energy가 중요함
→ Jacobian dynamics로 이동
→ path / transport / velocity geometry의 역할을 분리
→ CFG가 complexity를 낮추는 직접적인 deformation mechanism 확인

핵심적으로 연구 질문은 다음과 같이 바뀌었다.

> **“CFG는 contraction을 만드는가?”**
> ↓
> **“어떤 velocity/deformation history가 low-complexity flow map을 만드는가?”**

---

# 1. From CFG Contraction to Mapping Complexity

## Exp 1 — CFG-Induced Re-Mapping Contraction

### Question
**CFG를 강하게 하면 induced re-mapping $g_\omega$가 contraction하는가?**

CFG flow와 unguided flow 사이의 data-to-data re-mapping을

```math
g_\omega
=
f^{(\omega)} \circ f^{-1}
```

로 두고 Jacobian operator norm을 측정했다.

### Result
한 방향에서는 contraction이 나타났지만, 다른 방향이 identity로 남아 있어 operator norm은 1이었다.

### Interpretation
CFG가 directional contraction을 만들고 있다는 신호는 보였지만, strict full-space contraction은 아니었다.

**→ 그렇다면 이것은 단순한 toy degeneracy 때문인가?**

---

## Exp 2 — Removing the Identity-Direction Degeneracy

### Question
**Exp 1의 결과가 identity direction 때문에 생긴 특수한 현상인가?**

Class covariance를 anisotropic하게 바꾸어 Exp 1에서 남아 있던 identity direction을 제거했다.

### Result
$\omega>1$에서 sampled region의 local operator norm이 1보다 작아졌다.

즉, 이 toy에서는 CFG가 **strict local contraction**을 만들 수 있었다.

### Interpretation
CFG-induced contraction 자체는 실제로 존재했다. 다만 sampled local contraction이지 global contraction의 증명은 아니다.

**→ 그렇다면 이 contraction이 실제 flow-map complexity 감소로 이어지는가?**

---

## Exp 3 — Mapping Complexity over the CFG Family

### Question
**CFG를 강하게 할수록 실제 flow map도 더 simple해지는가?**

Mapping complexity를

```math
\mathcal{C}(f^{(\omega)})
=
\mathbb{E}_z
\left[
\left\|
\nabla f^{(\omega)}(z)
\right\|_F^2
\right]
```

로 측정했다.

| CFG scale $\omega$ | Mapping complexity |
|---:|---:|
| 1.0 | 1.2500 |
| 1.5 | 0.8957 |
| 2.0 | 0.7611 |
| 3.0 | 0.6376 |
| 4.0 | 0.5768 |

### Result
CFG scale이 증가할수록 mapping complexity가 크게 감소했다.

특히 이 실험에서는 analytic marginal velocity를 사용했기 때문에 neural velocity estimator가 개입하지 않는다.

### Interpretation
적어도 이 toy에서는 CFG의 이득을 **velocity-network approximation error만으로 설명할 수 없다.**
CFG 자체가 induced flow-map geometry를 변화시키고 있었다.

**→ 그렇다면 Exp 1–2에서 본 operator-norm contraction이 이 감소를 잘 설명하는가?**

---

## Exp 3 Diagnostic — Why Is the Operator-Norm Bound Loose?

수학적으로는

```math
\|G F\|_F^2
\le
\|G\|_{\mathrm{op}}^2
\|F\|_F^2
```

라는 upper bound를 사용할 수 있다.

하지만 실제 실험에서는 이 bound가 상당히 loose했다.

### Directional Diagnostic

이유는 **모든 input-space direction이 reference flow map $f$의 mapping complexity에 똑같이 기여하지 않기 때문**이었다.

Reference Jacobian energy가 많이 실린 방향은 CFG에 의해 강하게 contract하고 있었고, energy가 적게 실린 방향은 상대적으로 약하게 contract하고 있었다.

즉, 중요한 것은 단순히

> “가장 덜 contract한 방향이 무엇인가?”

가 아니라

> **“reference flow의 Jacobian energy가 많이 실린 방향이 어떻게 변하는가?”**

였다.

**→ 그렇다면 일부 방향이 expand하더라도, high-energy direction이 충분히 contract하면 전체 complexity는 감소할 수 있는가?**

---

## Exp 4 — Contraction / Mapping Complexity Decoupling

### Question
**Global contraction 없이도 mapping complexity가 감소할 수 있는가?**

Mean-level에서 미리 정의한 강한 decoupling criterion은 성립하지 않았다.

하지만 individual samples에서는 실제 decoupling이 나타났다.

### Result
Decoupled sample들은 모두 같은 구조를 보였다.

> **low-weight direction: expansion**
> **high-weight direction: contraction**

즉, 어떤 방향에서는 expansion이 존재해도 reference Jacobian energy가 많이 실린 방향이 충분히 contract하면 전체 Frobenius complexity는 감소할 수 있었다.

### What Changed Here?

여기서 초기 가설

> **CFG → contraction → low mapping complexity**

은 너무 강한 설명이라는 것이 드러났다.

Contraction은 유용한 sufficient condition이지만, mapping complexity 감소를 이해하려면 **flow가 어떤 directional deformation을 시간에 따라 누적하는지**를 봐야 했다.

---

# 2. Moving One Step Back: Jacobian Dynamics

Exp 1–4에서는 이미 만들어진 flow map의 Jacobian을 비교했다.

그다음 질문은 더 근본적이었다.

> **그렇다면 $J_f$ 자체는 어디서 만들어지는가?**

Flow trajectory가

```math
\frac{d x_t}{dt}
=
v_t(x_t)
```

를 따른다면, flow Jacobian $J_t$는

```math
\frac{dJ_t}{dt}
=
\nabla_x v_t(x_t) J_t
```

를 따른다.

즉, **velocity가 공간의 위치에 따라 어떻게 달라지는지가 시간이 지나면서 flow map의 local deformation을 만들어낸다.**

Velocity Jacobian의 symmetric part를

```math
S_t
=
\frac{
\nabla v_t+\nabla v_t^\top
}{2}
```

라고 하면,

```math
\frac{d}{dt}
\|J_t\|_F^2
=
2\,\operatorname{tr}
\left(
S_t J_tJ_t^\top
\right)
```

를 얻는다.

여기서

```math
M_t = J_tJ_t^\top
```

라고 두면 population-level complexity dynamics는

```math
\frac{d}{dt}\mathcal{C}_t
=
2\,
\mathbb{E}
\left[
\operatorname{tr}(S_tM_t)
\right]
```

가 된다.

이 식이 이후 실험의 중심이 되었다.

- $S_t$: **현재 velocity field가 만드는 signed stretching / contraction**
- $M_t$: **그 시점까지 flow가 누적해 온 deformation geometry**

따라서 mapping complexity는 단순히 현재 velocity gradient의 크기로 결정되는 것이 아니라,

> **현재의 stretching geometry와 과거에 누적된 deformation geometry가 어떻게 align되는가**

에 의해 변한다.

---

# 3. Can We Remove the Full Flow Jacobian?

여기서 새로운 문제가 생긴다.

Exact dynamics의 오른쪽에도 여전히 $J_t$, 즉 $M_t$가 남아 있다.

결국 exact mapping complexity를 계산하려면 flow-map deformation history를 알아야 한다는 순환이 존재한다.

그래서 질문을 다음과 같이 바꾸었다.

> **Can flow-map complexity be predicted or controlled from path-level velocity geometry without explicitly constructing the full flow map?**

먼저 단순한 경우부터 확인했다.

- **1D:** rotation이나 directional alignment 문제가 없다.
- **Fixed-direction 2D:** directional stretching이 어떻게 누적되는지 볼 수 있다.
- **Rotating 2D:** 시간에 따라 중요한 방향이 바뀌면 단순한 scalar summary가 정보를 잃을 수 있다.

이 분석을 통해 $S_t$의 norm이나 최대 eigenvalue 같은 하나의 scalar만으로는 일반적인 deformation history를 보존하기 어렵다는 점이 분명해졌다.

또한 $M_t$의 top-energy direction 하나만 남기는 rank-1 approximation도 실제 complexity change를 제대로 복원하지 못했다.

**→ 따라서 Exp 5에서는 probability path, transport coupling, velocity statistics가 각각 얼마나 많은 정보를 제공하는지 직접 분리해서 확인했다.**

---

# 4. Exp 5 — What Information Determines Mapping Complexity?

## Exp 5-A — Rotating Gaussian Path

### Question
**시간에 따라 deformation direction을 회전시키면 final complexity가 달라지는가?**

### Result
테스트한 linear-Gaussian family에서는 회전을 바꾸어도 final Frobenius complexity가 동일했다.

### Interpretation
이 family에서는 endpoint covariance가 final complexity를 구조적으로 고정하고 있었다.

즉, **모든 toy family가 interpolation-dependent complexity를 연구하기에 적합한 것은 아니다.**

---

## Exp 5-B — Same Marginals, Different Transport

### Question
**동일한 marginal probability path $p_t$라도 transport coupling이 다르면 mapping complexity가 달라질 수 있는가?**

### Result
동일한 $p_t$를 유지하면서 transport coupling만 바꾸었을 때 population final complexity가 달라졌다.

### Interpretation
따라서

> **marginal probability path만으로 final mapping complexity가 유일하게 결정되지는 않는다.**

같은 marginals를 지나더라도 실제로 mass를 어떻게 transport하는지가 중요하다.

---

## Exp 5-C — Low-Information Velocity Statistics

### Question
**$M_t$를 버리고 velocity field의 간단한 scalar statistic만 사용해도 충분한가?**

### Result
단순한 trace-based statistic은 실제 complexity ordering과 반대 방향을 보일 수 있었다. Frobenius-based statistic은 이 실험에서는 ordering을 따라갔지만 exact quantity는 아니었다.

### Interpretation
Velocity field의 “크기”만 요약하면 **어느 방향이 contraction하고 expansion하는지, 그리고 그 방향에 얼마나 많은 deformation energy가 누적되어 있는지**를 잃는다.

---

## Exp 5-D — Different Probability Paths

### Question
**그렇다면 probability path 자체를 바꾸면 final complexity도 반드시 달라지는가?**

### Result
테스트한 interpolation family에서는 intermediate marginals가 달라졌지만 final complexity는 사실상 invariant했다.

### Interpretation
Probability path를 바꾼다는 사실만으로 complexity 변화가 보장되지는 않았다.

Exp 5-B와 함께 보면 중요한 구분이 생긴다.

> **Same marginals + different transport → complexity can change.**
> **Different marginals → complexity does not necessarily change.**

따라서 핵심은 probability path 자체의 단순한 형태보다 **그 path가 실제로 어떤 transport/deformation dynamics를 유도하는가**에 있다.

---

# 5. Exp 6 — Direct CFG Mechanism Test

마지막 질문은 처음의 CFG 문제로 돌아가는 것이었다.

> **그렇다면 CFG가 실제로 final mapping complexity를 낮추는 과정도 이 deformation dynamics로 설명할 수 있는가?**

$\omega=1$과 $\omega=2$를 직접 비교했다.

Final complexity는

| CFG scale | Final complexity |
|---:|---:|
| $\omega=1$ | 1.2500 |
| $\omega=2$ | 0.7611 |

이었고,

```math
\mathcal{C}_1^{(2)}
-
\mathcal{C}_1^{(1)}
\approx
-0.48887
```

이었다.

Exact rate를

```math
R_t
=
2\,
\mathbb{E}
\left[
\operatorname{tr}(S_tM_t)
\right]
```

라고 하면,

```math
\mathcal{C}_1^{(2)}
-
\mathcal{C}_1^{(1)}
=
\int_0^1
\left(
R_t^{(2)}-R_t^{(1)}
\right)dt
```

가 numerical integration error 범위에서 정확히 성립했다.

### What Actually Happened?

CFG가 모든 시간에서 더 contractive했던 것은 아니다.

실제로는

1. **초기에는 complexity를 더 강하게 감소시켰고,**
2. **complexity가 다시 증가하기 시작하는 시점을 늦추었으며,**
3. **후반의 re-expansion도 unguided flow보다 약하게 만들었다.**

즉,

> **stronger early complexity decrease + delayed / weaker late re-expansion**

이 누적되어 최종적으로 더 낮은 mapping complexity를 만들었다.

---

# Final Takeaway

처음에는 다음 설명을 기대했다.

**CFG → contraction → low mapping complexity**

Exp 1–4를 통해 contraction은 유용하지만 너무 강한 설명이라는 것을 확인했다.

연구를 dynamics까지 내려가면서 더 근본적인 구조는

**interpolation / transport → $v_t$ → $\nabla v_t$ → $J_t$ → $\mathcal{C}(f)$**

라는 것을 확인했다.

그리고 그 연결의 핵심은

```math
\frac{d}{dt}\mathcal{C}_t
=
2\,
\mathbb{E}
\left[
\operatorname{tr}(S_tM_t)
\right]
```

이다.

따라서 이 toy study에서 얻은 최종적인 해석은 다음과 같다.

> **CFG는 단순히 flow를 uniformly contract시키는 것이 아니다. CFG는 velocity-gradient와 accumulated deformation의 시간적 상호작용을 바꾸고, 그 결과 더 유리한 deformation history를 만들어 final flow-map complexity를 낮춘다.**

또한 Exp 5는 이 complexity가 endpoint나 marginal probability path, 또는 단순한 velocity scalar만으로 결정되지 않는다는 것을 보여준다. **실제로 induced된 transport와 deformation history가 중요하다.**

---

## Scope

이 프로젝트는 **analytic 2D toy setting에서 CFG와 flow-map geometry 사이의 mechanism을 분리해서 살펴본 exploratory study**이다.

여기서 사용한 Jacobian complexity는 geometric mapping-complexity measure이다. 따라서

> **lower Jacobian complexity = lower neural-network learning difficulty**

라고 바로 결론내릴 수는 없다.

실제 finite-capacity neural network에서 낮은 mapping complexity가 approximation 또는 learning difficulty 감소로 이어지는지는 별도의 검증 문제이다.

---

## Main Figures

<p align="center">
  <img src="results/figures/exp06_complexity_comparison.png" width="48%" alt="CFG complexity comparison">
  <img src="results/figures/exp06_rate_comparison.png" width="48%" alt="CFG complexity-rate comparison">
</p>

<p align="center">
  <img src="results/figures/exp06_cfg_rate_difference.png" width="48%" alt="CFG rate difference">
  <img src="results/figures/exp06_accumulated_cfg_effect.png" width="48%" alt="Accumulated CFG effect">
</p>

---

## Repository

```text
cfg-flow-complexity/
├── experiments/    # Exp 1–6 experiment scripts
├── src/            # distributions, velocity fields, flow integration, re-mapping
├── results/        # numerical outputs
└── results/figures # generated figures
```

Example:

```bash
source .venv/bin/activate
python -m experiments.exp03_mapping_complexity
python -m experiments.exp06_cfg_mechanism
```

**Status:** Core exploratory experiments completed through Exp 6.

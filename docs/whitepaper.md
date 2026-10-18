# The mathematics of dsgelab

A technical note on what the package solves and how it solves it. The formulas below are those of the code in `src/dsgelab`. Where a convention common in the literature produces a different number, the text states which of the two is implemented and in which model the two stop coinciding.

## 1. The object

A model is a residual function

\[
f(x_{t+1}, x_t, \varepsilon_{t+1}; \theta) = 0,
\]

with as many equations as variables. The state vector splits into predetermined variables \(k_t\), of dimension \(n_k\), and non-predetermined variables \(u_t\), of dimension \(n_u\),

\[
x_t = \begin{pmatrix} k_t \\ u_t \end{pmatrix}.
\]

The innovation \(\varepsilon_{t+1}\) is unknown at \(t\), with \(E_t \varepsilon_{t+1} = 0\). An autoregressive process is written

\[
\log a_{t+1} = \rho \log a_t + \varepsilon_{a,t+1}.
\]

Dating the innovation at \(t\), as though it were already in the information set when \(u_t\) is chosen, does not change the first-order solution, because certainty equivalence makes the two timings coincide at that order. At second order the risk correction comes out identically zero, since no uncertainty is left within the period. The test that pins down the convention is the lognormal AR(1) of section 7, in which the second-order risk coefficient on technology must be \(\sigma^2\). Under the inverted timing that coefficient is zero, and the rest of the solution still has the appearance of a solved model.

The same function \(f\) is linearised, perturbed at higher order and stacked under perfect foresight. A deterministic path written by hand over a different set of equations has no place in the package, because the two would stop being the same model as soon as one of them was edited.

## 2. Steady state

The deterministic steady state \(\bar x\) solves

\[
f(\bar x, \bar x, 0; \theta) = 0.
\]

Acceptance is decided by the maximum residual, with a default tolerance of \(10^{-11}\). The success flag of `hybr` plays no part in the decision. That method reports failure when its finite-difference Jacobian stops improving, and this includes a starting point that already is the solution, with a residual of order \(10^{-14}\).

Variables marked as positive are solved in logarithms. If \(z_i = \log x_i\), the Newton step is taken in \(z\), and \(x_i = e^{z_i}\) cannot cross zero. A truncation at zero would flatten the residual and stop the solver at the boundary.

Some equations depend on a quantity that is itself a function of the steady state, such as an adjustment cost centred on steady-state wealth. Replacing that quantity with the solution cancels the term and leaves the corresponding margin indeterminate. The quantity is carried instead in a context \(c\), updated by an outer loop

\[
c^{(n+1)} = \mathcal{C}(\bar x^{(n)}, c^{(n)}),
\]

which terminates once \(c\) no longer moves, with a default tolerance of \(10^{-12}\). The converged context is kept with the solution and inherited by the perturbation and by the transition.

## 3. Coordinates of the approximation

Linearisation and the higher orders work in the rescaled deviation

\[
x = \bar x + s \odot \hat y, \qquad
s_i = \begin{cases} |\bar x_i| & \text{if } |\bar x_i| > 10^{-8}, \\ 1 & \text{otherwise.} \end{cases}
\]

The change of variables is affine. The Hessian in deviations is the Hessian in levels multiplied by \(s_i s_j\), and an exact derivative supplied in levels is converted by that multiplication. A genuine log-linearisation, \(x_i = \bar x_i e^{\hat y_i}\), is nonlinear and would not have this property. The branch \(s_i = 1\) is chosen by the steady-state level, and the variable's name plays no part. A mean-zero shifter, with \(\bar x_i = 0\), would be annihilated by a rule that wrote it as \(\bar x_i(1+\hat y_i)\).

The rows of the pencil \((B, A)\) are then divided by the largest absolute entry of each row, and a row whose largest entry is below \(10^{-12}\) is left as it is. This operation multiplies the pencil on the left by a diagonal matrix, an equivalence transformation, so the generalised eigenvalues do not change. The conditioning does change, and it decides whether the Blanchard–Kahn count can be computed when a capital stock of order tens sits beside a mean-zero shifter.

## 4. First-order system

In the rescaled deviation, with \(f(\hat y_{t+1}, \hat y_t, \varepsilon_{t+1}) = 0\),

\[
A = \frac{\partial f}{\partial \hat y_{t+1}}, \quad
B = -\frac{\partial f}{\partial \hat y_t}, \quad
C = -\frac{\partial f}{\partial \varepsilon_{t+1}},
\]

and the canonical form of Klein (2000) is

\[
A\, E_t \hat y_{t+1} = B \hat y_t + C \varepsilon_{t+1}.
\]

\(A\) may be singular. Static equations put zero rows in it, and nothing in the solver inverts it.

If the model was declared in text, the Jacobian is the symbolic derivative, converted by the scale \(s\). Otherwise each column is a central difference with step \(h = 10^{-6}\),

\[
\frac{\partial f}{\partial \hat y_j}
  \approx \frac{f(h e_j) - f(-h e_j)}{2h}.
\]

A forward difference is first-order accurate, and its truncation error enters the eigenvalues from which the determinacy count is read.

## 5. Klein's solution

The pencil \((B, A)\) is brought to the real generalised Schur form, ordered so that the eigenvalues of modulus less than one occupy the leading block (Moler and Stewart, 1973; Klein, 2000). A \(2\times 2\) diagonal block of the real form holds a complex conjugate pair, whose two roots share their modulus, so the ordering by modulus never splits a block between the stable and the unstable subspaces. The solution inverts \(Z_{11}\), solves a linear system in \(T_{11}\) and multiplies by \(S_{11}\), each as a whole matrix, and never reads a block entry by entry. If the real reordering fails, the complex form, with \(1\times 1\) blocks, is used instead. The complex form was the default until October 2026. It was replaced after the package's tests with SciPy 1.18, in which the complex reordering lost up to \(10^{-8}\) on a pencil with a zero root and refused one with several infinite roots, while the real reordering was exact in both cases.

With `ordqz(B, A)`, \(S\) is the Schur form of \(B\) and \(T\) that of \(A\). The generalised eigenvalues are \(\lambda = |\alpha|/|\beta|\), with \(|\beta|\) bounded below by \(10^{-9}\) to avoid dividing by zero at an infinite eigenvalue. An eigenvalue counts as stable when \(\lambda < 1 - 10^{-9}\).

The Blanchard and Kahn (1980) condition is enforced, and it requires the number of stable eigenvalues to equal \(n_k\). Fewer stable roots means that no non-explosive equilibrium exists. More stable roots means a continuum of solutions. In either case \(F\) and \(P\) would have no economic interpretation, and the function raises an error instead of returning them. The margin between the largest stable root and the smallest unstable one is reported separately, by `blanchard_kahn_report`, because a root at \(0.9999\) passes the same test as a root at \(0.8\) and changes classification under a parameter perturbation too small to show up in a coarse sweep.

Partitioning \(Z\), the matrix of Schur vectors, into blocks of \(n_k\) and \(n_u\),

\[
F = Z_{21} Z_{11}^{-1}, \qquad
P = Z_{11}\, T_{11}^{-1} S_{11}\, Z_{11}^{-1},
\]

taking the real part. A singular \(Z_{11}\) means that the stable subspace does not span the states, and no map from states to controls exists. The first-order solution is

\[
u_t = F k_t, \qquad k_{t+1} = P k_t + Q \varepsilon_{t+1}.
\]

### The shock loading

\(Q\) is not read off the QZ transformation. The textbook formula

\[
Q = Z_{11} T_{11}^{-1} (Q_z^{\mathrm H} C)_1
\]

solves \(A x_{t+1} = B x_t + C\varepsilon\) as though every equation held ex post. With the innovation dated \(t+1\), an equation that contains a jump at \(t+1\) holds only in expectation. The jump responds to \(\varepsilon_{t+1}\), and the expectation of that response at \(t\) is zero. Imposing the row ex post adds a false restriction. The system for \(Q\) becomes inconsistent, and the QZ returns a compromise between the true loading and the false restriction.

Two independent AR(1) processes and a Phillips curve are enough to show the effect. The loading on the two processes must be the identity, whereas the textbook formula returns a dense matrix with off-diagonal terms of order \(0.12\). The eigenvalues and \(P\) remain correct. Moments, the variance decomposition and the risk correction, which all go through \(Q\), come out wrong. The error vanishes in the RBC model with log utility and full depreciation, because the Euler equation of that case holds ex post, so a test suite built on that model alone does not detect it.

The code uses the rows that hold ex post, those with no coefficient on a jump at \(t+1\). These are the transition equation of each exogenous process, through which the innovation enters, and the accumulation identity of each endogenous state, which forces the corresponding row of \(Q\) to zero because a stock chosen at \(t\) does not respond to a shock at \(t+1\). On that set of rows,

\[
M Q = C_{\text{rows}}, \qquad
M = A_{\text{rows},\,k} + A_{\text{rows},\,u} F.
\]

The second term is zero by construction on these rows. \(M\) has rank \(n_k\) exactly when every predetermined variable has a law of motion. A lower rank means a state whose law is missing or was written with a jump at \(t+1\). The residual \(\|MQ - C_{\text{rows}}\|_\infty\) is reported and sits at round-off in a well-posed model. Above \(10^{-8}\) times the larger of one and the largest absolute entry of \(C\), the function raises an error. A residual of that size in an equation without a jump at \(t+1\) arises in practice when the only forward term is an exogenous state, as in \(q_t = \beta f(a_{t+1})\). That forward term has to be declared as a jump, with its own defining equation, so that the expectational row is marked as expectational.

## 6. Impulses, moments and decomposition

An impulse to a named innovation sets \(k_0 = Q e\, m\), with \(m\) the size, and propagates \(k_{t+1} = P k_t\). An impulse to a state shifts that state at \(t = 0\), which is the counterfactual on an exogenous level and not on its innovation. An arbitrary impulse vector covers the orthogonalised case, a column of a Cholesky factor of the covariance. With correlated shocks, moving one innovation alone is not an experiment the model admits.

The deviations returned are in the rescaled units, multiplied by 100 when percentages are requested. Variables with \(\bar x_i \neq 0\) come out as proportional deviations, and those with \(\bar x_i = 0\) as absolute deviations.

The unconditional covariance of the states solves the discrete Lyapunov equation

\[
V_k = P V_k P^{\mathsf T} + Q \Sigma Q^{\mathsf T},
\]

and that of the full vector is \(V = J V_k J^{\mathsf T}\), with \(J\) the matrix that stacks \(I\) on top of \(F\). Standard deviations, correlations and autocorrelations are computed from \(V\) and from \(J P^j V_k J^{\mathsf T}\). They are exact for the linear solution. A long simulation reproduces these moments without being their source.

The variance decomposition attributes to each shock the variance obtained with the other shocks switched off, either unconditionally or as the forecast-error variance at horizon \(h\),

\[
V_k^{(m)}(h) = \sum_{j=0}^{h-1} P^j Q \Sigma^{(m)} Q^{\mathsf T} (P^{\mathsf T})^j,
\]

with \(\Sigma^{(m)}\) the matrix that keeps only the \((m,m)\) entry. The share is the ratio between the diagonal of \(J V_k^{(m)} J^{\mathsf T}\) and its sum over \(m\).

The order-of-accuracy test, which the user runs by calling `accuracy()` on a solution, feeds the rules back into the nonlinear function at shrinking displacements. A correct first-order solution leaves a residual of order \(\|\hat y\|^2\), so halving the displacement divides the residual by about four. A system assembled with a transposed block leaves a residual of order \(\|\hat y\|\), and the ratio comes out near two. The test reads this ratio and disregards the size of any single residual, because a wrong system can have a small residual at a small displacement and still leave a first-order error. In an exactly linear model the residual sits at round-off, and the report flags the case instead of deriving an order from machine noise.

## 7. Second order

With the tensors contracted on the states, the rule is

\begin{align*}
u_t &= F k_t + \tfrac12 g_{kk}[k_t, k_t] + \tfrac12 g_{ss}, \\
k_{t+1} &= P k_t + \tfrac12 h_{kk}[k_t, k_t] + \tfrac12 h_{ss} + Q\varepsilon_{t+1}.
\end{align*}

\(g_{kk}\) and \(h_{kk}\) are the second derivatives with respect to the states. \(g_{ss}\) and \(h_{ss}\) are the risk corrections, the terms that first order sets to zero by certainty equivalence. A first-order simulation keeps the mean of the states at the deterministic steady state. Any mean effect it reports comes from the curvature of the static map, without the shift of the rules themselves. With a convex adjustment cost that shift can change the sign of a mean effect as well as its magnitude, and welfare rankings that rest on such an effect can change with it.

The Hessian of each residual with respect to \(w = (\hat y_{t+1}, \hat y_t, \varepsilon)\) uses step \(h = 10^{-4}\). The second difference divides by \(h^2\). Round-off grows as \(\varepsilon_{\mathrm{mach}}/h^2\) while truncation falls as \(h^2\), so the balance sits near \(\varepsilon_{\mathrm{mach}}^{1/4}\). The step \(10^{-6}\), used for the first derivative, loses about six digits here. The mixed partials use a six-point symmetric form that reuses the single-coordinate displacements, at a cost of \(m^2 + m + 1\) evaluations instead of the \(3m^2 - m + 1\) that recomputing the displacements for each pair would take, with \(m = 2n + n_\varepsilon\). When the model was declared in text, the symbolic derivative replaces this Hessian.

Let \(f_1 = A\) and \(f_2 = -B\), already in the row scaling of section 3, and let \(H\) be the Hessian in the same scaling. The derivatives of the composite map with respect to the states are

\[
M = \begin{pmatrix} P \\ FP \end{pmatrix}, \quad
N = \begin{pmatrix} I \\ F \end{pmatrix}, \quad
Z = \begin{pmatrix} M \\ N \\ 0 \end{pmatrix}.
\]

The known term of the quadratic condition is \(K_e = Z^{\mathsf T} H_e Z\). The blocks that multiply the tensors are

\begin{align*}
A_1 &= f_{1,k} + f_{1,u} F, \\
A_2 &= f_{1,u}, \\
A_3 &= f_{2,u},
\end{align*}

and \(W = (h_{kk}, g_{kk})\), of shape \((n, n_k, n_k)\), solves the generalised Sylvester equation

\[
\bar A\, W + \bar B\, W\,(P \otimes P) = -K,
\]

with \(\bar A = [A_1 \mid A_3]\) and \(\bar B = [0 \mid A_2]\). The symmetry of \(W\) in its two state indices is imposed after the solve, and the residual of the equation is reported.

While \(n \cdot n_k^2\) does not exceed 6000, the system is formed and factorised. Above that count, a matrix-free linear operator solved by LGMRES replaces the dense Kronecker product, whose storage grows with the square of the count.

The risk corrections solve a second linear system, in which \(W\) is already known. With \(\Sigma\) the covariance of the innovations and

\[
L = \begin{pmatrix} Q \\ FQ \\ 0 \\ I \end{pmatrix}
\]

the derivative of \(w\) with respect to the innovations,

\[
[A_1 \mid A_2 + A_3]
\begin{pmatrix} h_{ss} \\ g_{ss} \end{pmatrix}
  = -\Big( A_2\,(g_{kk} : Q\Sigma Q^{\mathsf T}) + H : L\Sigma L^{\mathsf T} \Big).
\]

Since \(\Sigma\) enters only this system, a covariance sweep could reuse \(g_{kk}\) and \(h_{kk}\). The implementation solves the Sylvester equation again at every call of `solve_second_order`, and what a sweep can reuse is \(H\), passed in through `hessian=`.

For the process \(\log a' = \rho \log a + \varepsilon\), with \(\mathrm{Var}(\varepsilon) = \sigma^2\), one has \(E_t a' = a^{\rho} \exp(\sigma^2/2)\). In proportional deviation the shift is \(\sigma^2/2\). The rule adds \(h_{ss}/2\), so \(h_{ss}\) on technology comes out as \(\sigma^2\). Where the rule is exact in the model's coordinates, as for capital under full depreciation and log utility, the corresponding correction is zero. A nonzero correction on that capital stock is an error in the risk block.

### Pruning and the ergodic mean

The default simulation uses the pruning of Kim, Kim, Schaumburg and Sims (2008), in which the quadratic part depends on the first-order path alone,

\begin{align*}
k^f_{t+1} &= P k^f_t + Q\varepsilon_{t+1}, \\
k^s_{t+1} &= P k^s_t + \tfrac12 h_{kk}[k^f_t, k^f_t] + \tfrac12 h_{ss}, \\
k_t &= k^f_t + k^s_t.
\end{align*}

Without pruning, the quadratic term is evaluated on a state that already contains the quadratic terms of earlier periods, and a long sample explodes. Iterating a quadratic map produces this whatever the numerical settings, so an unpruned second-order sample cannot support a moment over a long horizon.

The ergodic mean is obtained in closed form from the pruned system instead of as a sample average. With \(V_f\) the first-order covariance of the states,

\[
E[k^s] = (I - P)^{-1}\Big(\tfrac12 h_{kk} : V_f + \tfrac12 h_{ss}\Big),
\]

and the mean of the jumps carries \(F\, E[k^s]\) plus the analogous terms in \(g_{kk}\) and \(g_{ss}\). The shift is of order \(\sigma^2\), whereas the standard deviation of the series is of order \(\sigma\). Separating the two by a sample average, in a persistent process, takes millions of periods. A shorter sample returns Monte Carlo noise with a plausible sign and magnitude.

The second-order accuracy test switches off the risk terms and reads only the quadratic block. The residual must fall as \(\|k\|^3\), so halving the displacement divides the residual by about eight.

## 8. Third order, as far as it is implemented

Second order corrects the mean of the rules. Third order corrects their slope in response to risk, which is what makes a risk premium vary over time. The form written in the code is

\begin{align*}
u_t &= Fk + \tfrac12 g_{kk}[k,k] + \tfrac12 g_{ss}
        + \tfrac16 g_{kkk}[k,k,k] + \tfrac12 g_{ssk}[k], \\
k_{t+1} &= Pk + \tfrac12 h_{kk}[k,k] + \tfrac12 h_{ss}
        + \tfrac16 h_{kkk}[k,k,k] + \tfrac12 h_{ssk}[k] + Q\varepsilon.
\end{align*}

The cubic coefficients \(g_{kkk}\) and \(h_{kkk}\) are computed. They solve the same Sylvester equation with one more Kronecker factor,

\[
\bar A\, W_3 + \bar B\, W_3\, (P \otimes P \otimes P) = -K_3,
\]

and symmetry in the three indices is imposed by averaging over the six permutations. The known term \(K_3\) combines the Hessian contracted with the second derivatives of the map and the third-derivative tensor contracted with \(Z\). The third derivative is a central difference of the Hessian, with an outer step of \(10^{-3}\), larger than the inner one, because differentiating an object already obtained by differencing needs a step that dominates the noise in that object. When the Hessian is symbolic, the difference is free of the round-off that a difference of finite differences accumulates, and it keeps the truncation error of a central difference, of order \(h^2\). The dense third-order tensor has size of order \(n m^3\). In a model with tens of variables it does not fit as a symbolic object, and at that size the route is the Dynare exporter with `order=3`.

\(g_{ssk}\) and \(h_{ssk}\) are computed from the curvature of the rule in \(\varepsilon\), before the expectation is taken. In the log AR(1), \(h_{ssk}[a,a] = \rho\sigma^2\), which is the cross term of \((1+\hat a)^{\rho}\exp(\sigma^2/2)\). The first-order solution still adds \(Q\varepsilon\). The \(\sigma^2 k\) term of the decision rule is assembled, while the tensor in which the innovation is an argument of every monomial, \(y_t = g(y_{t-1}, u_t, \sigma)\), is not.

Third-order simulation uses the three-layer pruning of Andreasen, Fernández-Villaverde and Rubio-Ramírez (2018), where each layer is fed only by the preceding ones. The accuracy test, with risk switched off, requires a residual of order \(\|k\|^4\). Halving the displacement divides the residual by about sixteen.

## 9. Perfect foresight

The deterministic transition from an initial steady state to a terminal one, over \(T\) periods, uses the same \(f\), with \(\varepsilon = 0\) at every date. The unknowns are the \(n_u\) jumps at \(t = 0\) and the full vector at \(t = 1,\ldots,T\). The \(n_k\) states at \(t = 0\) come from the initial state. At \(t = T\), the \(n_u\) jumps are fixed at the terminal state. Fixing the whole terminal vector would overdetermine the system by \(n_k\) equations. The symptom is a Newton iteration that stops converging and looks like a hard problem.

The stacked residual has \(T n + n_u\) entries. The first \(Tn\) are \(f(y_{t+1}, y_t, 0)\) for \(t = 0,\ldots,T-1\). The remaining ones are the difference between the terminal jumps and those of the destination steady state. The Jacobian is block-bidiagonal and is assembled in sparse form. By default it uses the model's exact derivatives when the model was declared in text, and a forward difference otherwise. The accuracy of the converged path is set by the residual tolerance, \(10^{-10}\) by default, and not by the order of the Jacobian. A central difference is available when the residual is noisy.

A surprise is a change in the terminal state with constant parameters. An announced path is a sequence of parameters, or of context values, of length \(T+1\), known from \(t = 0\). The two produce different impacts, and that difference is what the experiment measures. Next-period parameters, when an equation needs them, are carried in the context, which also accepts a path.

## 10. Calibration to targets

Given free parameters \(\theta\) and targets \(m^\star\), calibration solves

\[
\frac{m(\theta) - m^\star}{|m^\star|} = 0
\]

by bounded least squares, with denominator 1 when the target is smaller than \(10^{-12}\) in absolute value. \(m(\theta)\) can be a steady-state variable or a function of one, such as a capital–output ratio. The steady state is solved at each evaluation, warm-started from the previous point. The accepted solution is re-evaluated at the returned point, because the solver's last trial step is not necessarily its best. The default tolerance on the maximum relative error is \(10^{-10}\).

Identification is assessed by `target_jacobian`, a separate call that `calibrate` does not make, from the singular values of the elasticity matrix \(\partial \log m_i / \partial \log \theta_j\), obtained by differencing with step \(10^{-5}\). A rank deficiency means a direction of \(\theta\) to which no target responds. The solver then stops at a point that depends on the start, and no tolerance corrects that.

## 11. First-order conditions

A recursive problem declares the period return \(u(y_t, k_t)\), controls \(y\), states \(k\), equality constraints \(h_j = 0\) and a discount parameter \(\beta\). The period Lagrangian is

\[
\mathcal L_t = u(y_t, k_t) + \sum_j \lambda_{j,t}\, h_j(y_t, k_t, k_{t+1}).
\]

The conditions returned, as text equations ready for the symbolic front end, are

\begin{align*}
\frac{\partial \mathcal L_t}{\partial y_t} &= 0, \\
\frac{\partial \mathcal L_t}{\partial k_{t+1}}
  + \beta\, E_t\!\left[\frac{\partial \mathcal L_{t+1}}{\partial k_{t+1}}\right] &= 0,
\end{align*}

plus the constraints themselves. The second term of the Euler equation is the derivative of \(\mathcal L_t\) with respect to the inherited state, shifted one period ahead. If that shift produces a term at \(t+2\), the problem does not fit in two periods, and the function refuses it and names the state. The derivative is not simplified, since a simplification that divided the residual by a variable would stop being equivalent as that variable approached zero.

Inequalities, complementarity and transversality conditions are not derived. The output contains what was declared and nothing more.

## 12. Translating timing in the export

The exporter accepts only a model declared in text, because a Python residual function cannot be decompiled into equations. The part that is not mechanical reprinting is the timing, which differs between the conventions and has to be translated.

Predetermined stocks, the states whose row of \(Q\) is zero, enter Dynare as `predetermined_variables`. The variable without a lag is the stock available at \(t\), and \(k_{t+1}\) is written \(k(+1)\). In MacroModelling.jl the same stock is \(k[-1]\). Exogenous processes follow Dynare's standard timing, in which the innovation is realised at \(t\) and moves the variable at \(t\), so the equation is written one period back, \(\,a = \rho\, a(-1) + \varepsilon_a\,\). Writing these processes with a lead produces a model that solves and whose risk corrections vanish, for the reason given in section 1.

In Dynare the standard deviations go in the `shocks` block, including the off-diagonal covariances. MacroModelling treats the innovations as standard normal, and the standard deviation enters the equation as a parameter `sig`. Omitting it produces unit variances.

## 13. Estimation, projection, and what remains outside

The first-order solution is a linear state space. On it, the Kalman filter evaluates the likelihood of series in levels, centred on the steady state of the current parameters. The posterior mode is a numerical maximum. Draws come from a random-walk Metropolis–Hastings or from sequential Monte Carlo (SMC). A parameter that violates Blanchard–Kahn contributes a zero likelihood. The Kalman smoother returns the states given the whole sample. There is no particle filter, so the likelihood is always that of the first-order solution.

Projection places the same residual on a Chebyshev grid. The rows of the Jacobian without a forward jump determine the next state at each Gauss–Hermite node, and the other rows are integrated. The policy for the jumps is the polynomial. The tensor grid stops at three states and the quadrature at two shocks.

Heterogeneous agents, in the sense of a distribution over agents, remain outside the package. The map of what each routine does not cover is in `ROADMAP.md`.

A period equilibrium calibrated on a social accounting matrix is not an instance of \(f\). If a model of that kind needed an Euler equation or an announced path, the intertemporal residual is what would come here, while the matrix would stay in the other package. The corresponding section is in the `cgelab` roadmap.

## References

Andreasen, M. M., Fernández-Villaverde, J. and Rubio-Ramírez, J. F. (2018). The pruned state-space system for non-linear DSGE models: theory and empirical applications. *Review of Economic Studies*, 85(1), 1–49.

Blanchard, O. J. and Kahn, C. M. (1980). The solution of linear difference models under rational expectations. *Econometrica*, 48(5), 1305–1311.

Kim, J., Kim, S., Schaumburg, E. and Sims, C. A. (2008). Calculating and using second-order accurate solutions of discrete time dynamic equilibrium models. *Journal of Economic Dynamics and Control*, 32(11), 3397–3414.

Klein, P. (2000). Using the generalized Schur form to solve a multivariate linear rational expectations model. *Journal of Economic Dynamics and Control*, 24(10), 1405–1423.

Moler, C. B. and Stewart, G. W. (1973). An algorithm for generalized matrix eigenvalue problems. *SIAM Journal on Numerical Analysis*, 10(2), 241–256.

Schmitt-Grohé, S. and Uribe, M. (2004). Solving dynamic general equilibrium models using a second-order approximation to the policy function. *Journal of Economic Dynamics and Control*, 28(4), 755–775.

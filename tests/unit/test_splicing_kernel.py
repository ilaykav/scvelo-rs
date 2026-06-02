"""Port of c:/projects/scvelo/tests/core/test_models.py - same scipy.odeint
ground-truth comparison for the splicing dynamics ODE solution, but applied to
our Rust splicing kernel (`splicing_dynamics_eval_kernel`) AS WELL AS scvelo's
`SplicingDynamics`. If both kernels solve the same ODE, both should match
odeint within numerical tolerance.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pytest
from scipy.integrate import odeint
from scvelo.core import SplicingDynamics
from scvelo_rs._scvelo_rs import splicing_dynamics_eval_kernel


def _ode_rhs(y, t, alpha, beta, gamma):
    """The two-state splicing ODE: du/dt = α - β·u; ds/dt = β·u - γ·s."""
    return np.array([alpha - beta * y[0], beta * y[0] - gamma * y[1]])


# ---------------------------------------------------------------------------
# Output-form checks (scvelo asserts shape/dtype contracts on its solver).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,beta,gamma,initial_state",
    [
        (5.0, 0.5, 0.4, [0.0, 1.0]),
        (1.5, 0.8, 0.4, [0.5, 0.3]),
        (0.0, 0.5, 0.4, [1.0, 1.0]),  # alpha=0 (repression branch)
    ],
)
def test_output_shapes_upstream(alpha, beta, gamma, initial_state):
    """Mirror scvelo's `test_output_form` - SplicingDynamics returns (n, 2) ndarray."""
    sd = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma, initial_state=initial_state)
    t = np.linspace(0, 10, 50)
    u, s = sd.get_solution(t, stacked=False)
    assert u.shape == (50,)
    assert s.shape == (50,)


@pytest.mark.parametrize(
    "alpha,beta,gamma,initial_state",
    [
        (5.0, 0.5, 0.4, [0.0, 1.0]),
        (1.5, 0.8, 0.4, [0.5, 0.3]),
        (0.0, 0.5, 0.4, [1.0, 1.0]),
    ],
)
def test_output_shapes_rust(alpha, beta, gamma, initial_state):
    """Same shape contract for our Rust `splicing_dynamics_eval_kernel`."""
    t = np.linspace(0, 10, 50)
    u, s = splicing_dynamics_eval_kernel(t, alpha, beta, gamma, initial_state[0], initial_state[1])
    assert u.shape == (50,)
    assert s.shape == (50,)


# ---------------------------------------------------------------------------
# Solution correctness vs scipy.odeint - scvelo's flagship test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,beta,gamma,initial_state",
    [
        (5.0, 0.5, 0.4, [0, 1]),
    ],
)
def test_solution_matches_odeint_upstream(alpha, beta, gamma, initial_state):
    """Verbatim from scvelo's `test_solution`: SplicingDynamics matches odeint."""
    t = np.linspace(0, 20, 10000)
    sd = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma, initial_state=initial_state)
    exact = sd.get_solution(t)

    numerical = odeint(_ode_rhs, np.array(initial_state), t, args=(alpha, beta, gamma))
    assert np.allclose(numerical, exact)


@pytest.mark.parametrize(
    "alpha,beta,gamma,initial_state",
    [
        (5.0, 0.5, 0.4, [0, 1]),
        (2.0, 0.7, 0.3, [0.1, 0.2]),
        (0.0, 0.5, 0.4, [1.5, 1.2]),  # repression branch
        (3.0, 0.4, 0.4 + 1e-6, [0, 0]),  # gamma ≈ beta (degenerate-ish, scvelo offsets)
    ],
)
def test_solution_matches_odeint_rust(alpha, beta, gamma, initial_state):
    """SAME assertion as scvelo's `test_solution`, but against our Rust kernel.
    Rust's `splicing_dynamics_eval_kernel` should match scipy's odeint."""
    t = np.linspace(0, 20, 10000)
    u, s = splicing_dynamics_eval_kernel(
        t, alpha, beta, gamma, float(initial_state[0]), float(initial_state[1])
    )
    rust = np.column_stack([u, s])

    numerical = odeint(_ode_rhs, np.array(initial_state, dtype=float), t, args=(alpha, beta, gamma))
    assert np.allclose(numerical, rust, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Cross-check: rust kernel and scvelo's SplicingDynamics agree bit-exact.
# (Both implement the same closed-form solution.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,beta,gamma,initial_state",
    [
        (5.0, 0.5, 0.4, [0, 1]),
        (2.0, 0.7, 0.3, [0.1, 0.2]),
        (0.0, 0.5, 0.4, [1.5, 1.2]),
        (1.5, 0.8, 0.4, [0.0, 0.0]),
    ],
)
def test_rust_matches_upstream_bit_exact(alpha, beta, gamma, initial_state):
    """Both implement the same closed-form ODE solution → agree to machine
    precision (1-2 ULP, ~2e-15). Not perfect bit-equality because the two
    formulations evaluate `c * (exps - expu)` in slightly different op order."""
    t = np.linspace(0, 20, 1000)

    sd = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma, initial_state=initial_state)
    u_up, s_up = sd.get_solution(t, stacked=False)
    u_rs, s_rs = splicing_dynamics_eval_kernel(
        t, alpha, beta, gamma, float(initial_state[0]), float(initial_state[1])
    )
    assert np.allclose(u_up, u_rs, atol=1e-14, rtol=1e-14), (
        f"unspliced differs: max_abs={np.abs(u_up - u_rs).max():.3e}"
    )
    assert np.allclose(s_up, s_rs, atol=1e-14, rtol=1e-14), (
        f"spliced differs: max_abs={np.abs(s_up - s_rs).max():.3e}"
    )


# ---------------------------------------------------------------------------
# Steady-state values (scvelo's `test_steady_state_1d`).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,beta,gamma",
    [
        (5.0, 0.5, 0.4),
        (1.5, 0.8, 0.4),
        (10.0, 1.2, 0.6),
    ],
)
def test_steady_state_upstream(alpha, beta, gamma):
    """SplicingDynamics(alpha,beta,gamma).get_steady_states() == (alpha/beta, alpha/gamma)."""
    sd = SplicingDynamics(alpha=alpha, beta=beta, gamma=gamma)
    u_ss, s_ss = sd.get_steady_states(stacked=False)
    assert np.isclose(u_ss, alpha / beta)
    assert np.isclose(s_ss, alpha / gamma)


@pytest.mark.parametrize(
    "alpha,beta,gamma",
    [
        (5.0, 0.5, 0.4),
        (1.5, 0.8, 0.4),
        (10.0, 1.2, 0.6),
    ],
)
def test_steady_state_rust_via_long_time(alpha, beta, gamma):
    """Equivalent: evaluate Rust kernel at large t; should approach (alpha/beta, alpha/gamma)."""
    t = np.array([1e3])  # far enough into the future
    u, s = splicing_dynamics_eval_kernel(t, alpha, beta, gamma, 0.0, 0.0)
    assert np.isclose(u[0], alpha / beta, atol=1e-9)
    assert np.isclose(s[0], alpha / gamma, atol=1e-9)

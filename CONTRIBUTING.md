# Contributing

Thanks for your interest. The project's near-term goals are bug-fix /
test-coverage / docs improvements. A full architecture write-up ships
with the v0.2 documentation - for now, the
[`README`](README.md) and the source layout under `src/` and
`python/scvelo_rs/` are the reference.

## Development setup

```bash
git clone https://github.com/ilaykav/scvelo-rs
cd scvelo-rs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
maturin develop --release
pytest tests/unit tests/integration
```

Requires Rust 1.75+ and Python 3.10+.

## Tests

- `pytest tests/unit` - fast Rust kernel smoke + ODE / splicing equivalence.
- `pytest tests/integration` - side-by-side scVelo vs scvelo-rs (the
  gating tests: `test_bit_exact.py`, `test_equivalence.py`,
  `test_scvelo_pipeline.py`, `test_velocity*.py`, `test_pp.py`,
  `test_compat.py`, `test_edge_cases.py`).
- `pytest -m benchmark` - opt-in speedup benchmarks under
  `tests/benchmarks/` (skipped by default; takes 60+ minutes).
- `python notebooks/01_parity.py` - full-fixture numerical-equivalence
  dashboard against pancreas + dentategyrus.
- `python notebooks/02_benchmarks.py` - wall-time + memory suite.

## Style

- Rust: `cargo fmt`, `cargo clippy`. Comments are sparse - only WHY when
  it's not obvious from the code.
- Python: stay close to scvelo's API surface. The wrapper layer is glue;
  never put compute logic there.

## Pull requests

- Open an issue first if it's larger than a small fix.
- Bit-exact behavior vs scvelo is the contract for the Rust-backed
  functions. If your PR can't preserve that, document why and add a
  per-cell / per-gene tolerance test.
- Add tests under `tests/integration/` for any new Rust-backed module.

## License

By submitting a pull request, you agree to license your contributions
under [BSD-3-Clause](LICENSE).

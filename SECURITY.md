# Security policy

## Supported versions

`scvelo-rs` is on the `0.1.x` line. Security fixes will be applied to the
latest minor and shipped as patch releases. Older minors are not
backported.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports. Email
the maintainer directly:

- **Ilay Kavitzky** - ilay.kavitzky@gmail.com

Reasonable expectations:

- Acknowledgement within 7 days.
- Triage and a plan within 14 days.
- A coordinated disclosure timeline if the fix is non-trivial.

## Scope

`scvelo-rs` is a numerical kernel - it processes user-supplied arrays and
sparse matrices, not network input. The realistic threat surface is:

- Memory-safety bugs in the Rust kernels (panics, out-of-bounds, use-after-free)
- Arithmetic overflows that produce silently wrong results
- Build-supply-chain compromises (Cargo or PyPI dependencies)

Anything in those categories qualifies. Numerical drift vs upstream
scVelo is a **parity** issue, not a security one - please use the
[parity issue template](.github/ISSUE_TEMPLATE/parity_issue.md) for those.

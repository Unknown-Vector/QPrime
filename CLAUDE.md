# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research notebooks that factor an integer n over a fixed prime basis with a variational quantum circuit (QAOA-style ansatz, later VQE). The result is an approximation: the circuit picks exponents e_i so that prod(p_i ** e_i) lands close to n, not an exact factorisation (README).

## Repo shape

- `2.0v1/QPrime_Full.ipynb` is the current end-to-end pipeline. Everything under `1.0V1-10/` is history kept as the experiment record.
- `1.0V1-10/v1` to `v10` are chronological. The part of a notebook name after `v X.Y -` names the one thing it changes from the matching `Control` notebook (`_optTNC_gtol` sweeps TNC's `gtol`, `No Penalty` drops the objective penalty, and so on). Each sweep ends with `np.save("<experiment>", ll)`, writing a loss array beside the notebook; those `.npy` results were never committed. v8 to v10 are a gradient-validation thread (below).
- `v1/Newbie Ansatz Calculation.ipynb` and `Pro Ansatz Calculation.ipynb` are sympy-only: the symbolic Jacobian of a two-qubit ansatz, no Qiskit.
- `.ipynb_checkpoints/` folders are committed Jupyter autosaves. Edit the parent notebook, never the checkpoint.
- There is no build, lint, test or dependency file. Every notebook is code cells only (no markdown), so reading one means dumping its cells:

```
python -c "import json,sys; [print(''.join(c['source'])) for c in json.load(open(sys.argv[1], encoding='utf-8'))['cells']]" "2.0v1\QPrime_Full.ipynb"
```

## Environment

- Every notebook's kernel metadata records Python 3.8.5, and all of them use the legacy Qiskit API: `from qiskit import *` for `Aer`, `execute`, `assemble`; `qiskit.opflow` (`I, X, Z`, `StateFn`, `CircuitStateFn`, `Gradient`); `qiskit.algorithms` (`VQE` and the optimizers); `qiskit.utils.QuantumInstance`; `qiskit.circuit.qpy_serialization`. IBM's Qiskit 1.0 migration guide lists `qiskit.opflow`, `qiskit.algorithms`, `qiskit.utils`, `qiskit.execute` and `qiskit.Aer` as removed in 1.0, so running these needs a pre-1.0 Qiskit pinned next to Python 3.8. The saved output in 2.0 already shows a DeprecationWarning for `circuit += other`, so the original environment sat late in that line.
- On this machine the pyenv Python is 3.12.10 with no qiskit installed; nothing here runs until that environment exists.
- Every 1.0 notebook calls `back.set_options(device='GPU')` on `aer_simulator`, which needs the GPU build of qiskit-aer; delete the line to run on CPU. 2.0 does not set it.
- Initial parameters come from `random.uniform` in both lines and the only seed lines (in 2.0) are commented out, so two runs of the same notebook differ.

## How the 2.0 pipeline works (`QPrime_Full.ipynb`)

1. **Encoding.** `primes` and `qbit_coeff` (binary weights `[1, 2, 4, 8, ...]`) are truncated to `no_primes` and `no_dits`. Prime i gets `no_dits` qubits, its exponent is `sum_j qbit_coeff[j] * bit_ij`, and the estimate is `n_hat = prod_i primes[i] ** e_i`. Qubit count is `no_primes * no_dits`.
2. **Hamiltonian.** `hamiltonian(n, primes, qbit_coeff)` builds the opflow operator `(ln(n) * I - X) ** 2` with `X = sum_i ln(p_i) * sum_j qbit_coeff[j] * (I - Z_ij) / 2`, so each computational basis state has energy `(ln n - ln n_hat) ** 2` and the ground state is the best factorisation the basis allows.
3. **Optimise.** `TwoLocal(rotation 'ry', entanglement 'cx', reps=layer)` under `VQE` with `ADAM` on the statevector simulator, starting from a random point in `[0, 2*pi)`.
4. **Read out.** The ansatz is rebuilt with `result.optimal_point`, sampled with 2048 shots on `aer_simulator`, and `composite_builder` decodes the most frequent bitstring: chunk i of `len(qbit_coeff)` characters belongs to `primes[i]`, and character j in a chunk carries `qbit_coeff[j]`. It prints `n_hat` and `n - n_hat`.

The committed run (n = 4482709258, primes `[2, 3, 5]`, 4 dits each, 12 qubits) ended at `n_hat = 3888000000`; a gap of that size is what "approximate" means here.

## How the 1.0 notebooks differ

- One qubit per prime (exponent 0 or 1). `hamiltonian(n, primes)` is `ln(n)**2 * I - ln(n) * X + X**2 / 4` with `X = sum_i ln(p_i) * (I - Z_i)`.
- The ansatz is hand-built in `generate_circuit` (earlier `generate_ZZcircuit`): Hadamard on every qubit, then per layer `rz(gamma1_i * 2 * ln(n / sqrt(prod p)) * ln p_i)` on each qubit, `rzz(gamma2_jk * ln p_j * ln p_k)` on every pair, and an `rx(beta_i)` mixer. Parameters per layer: `2 * len(primes) + C(len(primes), 2)`.
- The objective moved from samples (`obj_function`: most frequent bitstring, then `abs(n - n_hat)` plus an optional overshoot penalty) to the exact expectation `<psi|H|psi>` of a saved statevector (`obj_function2`, v10).
- Optimisers use the old `optimizer.optimize(num_vars, objective_function, initial_point, gradient_function)` call. `_convert_to_gradient_function` wraps opflow `Gradient(grad_method="param_shift")` of `~StateFn(H) @ CircuitStateFn(circuit)` into that `gradient_function`.
- **Gradient validation (v8 to v10).** `_convert_to_gradient_function2` hand-rolls the parameter-shift rule on statevectors and multiplies each shift by the gate's angle coefficient from `param_coeffs`, because the gate angle is `coeff * theta`. It is compared against Qiskit's gradient and an external Java implementation (`v8/gradient_java.npy` against `gradient_python.npy` in `v8/graident_test.ipynb`); `v9/Qcir_current.qpy` is the circuit exported for that comparison. The Java code is not in this repo.

## Qubit-order gotcha

Opflow tensor products (`Z ^ I ^ I`) and Qiskit count bitstrings both put the highest qubit index leftmost, while `circuit.rz(theta, i)` addresses qubit `i` directly. The 1.0 notebooks reconcile this by indexing `build_primes = primes[::-1]` inside `X_generator` and `obj_function` while `generate_circuit` keeps `primes[i]` on qubit `i`; change those together or the Hamiltonian, circuit and decoder disagree about which qubit is which prime. 2.0 needs no reversal: `genZ` and `composite_builder` both count positions from the left, and `TwoLocal` has no prime-specific gates.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `Unknown-Vector/QPrime`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage labels, unchanged: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.

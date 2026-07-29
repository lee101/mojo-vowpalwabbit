# mojo-vowpalwabbit

`mojo-vowpalwabbit` is a Mojo implementation of Vowpal Wabbit's hashed online
linear learner, exposed to Python through the familiar `Workspace` and
`Example` API.

This is a focused port, not a binding to the C++ library. It implements the
compute-heavy scalar gradient-descent path and VW-compatible MurmurHash3
feature hashing. The Python package is named `mojo_vowpalwabbit`, so it can be
installed beside the real `vowpalwabbit` package for differential testing.
For the covered subset, changing the import is normally sufficient.

```python
from mojo_vowpalwabbit import Workspace

with Workspace(quiet=True) as vw:
    vw.learn("1 |user age:0.4 premium |item price:0.2")
    vw.learn("-1 |user age:0.8 |item price:0.9")
    prediction = vw.predict("|user age:0.5 premium |item price:0.3")
    print(prediction)
```

## Coverage

The implementation covers:

| area | supported behavior |
| --- | --- |
| Python API | `Workspace`, `Example`, `SimpleLabel`, `NamespaceId`, `ExampleNamespace`, `pyvw`, `vw` |
| online learning | VW's default adaptive, normalized, invariant update; regular `--sgd`; explicit `--adaptive`, `--normalized`, and `--invariant` combinations |
| losses | `squared` / `classic`, `logistic`, `hinge` |
| schedules | `--learning_rate`, `--power_t`, `--initial_t`, example importance weights |
| hashing | VW MurmurHash3 x86-32, `--hash strings`, `--hash all`, numeric feature semantics, `-b` / `--bit_precision` |
| features | named and anonymous namespaces, namespace scaling, numeric feature values, implicit constant, `--noconstant`, `--constant` |
| interactions | two-namespace quadratic interactions with `-q` / `--quadratic`, including self-interactions |
| predictions | dynamic VW label clipping, explicit min/max prediction, identity and logistic links |
| model access | `get_weight`, `set_weight`, `get_weight_from_name`, `num_weights`, `save`, `initial_regressor` |
| batch extension | `pack`, `learn_many`, `predict_many`, `hash_features` |

The parity suite compares hashes, predictions, per-feature weights, schedules,
losses, update modes, interactions, constants, links, and parser behavior
directly with conda-forge `vowpalwabbit` 9.10. The checked results normally
match bit for bit and otherwise differ only by float32 roundoff.

The following are deliberately not covered:

- contextual bandits, cost-sensitive and multiclass reductions, search, LDA,
  neural networks, FTRL, BFGS, and distributed learning;
- multiline examples, daemon operation, file parsing, cache files, and the VW
  command-line executable;
- L1/L2 regularization, audit/invert-hash output, feature dictionaries, cubic
  and wildcard interactions;
- the upstream binary model format. `save` and `initial_regressor` round-trip
  this port's NumPy-based model files, not C++ VW model files.

Options for an unsupported learner or regularizer raise `NotImplementedError`
instead of silently selecting a different algorithm.

## Install and verify

The repository carries a pinned Mojo nightly and the upstream package used by
the parity suite.

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles the single Mojo unit into
`dist/libmojo-vowpalwabbit.so`. Python also rebuilds a missing or stale library
on first use. Set `MOJO_VOWPALWABBIT_LIB` to load a prebuilt library from a
different location.

## Packed online learning

Calling `learn` with VW strings is the most compatible interface. Common ASCII
linear examples use the native Mojo parser, while compatibility cases and
quadratic expansion use Python. For repeated prediction, multiple passes, or
already-materialized data, pack examples once:

```python
from mojo_vowpalwabbit import Workspace

lines = [
    "1 |a x:2 y |b z:0.5",
    "-1 |a x:-1 |b z",
]

vw = Workspace("--quiet -q ab")
batch = vw.pack(lines)
first_pass_predictions = vw.learn_many(batch, return_predictions=True)
second_pass_predictions = vw.learn_many(batch, return_predictions=True)
scores = vw.predict_many(batch)
```

`learn_many` remains online: examples are predicted and updated sequentially
inside Mojo in their original order. Packing only removes repeated Python
parsing, allocation, and FFI calls.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86_64, Python 3.12.13. Times compare the same examples against
conda-forge `vowpalwabbit` 9.10.

| case | mojo-vowpalwabbit | vowpalwabbit 9.10 | result |
| --- | ---: | ---: | ---: |
| hash 500k feature names | 219.06 ms | 281.39 ms | 1.28x faster |
| predict 30k packed, 16 features | 12.78 ms | 277.01 ms | 21.67x faster |
| learn 50k packed, 12 features | 42.59 ms | 132.61 ms | 3.11x faster |
| learn 20k VW strings, 12 features | 1310.68 ms | 267.60 ms | 4.90x slower |
| learn 15k packed with 8x8 quadratic | 21.01 ms | 45.73 ms | 2.18x faster |

The packed rows exclude preprocessing on both sides: this port receives a
reusable CSR batch and upstream receives reusable parsed `Example` objects.
They expose the intended compute path and avoid tens of thousands of Python
to native calls. The string row includes parsing. ASCII, non-interaction
`learn(str)` calls now parse in Mojo into reusable NumPy scratch buffers,
eliminating per-feature FFI calls and per-row CSR allocation. Non-ASCII input,
quadratic interactions, unusual numeric syntax, and rows over the scratch
capacity retain the Python compatibility path. Batched hashing avoids a
redundant input copy and is at parity with upstream in this run.

No GPU or parallel hashing path is included. Online learning preserves input
order and has a sequential update dependency between rows.

## How it works

Python parses dictionaries and the compatibility cases for VW text, applies
the upstream hashing rules, and packs examples into CSR-like storage. The
common ASCII scalar-learning path parses directly in Mojo:

- feature indices are contiguous `uint64`;
- values and model state are `float32`, matching upstream VW;
- example offsets are contiguous `int64`;
- the dense table has `2**bit_precision` slots, selected with a bit mask.

The learner keeps separate float32 arrays for the linear weight, adaptive
gradient accumulator, and per-feature normalizer. This is logically the same
state as VW's strided weight table without forcing unused state into every
mode. Quadratic indices use VW's FNV-like interaction mix before the usual
table mask.

The ctypes boundary passes NumPy buffer addresses as C `int64` values. The
exported Mojo functions rebuild
`UnsafePointer[..., AnyOrigin[mut=True]]` values internally. A batch crosses
the FFI once, then Mojo performs sparse dot products, loss updates, adaptive
state changes, and weight updates without allocating. Python validates array
dtypes, contiguity, dimensions, offsets, finite float32 values, and batch
ownership before native code dereferences a packed buffer. Sparse prediction
loads contiguous feature indices and values at the host SIMD width, gathers
weights, reduces each vector, and finishes with a scalar remainder loop.
Single-row text scratch storage is retained across calls. All exports live in
one compilation unit to keep the Mojo shared-library build simple.

## License

MIT

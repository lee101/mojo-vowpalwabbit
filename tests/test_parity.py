from __future__ import annotations

import ctypes

import numpy as np
import pytest
from vowpalwabbit import Workspace as UpstreamWorkspace

import mojo_vowpalwabbit as mvw
from mojo_vowpalwabbit._lib import addr, bytes_array, lib


HASH_CASES = [
    ("", 0),
    ("a", 0),
    ("ab", 0),
    ("user", 0),
    ("ü", 0),
]


@pytest.mark.parametrize("namespace,seed", HASH_CASES)
def test_namespace_hash_parity(namespace, seed):
    upstream = UpstreamWorkspace("--quiet -b 20")
    ours = mvw.Workspace("--quiet -b 20")
    assert ours.hash_space(namespace) == upstream.hash_space(namespace)
    upstream.finish()


@pytest.mark.parametrize("hash_mode", ["strings", "all"])
def test_feature_hash_parity(hash_mode):
    args = f"--quiet -b 20 --hash {hash_mode}"
    upstream, ours = UpstreamWorkspace(args), mvw.Workspace(args)
    for namespace in (" ", "a", "ab", "user", "ü"):
        upstream_seed = upstream.hash_space(namespace)
        ours_seed = ours.hash_space(namespace)
        for feature in ("x", "feature", "5", "hello world", "ü", "  42  "):
            assert ours.hash_feature(feature, ours_seed) == upstream.hash_feature(
                feature, upstream_seed
            )
    upstream.finish()


def test_published_murmur_hash_vectors():
    vectors = {
        b"t": 3397902157,
        b"te": 3988319771,
        b"tes": 196677210,
        b"test": 3127628307,
        b"tested": 2247989476,
    }
    for data, expected in vectors.items():
        storage, length = bytes_array(data)
        assert lib().mvw_uniform_hash(addr(storage), length, 0) == expected


def test_hash_many_matches_scalar_and_upstream():
    names = [f"feature_{i}" for i in range(1000)] + ["5", "42", "ü", "x\0y"]
    ours = mvw.Workspace("--quiet -b 19")
    upstream = UpstreamWorkspace("--quiet -b 19")
    seed = ours.hash_space("a")
    got = ours.hash_features(names, seed)
    expected = np.array(
        [
            upstream.hash_feature(name, upstream.hash_space("a"))
            for name in names
        ],
        dtype=np.uint64,
    )
    assert np.array_equal(got, expected)
    upstream.finish()


def test_hash_many_parallel_path_matches_scalar():
    count = 1_000_003
    ours = mvw.Workspace("--quiet -b 19")
    seed = ours.hash_space("a")
    expected = ours.hash_feature("parallel", seed)
    got = ours.hash_features(["parallel"] * count, seed)
    assert got.shape == (count,)
    assert np.all(got == expected)


def test_text_parser_feature_parity():
    text = "1 2 0.25 'tag |user:0.5 age:2 country=NZ | words 5"
    upstream, ours = UpstreamWorkspace("--quiet"), mvw.Workspace("--quiet")
    upstream_ex, ours_ex = upstream.parse(text), ours.parse(text)
    upstream_features = list(upstream_ex.iter_features())
    ours_features = list(ours_ex.iter_features())
    assert ours_ex.get_simplelabel_label() == upstream_ex.get_simplelabel_label()
    assert ours_ex.get_simplelabel_weight() == upstream_ex.get_simplelabel_weight()
    assert ours_ex.get_simplelabel_initial() == upstream_ex.get_simplelabel_initial()
    assert ours_features == pytest.approx(upstream_features)
    upstream.finish_example(upstream_ex)
    upstream.finish()


def _learn_and_compare(args: str, lines: list[str], probe: str):
    upstream, ours = UpstreamWorkspace(args), mvw.Workspace(args)
    upstream_predictions = []
    ours_predictions = []
    for line in lines:
        upstream_ex, ours_ex = upstream.parse(line), ours.parse(line)
        upstream.learn(upstream_ex)
        ours.learn(ours_ex)
        upstream_predictions.append(upstream_ex.get_simplelabel_prediction())
        ours_predictions.append(ours_ex.get_simplelabel_prediction())
        upstream.finish_example(upstream_ex)
    assert ours_predictions == pytest.approx(upstream_predictions, abs=2e-6)
    assert ours.predict(probe) == pytest.approx(upstream.predict(probe), abs=2e-6)
    for namespace, feature in (("a", "x"), ("a", "y"), ("b", "z")):
        assert ours.get_weight_from_name(feature, namespace) == pytest.approx(
            upstream.get_weight_from_name(feature, namespace), abs=2e-6
        )
    upstream.finish()


TRAIN_LINES = [
    "1 |a x:2 y |b z:0.5",
    "-1 2 |a x:-1 q:3 |b z",
    "0.5 |a y:2 |b z:-2",
    "1 |a x:0.2 y:4 |b z:2",
]


def test_default_adaptive_normalized_invariant_parity():
    _learn_and_compare("--quiet", TRAIN_LINES, "|a x:2 y |b z:.5")


@pytest.mark.parametrize("loss", ["squared", "logistic", "hinge"])
def test_regular_sgd_loss_parity(loss):
    _learn_and_compare(
        f"--quiet --sgd --learning_rate 0.1 --power_t 0 --noconstant "
        f"--min_prediction -100 --max_prediction 100 --loss_function {loss}",
        TRAIN_LINES,
        "|a x:2 y |b z:.5",
    )


@pytest.mark.parametrize("power_t", [0.0, 0.3, 0.5, 1.0])
def test_sgd_schedule_and_importance_parity(power_t):
    _learn_and_compare(
        f"--quiet --sgd --learning_rate .1 --power_t {power_t} "
        "--noconstant --min_prediction -100 --max_prediction 100 "
        "--loss_function classic",
        ["1 |a x", "1 |a x", "1 2 |a x", "-1 .5 |a x"],
        "|a x",
    )


@pytest.mark.parametrize(
    "mode",
    [
        "--adaptive",
        "--normalized --invariant",
        "--adaptive --normalized --invariant",
        "--invariant",
    ],
)
def test_explicit_update_modes_parity(mode):
    _learn_and_compare(
        f"--quiet {mode} --learning_rate .2 --power_t .3",
        TRAIN_LINES,
        "|a x y |b z",
    )


@pytest.mark.parametrize("quadratic", ["ab", "aa"])
def test_quadratic_interaction_parity(quadratic):
    _learn_and_compare(
        f"--quiet -q {quadratic}",
        TRAIN_LINES,
        "|a x:2 y |b z:.5",
    )


def test_constant_and_no_constant_parity():
    for args in (
        "--quiet --constant 2 --min_prediction -100 --max_prediction 100",
        "--quiet --noconstant --min_prediction -100 --max_prediction 100",
    ):
        upstream, ours = UpstreamWorkspace(args), mvw.Workspace(args)
        assert ours.predict("|") == upstream.predict("|")
        upstream.learn("1 |a x")
        ours.learn("1 |a x")
        assert ours.predict("|a x") == pytest.approx(upstream.predict("|a x"), abs=2e-6)
        upstream.finish()


def test_logistic_link_parity():
    _learn_and_compare(
        "--quiet --loss_function logistic --link logistic",
        ["1 |a x", "-1 |a y", "1 |a x:2"],
        "|a x y",
    )


def test_example_dict_and_mutation_api():
    workspace = mvw.Workspace("--quiet --noconstant")
    example = workspace.example({"a": {"x": 2.0, "y": -1.0}})
    example.push_feature("b", "z", 3.0)
    assert example.num_namespaces() == 2
    assert example.num_features_in("a") == 2
    assert example.sum_feat_sq("a") == 5.0
    assert list(example["b"].iter_features())[0][1] == 3.0
    assert example.pop_feature("b")
    assert not example.pop_feature("b")
    example.set_label_string("1 2 0.5")
    label = example.get_label()
    assert (label.label, label.weight, label.initial) == (1.0, 2.0, 0.5)


def test_batch_learning_equals_sequential_learning():
    lines = TRAIN_LINES * 20
    batch = mvw.Workspace("--quiet -q ab")
    sequential = mvw.Workspace("--quiet -q ab")
    packed = batch.pack(lines)
    predictions = batch.learn_many(packed, return_predictions=True)
    sequential_predictions = []
    for line in lines:
        example = sequential.parse(line)
        sequential.learn(example)
        sequential_predictions.append(example.get_simplelabel_prediction())
    assert predictions == pytest.approx(sequential_predictions, abs=2e-6)
    assert np.array_equal(batch.weights, sequential.weights)
    assert np.array_equal(batch._accumulators, sequential._accumulators)


def test_batch_prediction_equals_scalar_prediction():
    workspace = mvw.Workspace("--quiet")
    workspace.learn_many(TRAIN_LINES * 4)
    examples = ["|a x", "|a y:2 |b z", "|a q:-1"]
    batch = workspace.predict_many(examples)
    scalar = np.array([workspace.predict(example) for example in examples])
    assert batch == pytest.approx(scalar)


def test_batch_prediction_simd_full_width_and_tail():
    workspace = mvw.Workspace(
        "--quiet --noconstant -b 20 --min_prediction -1e9 --max_prediction 1e9"
    )
    lines = [
        "|a " + " ".join(f"x{i}:{i + 1}" for i in range(count))
        for count in (7, 8, 9, 17)
    ]
    packed = workspace.pack(lines)
    used = np.unique(packed.indices & workspace.mask)
    workspace.weights[used] = (
        (used % 97).astype(np.float32) - np.float32(48.0)
    ) / np.float32(31.0)
    expected = []
    for row in range(len(lines)):
        start, end = packed.offsets[row : row + 2]
        total = np.float32(0.0)
        for index, value in zip(
            packed.indices[start:end], packed.values[start:end]
        ):
            total += workspace.weights[index & workspace.mask] * value
        expected.append(total)
    assert workspace.predict_many(packed) == pytest.approx(expected, abs=2e-5)


@pytest.mark.parametrize(
    "args",
    [
        "--quiet",
        "--quiet --sgd --learning_rate .1 --power_t 0 --noconstant",
        "--quiet --hash all --loss_function logistic",
    ],
)
def test_native_text_learning_matches_packed_path(args):
    lines = [
        "1 2 .25 'tag |a:0.5 x:2 42:-3e-1 | y:.75",
        "-1 |a x:-.2 z:4 |b q:1e-2",
        ".5 .25 |long_namespace value:3.5",
    ]
    direct = mvw.Workspace(args)
    packed = mvw.Workspace(args)
    for line in lines:
        direct.learn(line)
        packed.learn_many([packed.parse(line)])
    assert direct._state == pytest.approx(packed._state)
    assert direct._bounds == pytest.approx(packed._bounds)
    np.testing.assert_allclose(direct.weights, packed.weights, rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        direct._accumulators, packed._accumulators, rtol=0, atol=2e-6
    )


def test_native_text_fused_simd_full_width_and_tail():
    direct = mvw.Workspace("--quiet --noconstant")
    packed = mvw.Workspace("--quiet --noconstant")
    for count in (3, 4, 5, 9):
        line = "1 |a " + " ".join(
            f"x{i}:{(i + 1) / 7}" for i in range(count)
        )
        direct.learn(line)
        packed.learn_many([packed.parse(line)])
    assert direct._state == pytest.approx(packed._state)
    np.testing.assert_allclose(direct.weights, packed.weights, rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        direct._accumulators, packed._accumulators, rtol=0, atol=2e-6
    )


@pytest.mark.parametrize(
    "line",
    [
        "1 |a ü:2",
        "1 |a " + " ".join(f"x{i}:{i + 1}" for i in range(260)),
    ],
)
def test_text_learning_compatibility_fallbacks(line):
    direct = mvw.Workspace("--quiet")
    packed = mvw.Workspace("--quiet")
    direct.learn(line)
    packed.learn_many([packed.parse(line)])
    np.testing.assert_allclose(direct.weights, packed.weights, rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        direct._accumulators, packed._accumulators, rtol=0, atol=2e-6
    )


def test_save_and_reload(tmp_path):
    args = "--quiet -q ab"
    workspace = mvw.Workspace(args)
    workspace.learn_many(TRAIN_LINES * 3)
    before = workspace.predict("|a x y |b z")
    path = tmp_path / "model.vw"
    workspace.save(path)
    loaded = mvw.Workspace(args, initial_regressor=str(path))
    assert loaded.predict("|a x y |b z") == before
    assert np.array_equal(loaded.weights, workspace.weights)
    assert np.array_equal(loaded._accumulators, workspace._accumulators)


def test_weight_access_and_context_manager():
    with mvw.Workspace("--quiet -b 10") as workspace:
        index = workspace.hash_feature("x", workspace.hash_space("a"))
        workspace.set_weight(index, 0, 1.25)
        assert workspace.get_weight(index) == 1.25
        assert workspace.num_weights() == 1024
        assert workspace.get_label_type() is mvw.LabelType.SIMPLE
        assert workspace.get_prediction_type() is mvw.PredictionType.SCALAR
    assert workspace.finished


@pytest.mark.parametrize("option", ["--cb 4", "--oaa 3", "--l2 .1", "--nn 4"])
def test_unsupported_reductions_fail_explicitly(option):
    with pytest.raises(NotImplementedError):
        mvw.Workspace(f"--quiet {option}")


def test_ctypes_signature_uses_float32():
    assert ctypes.sizeof(ctypes.c_float) == np.dtype(np.float32).itemsize


def test_hash_many_rejects_mismatched_or_narrowed_seeds():
    workspace = mvw.Workspace("--quiet")
    with pytest.raises(ValueError, match="one entry"):
        workspace.hash_features(["x", "y"], [1])
    with pytest.raises(ValueError, match="uint32"):
        workspace.hash_features(["x"], [2**32])
    with pytest.raises(ValueError, match="uint32"):
        workspace.hash_features(["x"], -1)


def test_packed_batch_rejects_invalid_ffi_layout():
    workspace = mvw.Workspace("--quiet")
    batch = workspace.pack(["1 |a x", "-1 |a y"])
    batch.offsets[1] = 100
    with pytest.raises(ValueError, match="offsets"):
        workspace.learn_many(batch)


def test_packed_batch_rejects_replaced_dtype_and_wrong_workspace():
    workspace = mvw.Workspace("--quiet")
    batch = workspace.pack(["1 |a x"])
    batch.values = batch.values.astype(np.float64)
    with pytest.raises(TypeError, match="float32"):
        workspace.predict_many(batch)

    other = mvw.Workspace("--quiet")
    with pytest.raises(ValueError, match="Workspace"):
        other.predict_many(workspace.pack(["|a x"]))

    workspace.weights = workspace.weights.astype(np.float64)
    with pytest.raises(TypeError, match="model weights"):
        workspace.predict("|a x")


@pytest.mark.parametrize("line", ["1e999 |a x", "1 |a x:1e999"])
def test_native_text_path_rejects_float32_overflow(line):
    workspace = mvw.Workspace("--quiet")
    with pytest.raises(ValueError, match="float32"):
        workspace.learn(line)


def test_predict_allows_missing_labels_but_learn_does_not():
    workspace = mvw.Workspace("--quiet")
    batch = workspace.pack(["|a x"])
    assert workspace.predict_many(batch).shape == (1,)
    with pytest.raises(ValueError, match="label"):
        workspace.learn_many(batch)


def test_initial_t_schedule_parity():
    _learn_and_compare(
        "--quiet --sgd --learning_rate .1 --power_t .5 --initial_t 4",
        TRAIN_LINES,
        "|a x y |b z",
    )


def test_documented_api_aliases():
    assert mvw.vw is mvw.Workspace
    assert mvw.pyvw.Workspace is mvw.Workspace


@pytest.mark.parametrize(
    "args",
    [
        "--learning_rate 1e999",
        "--power_t nan",
        "--logistic_min 1 --logistic_max 1",
        "--min_prediction 2 --max_prediction 1",
    ],
)
def test_invalid_float_options_fail_before_ffi(args):
    with pytest.raises(ValueError):
        mvw.Workspace(f"--quiet {args}")

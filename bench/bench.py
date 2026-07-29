"""Benchmarks against vowpalwabbit 9.10 on identical examples."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
from vowpalwabbit import Workspace as UpstreamWorkspace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_vowpalwabbit as mvw  # noqa: E402


def best_time(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def machine() -> str:
    model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model}; {platform.system()} {platform.machine()}; Python {platform.python_version()}"


def make_lines(count: int, features: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    result = []
    for row in range(count):
        label = 1 if row % 2 else -1
        values = rng.normal(size=features)
        split = features // 2
        left = " ".join(f"a{i}:{values[i]:.7g}" for i in range(split))
        right = " ".join(f"b{i}:{values[i]:.7g}" for i in range(split, features))
        result.append(f"{label} |a {left} |b {right}")
    return result


def row(name: str, mojo_seconds: float, upstream_seconds: float) -> str:
    ratio = upstream_seconds / mojo_seconds
    result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
    return (
        f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
        f"{upstream_seconds * 1e3:.2f} ms | {result} |"
    )


def main() -> None:
    results: list[tuple[str, float, float]] = []

    names = [f"feature_{i}" for i in range(500_000)]
    ours_hash = mvw.Workspace("--quiet -b 20")
    upstream_hash = UpstreamWorkspace("--quiet -b 20")
    ours_seed = ours_hash.hash_space("a")
    upstream_seed = upstream_hash.hash_space("a")
    ours_hash.hash_features(names[:10], ours_seed)
    results.append(
        (
            "hash 500k feature names",
            best_time(lambda: ours_hash.hash_features(names, ours_seed)),
            best_time(
                lambda: [
                    upstream_hash.hash_feature(name, upstream_seed) for name in names
                ]
            ),
        )
    )

    predict_lines = make_lines(30_000, 16)
    ours_predict = mvw.Workspace("--quiet")
    upstream_predict = UpstreamWorkspace("--quiet")
    ours_predict.learn_many(predict_lines[:1000])
    for line in predict_lines[:1000]:
        upstream_predict.learn(line)
    ours_examples = [ours_predict.parse(line) for line in predict_lines]
    ours_predict_batch = ours_predict.pack(ours_examples)
    upstream_examples = [upstream_predict.parse(line) for line in predict_lines]
    results.append(
        (
            "predict 30k packed, 16 features",
            best_time(lambda: ours_predict.predict_many(ours_predict_batch)),
            best_time(lambda: [upstream_predict.predict(ex) for ex in upstream_examples]),
        )
    )

    train_lines = make_lines(50_000, 12, seed=1)
    ours_train = mvw.Workspace("--quiet")
    upstream_train = UpstreamWorkspace("--quiet")
    ours_train_examples = [ours_train.parse(line) for line in train_lines]
    ours_train_batch = ours_train.pack(ours_train_examples)
    upstream_train_examples = [upstream_train.parse(line) for line in train_lines]
    start = time.perf_counter()
    ours_train.learn_many(ours_train_batch)
    ours_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for example in upstream_train_examples:
        upstream_train.learn(example)
    upstream_seconds = time.perf_counter() - start
    results.append(
        ("learn 50k packed, 12 features", ours_seconds, upstream_seconds)
    )

    stream_lines = make_lines(20_000, 12, seed=2)
    ours_stream = mvw.Workspace("--quiet")
    upstream_stream = UpstreamWorkspace("--quiet")
    start = time.perf_counter()
    for line in stream_lines:
        ours_stream.learn(line)
    ours_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for line in stream_lines:
        upstream_stream.learn(line)
    upstream_seconds = time.perf_counter() - start
    results.append(("learn 20k VW strings, 12 features", ours_seconds, upstream_seconds))

    interaction_lines = make_lines(15_000, 16, seed=3)
    ours_interactions = mvw.Workspace("--quiet -q ab")
    upstream_interactions = UpstreamWorkspace("--quiet -q ab")
    ours_interaction_examples = [
        ours_interactions.parse(line) for line in interaction_lines
    ]
    ours_interaction_batch = ours_interactions.pack(ours_interaction_examples)
    upstream_interaction_examples = [
        upstream_interactions.parse(line) for line in interaction_lines
    ]
    start = time.perf_counter()
    ours_interactions.learn_many(ours_interaction_batch)
    ours_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for example in upstream_interaction_examples:
        upstream_interactions.learn(example)
    upstream_seconds = time.perf_counter() - start
    results.append(
        ("learn 15k packed with 8x8 quadratic", ours_seconds, upstream_seconds)
    )

    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-vowpalwabbit | vowpalwabbit 9.10 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_seconds, upstream_seconds in results:
        print(row(name, mojo_seconds, upstream_seconds))

    for example in upstream_examples:
        upstream_predict.finish_example(example)
    for example in upstream_train_examples:
        upstream_train.finish_example(example)
    for example in upstream_interaction_examples:
        upstream_interactions.finish_example(example)
    upstream_hash.finish()
    upstream_predict.finish()
    upstream_train.finish()
    upstream_stream.finish()
    upstream_interactions.finish()


if __name__ == "__main__":
    main()

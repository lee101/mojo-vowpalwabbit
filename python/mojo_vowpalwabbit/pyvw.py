"""The covered subset of Vowpal Wabbit's ``pyvw`` interface."""

from __future__ import annotations

import enum
import math
import os
import shlex
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ._lib import addr, bytes_array, lib

CONSTANT_HASH = 11_650_396
FNV_PRIME = 16_777_619
F32_MAX = float(np.finfo(np.float32).max)


def _float32(value, name: str) -> float:
    """Convert without accepting NaN, infinity, or an overflowing narrowing."""
    result = float(value)
    if not math.isfinite(result) or abs(result) > F32_MAX:
        raise ValueError(f"{name} must be a finite float32 value")
    return result


class LabelType(enum.IntEnum):
    SIMPLE = 1


class PredictionType(enum.IntEnum):
    SCALAR = 0


class SimpleLabel:
    def __init__(
        self,
        label: float = 0.0,
        weight: float = 1.0,
        initial: float = 0.0,
        prediction: float = 0.0,
    ):
        self.label = label
        self.weight = weight
        self.initial = initial
        self.prediction = prediction

    @staticmethod
    def from_example(ex: "Example") -> "SimpleLabel":
        return SimpleLabel(ex.label or 0.0, ex.weight, ex.initial, ex._prediction)

    def __str__(self) -> str:
        result = str(self.label)
        if self.weight != 1.0:
            result += ":" + str(self.weight)
        return result


class NamespaceId:
    def __init__(self, ex: "Example", id: int | str):
        self.full: str | None = None
        if isinstance(id, int):
            names = list(ex._namespaces)
            if id < 0 or id >= len(names):
                raise IndexError(f"namespace {id} out of bounds")
            self.id = id
            self.full = names[id]
            self.ns = self.full[0] if self.full else " "
        elif isinstance(id, str):
            self.id = None
            self.full = id or " "
            self.ns = self.full[0]
        else:
            raise TypeError("namespace must be an integer or string")
        self.ord_ns = ord(self.ns)


class ExampleNamespace:
    def __init__(self, ex: "Example", ns: NamespaceId):
        self.ex = ex
        self.ns = ns

    def __len__(self) -> int:
        return self.num_features_in()

    def __getitem__(self, i: int) -> tuple[int, float]:
        return self.ex.feature(self.ns, i), self.ex.feature_weight(self.ns, i)

    def num_features_in(self) -> int:
        return self.ex.num_features_in(self.ns)

    def iter_features(self) -> Iterator[tuple[int, float]]:
        yield from (self[i] for i in range(len(self)))

    def push_feature(self, feature: str | int, v: float = 1.0):
        self.ex.push_feature(self.ns, feature, v)

    def push_features(self, feature_list):
        self.ex.push_features(self.ns, feature_list)

    def pop_feature(self) -> bool:
        return self.ex.pop_feature(self.ns)


class Example:
    def __init__(
        self,
        vw: "Workspace",
        initStringOrDictOrRawExample=None,
        labelType: LabelType | int | None = None,
    ):
        if labelType not in (None, 0, LabelType.SIMPLE, 1):
            raise NotImplementedError("only simple scalar labels are covered")
        self.vw = vw
        self.labelType = LabelType.SIMPLE
        self.label: float | None = None
        self.weight = 1.0
        self.initial = 0.0
        self.tag = b""
        self._prediction = 0.0
        self._namespaces: OrderedDict[str, list[tuple[int, float]]] = OrderedDict()
        self.setup_done = False
        self.finished = False
        if isinstance(initStringOrDictOrRawExample, str):
            self._parse(initStringOrDictOrRawExample)
            self.setup_done = True
        elif isinstance(initStringOrDictOrRawExample, dict):
            self._from_dict(initStringOrDictOrRawExample)
        elif initStringOrDictOrRawExample is not None:
            raise TypeError("expecting a string, dict, or None")

    def _parse(self, line: str) -> None:
        pieces = line.replace("\r", "").split("|")
        header = pieces[0].strip()
        if header:
            tokens = header.split()
            try:
                self.label = float(tokens[0])
                if len(tokens) > 1:
                    self.weight = float(tokens[1])
                if len(tokens) > 2:
                    self.initial = float(tokens[2])
                if len(tokens) > 3:
                    self.tag = tokens[3].lstrip("'").encode()
            except ValueError:
                self.tag = header.lstrip("'").encode()
        for segment in pieces[1:]:
            anonymous = not segment or segment[0].isspace()
            tokens = segment.split()
            if anonymous:
                namespace, feature_tokens, scale = " ", tokens, 1.0
            elif tokens:
                namespace_token = tokens[0]
                if ":" in namespace_token:
                    namespace, scale_text = namespace_token.split(":", 1)
                    scale = float(scale_text)
                else:
                    namespace, scale = namespace_token, 1.0
                feature_tokens = tokens[1:]
            else:
                continue
            bucket = self._namespaces.setdefault(namespace or " ", [])
            seed = self.vw.hash_space(namespace)
            for token in feature_tokens:
                if ":" in token:
                    name, value_text = token.rsplit(":", 1)
                    if not name:
                        continue
                    value = float(value_text)
                else:
                    name, value = token, 1.0
                bucket.append((self.vw.hash_feature(name, seed), float(value * scale)))

    def _from_dict(self, data: dict) -> None:
        for namespace, features in data.items():
            if isinstance(features, dict):
                features = list(features.items())
            self.push_features(namespace, features)

    def get_ns(self, id: NamespaceId | int | str) -> NamespaceId:
        return id if isinstance(id, NamespaceId) else NamespaceId(self, id)

    def __getitem__(self, id: NamespaceId | int | str) -> ExampleNamespace:
        return ExampleNamespace(self, self.get_ns(id))

    def num_namespaces(self) -> int:
        return len(self._namespaces) + (0 if self.vw.noconstant else 1)

    def namespace(self, i: int) -> int:
        names = list(self._namespaces)
        if i < len(names):
            return ord(names[i][0] if names[i] else " ")
        if i == len(names) and not self.vw.noconstant:
            return 128
        raise IndexError(i)

    def num_features_in(self, ns: NamespaceId | int | str) -> int:
        ns = self.get_ns(ns)
        if ns.ord_ns == 128:
            return 0 if self.vw.noconstant else 1
        return len(self._features_for_namespace(ns))

    def _features_for_namespace(self, ns: NamespaceId) -> list[tuple[int, float]]:
        if ns.full in self._namespaces:
            return self._namespaces[ns.full]
        for name, features in self._namespaces.items():
            if (name[0] if name else " ") == ns.ns:
                return features
        return []

    def feature(self, ns: NamespaceId | int | str, i: int) -> int:
        ns = self.get_ns(ns)
        if ns.ord_ns == 128:
            if i != 0 or self.vw.noconstant:
                raise IndexError(i)
            return CONSTANT_HASH
        return self._features_for_namespace(ns)[i][0]

    def feature_weight(self, ns: NamespaceId | int | str, i: int) -> float:
        ns = self.get_ns(ns)
        if ns.ord_ns == 128:
            if i != 0 or self.vw.noconstant:
                raise IndexError(i)
            return 1.0
        return self._features_for_namespace(ns)[i][1]

    def iter_features(self) -> Iterator[tuple[int, float]]:
        for features in self._namespaces.values():
            yield from features
        if not self.vw.noconstant:
            yield CONSTANT_HASH, 1.0

    def get_feature_id(
        self,
        ns: NamespaceId | str | int,
        feature: str | int,
        ns_hash: int | None = None,
    ) -> int:
        if isinstance(feature, int):
            return feature
        ns = self.get_ns(ns)
        return self.vw.hash_feature(
            feature,
            self.vw.hash_space(ns.full or ns.ns) if ns_hash is None else ns_hash,
        )

    def ensure_namespace_exists(self, ns: NamespaceId | str | int):
        ns = self.get_ns(ns)
        self._namespaces.setdefault(ns.full or ns.ns, [])

    def push_feature(
        self,
        ns: NamespaceId | str | int,
        feature: str | int,
        v: float = 1.0,
        ns_hash: int | None = None,
    ) -> None:
        ns = self.get_ns(ns)
        name = ns.full or ns.ns
        feature_id = self.get_feature_id(ns, feature, ns_hash)
        if not 0 <= feature_id <= np.iinfo(np.uint64).max:
            raise ValueError("feature index must fit in uint64")
        self._namespaces.setdefault(name, []).append(
            (feature_id, _float32(v, "feature value"))
        )
        self.setup_done = False

    def push_hashed_feature(
        self, ns: NamespaceId | str | int, f: int, v: float = 1.0
    ) -> None:
        self.push_feature(ns, int(f), v)

    def push_features(self, ns, featureList) -> None:
        for item in featureList:
            if isinstance(item, tuple):
                self.push_feature(ns, item[0], item[1])
            else:
                self.push_feature(ns, item)

    def pop_feature(self, ns) -> bool:
        features = self._features_for_namespace(self.get_ns(ns))
        if not features:
            return False
        features.pop()
        self.setup_done = False
        return True

    def push_namespace(self, ns) -> None:
        self.ensure_namespace_exists(ns)

    def pop_namespace(self) -> bool:
        if not self._namespaces:
            return False
        self._namespaces.popitem()
        return True

    def sum_feat_sq(self, ns) -> float:
        return sum(value * value for _, value in self._features_for_namespace(self.get_ns(ns)))

    def set_label_string(self, string: str) -> None:
        tokens = string.split()
        self.label = _float32(tokens[0], "label")
        self.weight = _float32(tokens[1], "importance") if len(tokens) > 1 else 1.0
        self.initial = _float32(tokens[2], "initial prediction") if len(tokens) > 2 else 0.0

    def get_simplelabel_label(self) -> float:
        return float(self.label if self.label is not None else math.inf)

    def get_simplelabel_weight(self) -> float:
        return self.weight

    def get_simplelabel_initial(self) -> float:
        return self.initial

    def get_simplelabel_prediction(self) -> float:
        return self._prediction

    def get_prediction(self, prediction_type=None) -> float:
        if prediction_type not in (None, 0, PredictionType.SCALAR):
            raise NotImplementedError("only scalar predictions are covered")
        return self._prediction

    def get_label(self, label_class=None) -> SimpleLabel:
        return SimpleLabel.from_example(self)

    def setup_example(self):
        self.setup_done = True

    def unsetup_example(self):
        self.setup_done = False

    def learn(self):
        return self.vw.learn(self)

    def _expanded(self) -> tuple[list[int], list[float]]:
        indices: list[int] = []
        values: list[float] = []
        by_char: dict[str, list[tuple[int, float]]] = {}
        for name, features in self._namespaces.items():
            indices.extend(index for index, _ in features)
            values.extend(value for _, value in features)
            by_char.setdefault(name[0] if name else " ", []).extend(features)
        for left_ns, right_ns in self.vw.quadratics:
            left = by_char.get(left_ns, ())
            right = by_char.get(right_ns, ())
            for i, (left_index, left_value) in enumerate(left):
                start = i if left_ns == right_ns and not self.vw.permutations else 0
                for right_index, right_value in right[start:]:
                    indices.append(
                        ((FNV_PRIME * left_index) ^ right_index) & self.vw.mask
                    )
                    values.append(left_value * right_value)
        if not self.vw.noconstant:
            indices.append(CONSTANT_HASH & self.vw.mask)
            values.append(1.0)
        return indices, values


class ExampleBatch:
    """Reusable CSR storage for one-call prediction or sequential online updates."""

    def __init__(
        self,
        workspace: "Workspace",
        examples: list[Example],
        indices: np.ndarray,
        values: np.ndarray,
        offsets: np.ndarray,
        initial: np.ndarray,
    ):
        self.workspace = workspace
        self.examples = examples
        self.indices = indices
        self.values = values
        self.offsets = offsets
        self.initial = initial
        self.has_missing_label = any(ex.label is None for ex in examples)
        self.labels = np.ascontiguousarray(
            [np.nan if ex.label is None else ex.label for ex in examples],
            dtype=np.float32,
        )
        self.importance = np.ascontiguousarray(
            [ex.weight for ex in examples], dtype=np.float32
        )
    def __len__(self) -> int:
        return len(self.examples)

    def _validated_addresses(
        self, workspace: "Workspace", *, require_labels: bool
    ) -> tuple[int, ...]:
        if self.workspace is not workspace:
            raise ValueError("a packed batch can only be used with the Workspace that packed it")
        arrays = (
            ("indices", self.indices, np.dtype(np.uint64)),
            ("values", self.values, np.dtype(np.float32)),
            ("offsets", self.offsets, np.dtype(np.int64)),
            ("initial", self.initial, np.dtype(np.float32)),
            ("labels", self.labels, np.dtype(np.float32)),
            ("importance", self.importance, np.dtype(np.float32)),
        )
        for name, array, dtype in arrays:
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 1
                or array.dtype != dtype
                or not array.flags.c_contiguous
            ):
                raise TypeError(f"batch {name} must be a contiguous 1-D {dtype} array")
        rows = len(self.examples)
        if len(self.offsets) != rows + 1:
            raise ValueError("batch offsets length must equal row count plus one")
        if len(self.initial) != rows or len(self.labels) != rows or len(self.importance) != rows:
            raise ValueError("batch row arrays must match the example count")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.indices):
            raise ValueError("batch offsets must span the feature arrays")
        if len(self.values) != len(self.indices) or np.any(self.offsets[1:] < self.offsets[:-1]):
            raise ValueError("batch feature arrays and offsets are inconsistent")
        if not (
            np.all(np.isfinite(self.values))
            and np.all(np.isfinite(self.initial))
            and np.all(np.isfinite(self.importance))
        ):
            raise ValueError("batch floating-point arrays must contain only finite values")
        if require_labels and not np.all(np.isfinite(self.labels)):
            raise ValueError("batch labels must contain only finite values")
        return tuple(addr(array) for _, array, _ in arrays)


def _command_tokens(arg_str, arg_list, kwargs) -> list[str]:
    tokens: list[str] = []
    if arg_str:
        tokens.extend(shlex.split(arg_str))
    if arg_list:
        tokens.extend(arg_list)
    for key, value in kwargs.items():
        name = ("-" if len(key) == 1 else "--") + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                tokens.append(name)
        elif isinstance(value, (list, tuple)):
            for item in value:
                tokens.extend((name, str(item)))
        else:
            tokens.extend((name, str(value)))
    return tokens


class Workspace:
    def __init__(
        self,
        arg_str: str | None = None,
        enable_logging: bool = False,
        arg_list: list[str] | None = None,
        **kw,
    ):
        self.enable_logging = enable_logging
        self.finished = False
        self._logs: list[str] = []
        self._tokens = _command_tokens(arg_str, arg_list, kw)
        options, supplied = self._parse_options(self._tokens)
        self.bits = int(options.get("bit_precision", 18))
        if not 1 <= self.bits <= 30:
            raise ValueError("bit precision must be between 1 and 30")
        self.mask = (1 << self.bits) - 1
        self.hash_all = options.get("hash", "strings") == "all"
        self._namespace_hash_cache: dict[str, int] = {}
        self._feature_hash_cache: dict[tuple[int, str], int] = {}
        self.noconstant = bool(options.get("noconstant", False))
        self.permutations = bool(options.get("permutations", False))
        self.quadratics = self._parse_interactions(options.get("quadratic", []))
        self.loss_name = str(options.get("loss_function", "squared"))
        if self.loss_name == "classic":
            self.loss_name = "squared"
        if self.loss_name not in {"squared", "logistic", "hinge"}:
            raise NotImplementedError(f"loss {self.loss_name!r} is not covered")
        self.loss_code = {"squared": 0, "logistic": 1, "hinge": 2}[self.loss_name]
        self.logistic_min = _float32(options.get("logistic_min", -1.0), "logistic minimum")
        self.logistic_max = _float32(options.get("logistic_max", 1.0), "logistic maximum")
        if self.logistic_min >= self.logistic_max:
            raise ValueError("logistic minimum must be less than logistic maximum")
        self.link_name = str(options.get("link", "identity"))
        if self.link_name not in {"identity", "logistic"}:
            raise NotImplementedError(f"link {self.link_name!r} is not covered")
        self.link_code = int(self.link_name == "logistic")

        nondefault = any(name in supplied for name in ("sgd", "adaptive", "invariant", "normalized"))
        if nondefault:
            self.adaptive = "adaptive" in supplied
            self.normalized = "normalized" in supplied
            self.invariant = "invariant" in supplied
        else:
            self.adaptive = self.normalized = self.invariant = True
        eta_supplied = "learning_rate" in supplied
        self.eta = _float32(options.get("learning_rate", 0.5), "learning rate")
        if nondefault and not eta_supplied and not (self.adaptive and self.normalized):
            self.eta = 10.0
        self.power_t = _float32(options.get("power_t", 0.5), "power_t")
        initial_t = options.get("initial_t")
        if initial_t is None:
            initial_t = 1.0 if not self.adaptive and not self.normalized else 0.0
        initial_t = _float32(initial_t, "initial_t")
        if nondefault and not self.adaptive and not self.normalized:
            self.eta *= initial_t**self.power_t
            self.eta = _float32(self.eta, "effective learning rate")

        size = 1 << self.bits
        self.weights = np.zeros(size, dtype=np.float32)
        self._accumulators = np.zeros(size, dtype=np.float32)
        self._normalizers = np.zeros(size, dtype=np.float32)
        self._state = np.array([initial_t, initial_t, initial_t], dtype=np.float64)
        if "min_prediction" in supplied or "max_prediction" in supplied:
            self._dynamic_bounds = False
            lo = _float32(options.get("min_prediction", -F32_MAX), "minimum prediction")
            hi = _float32(options.get("max_prediction", F32_MAX), "maximum prediction")
            if lo > hi:
                raise ValueError("minimum prediction must not exceed maximum prediction")
        else:
            self._dynamic_bounds = True
            lo = hi = 0.0
        self._bounds = np.array([lo, hi], dtype=np.float32)
        self._text_indices = np.empty(256, dtype=np.uint64)
        self._text_values = np.empty(256, dtype=np.float32)
        self._text_offsets = np.array([0, 0], dtype=np.int64)
        self._text_labels = np.empty(1, dtype=np.float32)
        self._text_importance = np.empty(1, dtype=np.float32)
        self._text_initial = np.empty(1, dtype=np.float32)
        self._text_prediction = np.empty(1, dtype=np.float32)
        self._text_addresses = tuple(
            addr(array)
            for array in (
                self._text_indices,
                self._text_values,
                self._text_offsets,
                self._text_labels,
                self._text_importance,
                self._text_initial,
                self._text_prediction,
            )
        )
        constant = _float32(options.get("constant", 0.0), "constant")
        self.weights[CONSTANT_HASH & self.mask] = constant
        model = options.get("initial_regressor")
        if model:
            self._load(model)
        self._model_arrays = (
            self.weights,
            self._accumulators,
            self._normalizers,
            self._state,
            self._bounds,
        )
        self._model_addresses = tuple(addr(array) for array in self._model_arrays)
        self._learn_text_native = lib().mvw_learn_text

    def _validated_model_addresses(self) -> tuple[int, ...]:
        if (
            self.weights is self._model_arrays[0]
            and self._accumulators is self._model_arrays[1]
            and self._normalizers is self._model_arrays[2]
            and self._state is self._model_arrays[3]
            and self._bounds is self._model_arrays[4]
        ):
            return self._model_addresses
        arrays = (
            ("weights", self.weights, np.dtype(np.float32), 1 << self.bits),
            ("accumulators", self._accumulators, np.dtype(np.float32), 1 << self.bits),
            ("normalizers", self._normalizers, np.dtype(np.float32), 1 << self.bits),
            ("state", self._state, np.dtype(np.float64), 3),
            ("bounds", self._bounds, np.dtype(np.float32), 2),
        )
        for name, array, dtype, length in arrays:
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 1
                or len(array) != length
                or array.dtype != dtype
                or not array.flags.c_contiguous
            ):
                raise TypeError(
                    f"model {name} must be a contiguous 1-D {dtype} array of length {length}"
                )
        self._model_arrays = tuple(array for _, array, _, _ in arrays)
        self._model_addresses = tuple(addr(array) for array in self._model_arrays)
        return self._model_addresses

    @staticmethod
    def _parse_options(tokens: list[str]) -> tuple[dict, set[str]]:
        aliases = {
            "b": "bit_precision",
            "bits": "bit_precision",
            "bit_precision": "bit_precision",
            "l": "learning_rate",
            "learning_rate": "learning_rate",
            "q": "quadratic",
            "interactions": "quadratic",
            "i": "initial_regressor",
            "initial_regressor": "initial_regressor",
        }
        values = {
            "bit_precision",
            "learning_rate",
            "power_t",
            "initial_t",
            "loss_function",
            "link",
            "min_prediction",
            "max_prediction",
            "constant",
            "hash",
            "quadratic",
            "logistic_min",
            "logistic_max",
            "initial_regressor",
        }
        flags = {
            "quiet",
            "no_stdin",
            "sgd",
            "adaptive",
            "normalized",
            "invariant",
            "noconstant",
            "permutations",
            "audit",
        }
        unsupported = {
            "l1",
            "l2",
            "cb",
            "cb_explore",
            "cb_explore_adf",
            "oaa",
            "csoaa",
            "multilabel_oaa",
            "search",
            "lda",
            "nn",
            "rank",
            "ftrl",
            "bfgs",
        }
        result: dict = {}
        supplied: set[str] = set()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if not token.startswith("-"):
                raise ValueError(f"unexpected argument {token!r}")
            raw = token.lstrip("-").replace("-", "_")
            attached = None
            if "=" in raw:
                raw, attached = raw.split("=", 1)
            name = aliases.get(raw, raw)
            if name in unsupported:
                raise NotImplementedError(f"option --{raw} is outside the covered scalar GD subset")
            if name in flags:
                result[name] = True
                supplied.add(name)
                i += 1
                continue
            if name not in values:
                raise NotImplementedError(f"option --{raw} is not covered")
            if attached is None:
                i += 1
                if i >= len(tokens):
                    raise ValueError(f"option --{raw} requires a value")
                attached = tokens[i]
            if name == "quadratic":
                result.setdefault(name, []).append(attached)
            else:
                result[name] = attached
            supplied.add(name)
            i += 1
        return result, supplied

    @staticmethod
    def _parse_interactions(values: list[str]) -> list[tuple[str, str]]:
        result = []
        for value in values:
            if len(value) != 2:
                raise NotImplementedError("only two-namespace quadratic interactions are covered")
            result.append((value[0], value[1]))
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()

    def hash_space(self, namespace: str | NamespaceId) -> int:
        if isinstance(namespace, NamespaceId):
            namespace = namespace.full or namespace.ns
        cached = self._namespace_hash_cache.get(namespace)
        if cached is not None:
            return cached
        storage, length = bytes_array(namespace)
        result = int(
            lib().mvw_hash_string(addr(storage), length, 0, int(self.hash_all))
        )
        self._namespace_hash_cache[namespace] = result
        return result

    def hash_feature(self, feature: str, namespace_hash: int) -> int:
        key = (int(namespace_hash), feature)
        cached = self._feature_hash_cache.get(key)
        if cached is not None:
            return cached
        storage, length = bytes_array(feature)
        result = int(
            lib().mvw_hash_string(
                addr(storage), length, int(namespace_hash), int(self.hash_all)
            )
        ) & self.mask
        self._feature_hash_cache[key] = result
        return result

    def hash_features(self, features: Iterable[str], namespace_hashes=0) -> np.ndarray:
        names = features if isinstance(features, (list, tuple)) else list(features)
        joined_text = "\0".join(names) + "\0"
        delimited = int(joined_text.isascii())
        if delimited:
            joined = joined_text.encode("ascii")
            lengths = (len(name) + 1 for name in names)
        else:
            encoded = [name.encode("utf-8") for name in names]
            joined = b"".join(encoded)
            lengths = (len(item) for item in encoded)
        data = np.frombuffer(joined if joined else b"\0", dtype=np.uint8)
        offsets = np.empty(len(names) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(
            np.fromiter(lengths, dtype=np.int64),
            out=offsets[1:],
        )
        if np.isscalar(namespace_hashes):
            seed = int(namespace_hashes)
            if not 0 <= seed <= np.iinfo(np.uint32).max:
                raise ValueError("namespace hash must fit in uint32")
            seeds = np.array([seed], dtype=np.uint32)
            shared_seed = 1
        else:
            raw_seeds = list(namespace_hashes)
            if len(raw_seeds) != len(names):
                raise ValueError("namespace_hashes must have one entry per feature")
            if any(not 0 <= int(seed) <= np.iinfo(np.uint32).max for seed in raw_seeds):
                raise ValueError("namespace hashes must fit in uint32")
            seeds = np.ascontiguousarray(raw_seeds, dtype=np.uint32)
            shared_seed = 0
        result = np.empty(len(names), dtype=np.uint64)
        if names:
            lib().mvw_hash_many(
                addr(data),
                addr(offsets),
                addr(seeds),
                addr(result),
                len(names),
                self.mask,
                int(self.hash_all),
                shared_seed,
                delimited,
            )
        return result

    def parse(self, str_ex, labelType=None):
        if isinstance(str_ex, Example):
            return str_ex
        if isinstance(str_ex, list):
            raise NotImplementedError("multiline learners are outside the covered subset")
        if not isinstance(str_ex, str):
            raise TypeError("a string or Example is required")
        lines = [line for line in str_ex.strip().splitlines() if line.strip()]
        if len(lines) != 1:
            raise TypeError("the covered scalar learner expects one line")
        return Example(self, lines[0], labelType)

    def example(self, stringOrDict=None, labelType=None) -> Example:
        return Example(self, stringOrDict, labelType)

    def _coerce_examples(self, examples) -> tuple[list[Example], bool]:
        if isinstance(examples, (str, Example, dict)):
            values = [examples]
            scalar = True
        else:
            values = list(examples)
            scalar = False
        converted = [
            item
            if isinstance(item, Example)
            else self.example(item) if isinstance(item, dict) else self.parse(item)
            for item in values
        ]
        return converted, scalar

    def _csr(self, examples: list[Example]):
        all_indices: list[int] = []
        all_values: list[float] = []
        offsets = np.zeros(len(examples) + 1, dtype=np.int64)
        for i, example in enumerate(examples):
            indices, values = example._expanded()
            for value in values:
                _float32(value, "feature value")
            _float32(example.initial, "initial prediction")
            all_indices.extend(indices)
            all_values.extend(values)
            offsets[i + 1] = len(all_indices)
        indices_array = np.ascontiguousarray(all_indices, dtype=np.uint64)
        values_array = np.ascontiguousarray(all_values, dtype=np.float32)
        initial = np.ascontiguousarray([ex.initial for ex in examples], dtype=np.float32)
        return indices_array, values_array, offsets, initial

    def pack(self, examples) -> ExampleBatch:
        converted, _ = self._coerce_examples(examples)
        for example in converted:
            if example.vw is not self:
                raise ValueError("examples can only be packed by their originating Workspace")
            if example.label is not None:
                _float32(example.label, "label")
            _float32(example.weight, "importance")
        return ExampleBatch(self, converted, *self._csr(converted))

    def predict_many(self, examples) -> np.ndarray:
        if isinstance(examples, ExampleBatch):
            batch = examples
        else:
            batch = self.pack(examples)
        converted = batch.examples
        result = np.empty(len(converted), dtype=np.float32)
        if converted:
            model_addresses = self._validated_model_addresses()
            indices_addr, values_addr, offsets_addr, initial_addr, _, _ = (
                batch._validated_addresses(self, require_labels=False)
            )
            lib().mvw_predict_many(
                model_addresses[0],
                indices_addr,
                values_addr,
                offsets_addr,
                initial_addr,
                addr(result),
                len(converted),
                self.mask,
                float(self._bounds[0]),
                float(self._bounds[1]),
                self.link_code,
            )
        for example, prediction in zip(converted, result):
            example._prediction = float(prediction)
        return result

    def predict(self, ec, prediction_type=None):
        if prediction_type not in (None, 0, PredictionType.SCALAR):
            raise NotImplementedError("only scalar predictions are covered")
        converted, scalar = self._coerce_examples(ec)
        result = self.predict_many(converted)
        return float(result[0]) if scalar else result

    def learn_many(self, examples, return_predictions: bool = False):
        if isinstance(examples, ExampleBatch):
            batch = examples
        else:
            batch = self.pack(examples)
        converted = batch.examples
        if batch.has_missing_label:
            raise ValueError("learn requires a label on every example")
        predictions = np.empty(len(converted), dtype=np.float32)
        if converted:
            model_addresses = self._validated_model_addresses()
            (
                indices_addr,
                values_addr,
                offsets_addr,
                initial_addr,
                labels_addr,
                importance_addr,
            ) = batch._validated_addresses(self, require_labels=True)
            lib().mvw_learn_many(
                model_addresses[0],
                model_addresses[1],
                model_addresses[2],
                indices_addr,
                values_addr,
                offsets_addr,
                labels_addr,
                importance_addr,
                initial_addr,
                addr(predictions),
                model_addresses[3],
                model_addresses[4],
                len(converted),
                self.mask,
                self.eta,
                self.power_t,
                self.logistic_min,
                self.logistic_max,
                self.loss_code,
                int(self.adaptive),
                int(self.normalized),
                int(self.invariant),
                int(self._dynamic_bounds),
                self.link_code,
            )
        for example, prediction in zip(converted, predictions):
            example._prediction = float(prediction)
            example.setup_done = True
        return predictions if return_predictions else None

    def _learn_text(self, line: str) -> bool:
        if (
            not line
            or "\n" in line
            or "\r" in line
            or self.quadratics
        ):
            return False
        try:
            encoded = line.encode("ascii")
        except UnicodeEncodeError:
            return False
        (
            indices_addr,
            values_addr,
            offsets_addr,
            labels_addr,
            importance_addr,
            initial_addr,
            prediction_addr,
        ) = self._text_addresses
        model_addresses = self._validated_model_addresses()
        count = self._learn_text_native(
            encoded,
            len(encoded),
            model_addresses[0],
            model_addresses[1],
            model_addresses[2],
            indices_addr,
            values_addr,
            offsets_addr,
            labels_addr,
            importance_addr,
            initial_addr,
            prediction_addr,
            model_addresses[3],
            model_addresses[4],
            len(self._text_indices),
            self.mask,
            self.eta,
            self.power_t,
            self.logistic_min,
            self.logistic_max,
            self.loss_code,
            int(self.adaptive),
            int(self.normalized),
            int(self.invariant),
            int(self._dynamic_bounds),
            self.link_code,
            int(self.hash_all),
            int(self.noconstant),
        )
        if count == -3:
            raise ValueError("label, importance, initial prediction, and features must fit in float32")
        if count < 0:
            return False
        return True

    def learn(self, ec) -> None:
        if isinstance(ec, str) and self._learn_text(ec):
            return
        converted, _ = self._coerce_examples(ec)
        self.learn_many(converted)

    def num_weights(self) -> int:
        return len(self.weights)

    def get_stride(self) -> int:
        return 4 if self.adaptive and self.normalized else 1

    def get_weight(self, index, offset=0) -> float:
        if offset != 0:
            if offset == 1:
                return float(self._accumulators[int(index) & self.mask])
            if offset == 2:
                return float(self._normalizers[int(index) & self.mask])
            raise IndexError(offset)
        return float(self.weights[int(index) & self.mask])

    def set_weight(self, index, offset, value) -> None:
        slot = int(index) & self.mask
        value = _float32(value, "weight")
        if offset == 0:
            self.weights[slot] = value
        elif offset == 1:
            self._accumulators[slot] = value
        elif offset == 2:
            self._normalizers[slot] = value
        else:
            raise IndexError(offset)

    def get_weight_from_name(self, feature_name: str, namespace_name: str = " ") -> float:
        return self.get_weight(self.hash_feature(feature_name, self.hash_space(namespace_name)))

    def get_label_type(self) -> LabelType:
        return LabelType.SIMPLE

    def get_prediction_type(self) -> PredictionType:
        return PredictionType.SCALAR

    def finish_example(self, ex) -> None:
        if isinstance(ex, list):
            for item in ex:
                item.finished = True
        else:
            ex.finished = True

    def save(self, filename: str | Path) -> None:
        with open(filename, "wb") as handle:
            np.savez(
                handle,
                weights=self.weights,
                accumulators=self._accumulators,
                normalizers=self._normalizers,
                state=self._state,
                bounds=self._bounds,
                bits=np.array(self.bits),
            )

    def _load(self, filename: str | os.PathLike) -> None:
        with np.load(filename) as model:
            if int(model["bits"]) != self.bits:
                raise ValueError("model bit precision does not match Workspace")
            self.weights[:] = model["weights"]
            self._accumulators[:] = model["accumulators"]
            self._normalizers[:] = model["normalizers"]
            self._state[:] = model["state"]
            self._bounds[:] = model["bounds"]

    def get_arguments(self) -> str:
        return " ".join(self._tokens)

    def get_log(self) -> list[str]:
        if not self.enable_logging:
            raise RuntimeError("enable_logging set to false")
        return list(self._logs)

    def finish(self) -> None:
        self.finished = True


vw = Workspace

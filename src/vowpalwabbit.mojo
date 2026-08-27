"""Hashed online linear learning kernels exposed through a C ABI."""

from max.algorithm import parallelize
from std.math import exp, log, pow, sqrt
from std.sys.info import num_physical_cores, simd_width_of
from std.utils.numerics import isfinite

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]

comptime MURMUR_C1: UInt32 = 0xCC9E2D51
comptime MURMUR_C2: UInt32 = 0x1B873593
comptime PARALLEL_HASH_THRESHOLD = 1_000_000
comptime MAX_HASH_WORKERS = 8


def rotl32(value: UInt32, shift: UInt32) -> UInt32:
    return (value << shift) | (value >> (UInt32(32) - shift))


def fmix32(value: UInt32) -> UInt32:
    var h = value
    h ^= h >> 16
    h *= UInt32(0x85EBCA6B)
    h ^= h >> 13
    h *= UInt32(0xC2B2AE35)
    h ^= h >> 16
    return h


def murmur32(data: BPtr, length: Int, seed: UInt32) -> UInt32:
    var h = seed
    var i = 0
    while i + 4 <= length:
        var k = (
            UInt32(data[i])
            | (UInt32(data[i + 1]) << 8)
            | (UInt32(data[i + 2]) << 16)
            | (UInt32(data[i + 3]) << 24)
        )
        k *= MURMUR_C1
        k = rotl32(k, 15)
        k *= MURMUR_C2
        h ^= k
        h = rotl32(h, 13)
        h = h * UInt32(5) + UInt32(0xE6546B64)
        i += 4

    var tail = UInt32(0)
    var remaining = length - i
    if remaining == 3:
        tail ^= UInt32(data[i + 2]) << 16
    if remaining >= 2:
        tail ^= UInt32(data[i + 1]) << 8
    if remaining >= 1:
        tail ^= UInt32(data[i])
        tail *= MURMUR_C1
        tail = rotl32(tail, 15)
        tail *= MURMUR_C2
        h ^= tail
    h ^= UInt32(length)
    return fmix32(h)


def hash_string(data: BPtr, length: Int, seed: UInt32, hash_all: Bool) -> UInt32:
    if hash_all:
        return murmur32(data, length, seed)
    var start = 0
    var end = length
    while start < end and data[start] <= UInt8(0x20):
        start += 1
    while end > start and data[end - 1] <= UInt8(0x20):
        end -= 1
    var number = UInt32(0)
    for i in range(start, end):
        var byte = data[i]
        if byte < UInt8(ord("0")) or byte > UInt8(ord("9")):
            return murmur32(data + start, end - start, seed)
        number = number * UInt32(10) + UInt32(byte - UInt8(ord("0")))
    return number + seed


def ascii_space(value: UInt8) -> Bool:
    return (
        value == UInt8(0x20)
        or (value >= UInt8(0x09) and value <= UInt8(0x0D))
        or (value >= UInt8(0x1C) and value <= UInt8(0x1F))
    )


def valid_ascii_float(data: BPtr, start: Int, end: Int) -> Bool:
    if start >= end:
        return False
    var i = start
    if data[i] == UInt8(ord("+")) or data[i] == UInt8(ord("-")):
        i += 1
    var digits = 0
    while i < end and data[i] >= UInt8(ord("0")) and data[i] <= UInt8(ord("9")):
        digits += 1
        i += 1
    if i < end and data[i] == UInt8(ord(".")):
        i += 1
        while i < end and data[i] >= UInt8(ord("0")) and data[i] <= UInt8(ord("9")):
            digits += 1
            i += 1
    if digits == 0:
        return False
    if i < end and (data[i] == UInt8(ord("e")) or data[i] == UInt8(ord("E"))):
        i += 1
        if i < end and (
            data[i] == UInt8(ord("+")) or data[i] == UInt8(ord("-"))
        ):
            i += 1
        var exponent_digits = 0
        while i < end and data[i] >= UInt8(ord("0")) and data[i] <= UInt8(ord("9")):
            exponent_digits += 1
            i += 1
        if exponent_digits == 0:
            return False
    return i == end


def parse_ascii_float(data: BPtr, start: Int, end: Int) -> Float64:
    var i = start
    var sign = Float64(1.0)
    if data[i] == UInt8(ord("-")):
        sign = Float64(-1.0)
        i += 1
    elif data[i] == UInt8(ord("+")):
        i += 1
    var value = Float64(0.0)
    while i < end and data[i] >= UInt8(ord("0")) and data[i] <= UInt8(ord("9")):
        value = value * Float64(10.0) + Float64(data[i] - UInt8(ord("0")))
        i += 1
    if i < end and data[i] == UInt8(ord(".")):
        i += 1
        var place = Float64(0.1)
        while i < end and data[i] >= UInt8(ord("0")) and data[i] <= UInt8(ord("9")):
            value += Float64(data[i] - UInt8(ord("0"))) * place
            place *= Float64(0.1)
            i += 1
    if i < end and (data[i] == UInt8(ord("e")) or data[i] == UInt8(ord("E"))):
        i += 1
        var exponent_sign = 1
        if data[i] == UInt8(ord("-")):
            exponent_sign = -1
            i += 1
        elif data[i] == UInt8(ord("+")):
            i += 1
        var exponent = 0
        while i < end:
            exponent = exponent * 10 + Int(data[i] - UInt8(ord("0")))
            i += 1
        value *= pow(Float64(10.0), Float64(exponent_sign * exponent))
    return sign * value


def clip_prediction(value: Float32, lo: Float32, hi: Float32) -> Float32:
    return min(max(value, lo), hi)


def apply_link(value: Float32, link: Int) -> Float32:
    if link == 1:
        return Float32(1.0) / (Float32(1.0) + exp(-value))
    return value


def sparse_predict(
    weights: F32Ptr,
    indices: U64Ptr,
    values: F32Ptr,
    start: Int,
    end: Int,
    mask: UInt64,
    initial: Float32,
) -> Float32:
    var prediction = initial
    comptime W = simd_width_of[DType.float64]()
    var j = start
    var vector_end = start + ((end - start) // W) * W
    while j < vector_end:
        var feature_indices = indices.load[width=W](j) & mask
        var feature_weights = weights.gather[width=W](feature_indices)
        prediction += (
            feature_weights * values.load[width=W](j)
        ).reduce_add()
        j += W
    while j < end:
        prediction += weights[Int(indices[j] & mask)] * values[j]
        j += 1
    return prediction


def rate_decay(
    accumulator: Float32,
    normalizer: Float32,
    power_t: Float32,
    adaptive: Bool,
    normalized: Bool,
) -> Float32:
    var rate = Float32(1.0)
    if adaptive:
        if power_t == Float32(0.5):
            rate = Float32(1.0) / sqrt(accumulator)
        else:
            rate = pow(accumulator, -power_t)
    if normalized:
        if power_t == Float32(0.5):
            if adaptive:
                rate /= normalizer
            else:
                rate /= normalizer * normalizer
        else:
            rate *= pow(normalizer * normalizer, power_t - Float32(1.0))
    return rate


def unsafe_loss_update(
    prediction: Float32,
    label: Float32,
    update_scale: Float32,
    loss: Int,
    loss_min: Float32,
    loss_max: Float32,
) -> Float32:
    if loss == 1:
        var std_label = (label - loss_min) / (loss_max - loss_min)
        var positive = update_scale / (Float32(1.0) + exp(prediction))
        var negative = -update_scale / (Float32(1.0) + exp(-prediction))
        return std_label * positive + (Float32(1.0) - std_label) * negative
    if loss == 2:
        if label * prediction >= Float32(1.0):
            return Float32(0.0)
        return label * update_scale
    return Float32(2.0) * (label - prediction) * update_scale


def invariant_loss_update(
    prediction: Float32,
    label: Float32,
    update_scale: Float32,
    pred_per_update: Float32,
    loss: Int,
    loss_min: Float32,
    loss_max: Float32,
) -> Float32:
    if pred_per_update <= Float32(0.0):
        return Float32(0.0)
    if loss == 2:
        if label * prediction >= Float32(1.0):
            return Float32(0.0)
        var error = Float32(1.0) - label * prediction
        return label * min(update_scale, error / pred_per_update)
    if loss == 1:
        var std_label = (label - loss_min) / (loss_max - loss_min)
        var result = Float32(0.0)
        for polarity in range(2):
            var target = Float32(1.0) if polarity == 0 else Float32(-1.0)
            var mix = std_label if polarity == 0 else Float32(1.0) - std_label
            if mix == Float32(0.0):
                continue
            var d = exp(target * prediction)
            if update_scale * pred_per_update < Float32(1e-6):
                result += (
                    mix
                    * target
                    * update_scale
                    / (Float32(1.0) + d)
                )
                continue
            var x = Float64(
                update_scale * pred_per_update + target * prediction + d
            )
            var w = (
                Float64(0.86) * x + Float64(0.01)
                if x >= Float64(1.0)
                else exp(Float64(0.8) * x - Float64(0.65))
            )
            var r = (
                x - log(w) - w
                if x >= Float64(1.0)
                else Float64(0.2) * x + Float64(0.65) - w
            )
            var t = Float64(1.0) + w
            var u = Float64(2.0) * t * (
                t + Float64(2.0) * r / Float64(3.0)
            )
            var wexpmx = Float32(
                w
                * (
                    Float64(1.0)
                    + r / t * (u - r) / (u - Float64(2.0) * r)
                )
                - x
            )
            result += mix * (
                -(target * wexpmx + prediction) / pred_per_update
            )
        return result
    var amount = Float32(2.0) * update_scale * pred_per_update
    if amount < Float32(1e-6):
        return Float32(2.0) * (label - prediction) * update_scale
    return (
        (label - prediction)
        * (Float32(1.0) - exp(-amount))
        / pred_per_update
    )


@export("mvw_uniform_hash")
def mvw_uniform_hash(data_addr: Int, length: Int, seed: Int) abi("C") -> Int:
    var data = BPtr(unsafe_from_address=data_addr)
    return Int(murmur32(data, length, UInt32(seed)))


@export("mvw_hash_string")
def mvw_hash_string(
    data_addr: Int, length: Int, seed: Int, hash_all: Int
) abi("C") -> Int:
    var data = BPtr(unsafe_from_address=data_addr)
    return Int(hash_string(data, length, UInt32(seed), hash_all != 0))


@export("mvw_hash_many")
def mvw_hash_many(
    data_addr: Int,
    offsets_addr: Int,
    seeds_addr: Int,
    result_addr: Int,
    count: Int,
    mask: Int,
    hash_all: Int,
    shared_seed: Int,
    delimited: Int,
) abi("C"):
    var data = BPtr(unsafe_from_address=data_addr)
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var seeds = U32Ptr(unsafe_from_address=seeds_addr)
    var result = U64Ptr(unsafe_from_address=result_addr)

    @parameter
    def hash_one(i: Int):
        var start = Int(offsets[i])
        var end = Int(offsets[i + 1])
        if delimited != 0:
            end -= 1
        var seed = seeds[0] if shared_seed != 0 else seeds[i]
        result[i] = UInt64(
            hash_string(data + start, end - start, seed, hash_all != 0)
        ) & UInt64(mask)

    if count >= PARALLEL_HASH_THRESHOLD:
        var workers = min(MAX_HASH_WORKERS, num_physical_cores())
        var chunk_size = (count + workers - 1) // workers

        @parameter
        def hash_chunk(worker: Int):
            var start = worker * chunk_size
            var end = min(start + chunk_size, count)
            for i in range(start, end):
                hash_one(i)

        parallelize[hash_chunk](workers, workers)
    else:
        for i in range(count):
            hash_one(i)


@export("mvw_parse_line")
def mvw_parse_line(
    data_addr: Int,
    length: Int,
    indices_addr: Int,
    values_addr: Int,
    label_addr: Int,
    importance_addr: Int,
    initial_addr: Int,
    capacity: Int,
    mask: Int,
    hash_all_arg: Int,
    noconstant_arg: Int,
) abi("C") -> Int:
    var data = BPtr(unsafe_from_address=data_addr)
    var indices = U64Ptr(unsafe_from_address=indices_addr)
    var values = F32Ptr(unsafe_from_address=values_addr)
    var labels = F32Ptr(unsafe_from_address=label_addr)
    var importance = F32Ptr(unsafe_from_address=importance_addr)
    var initial = F32Ptr(unsafe_from_address=initial_addr)
    var hash_all = hash_all_arg != 0

    var header_end = 0
    while header_end < length and data[header_end] != UInt8(ord("|")):
        header_end += 1
    var position = 0
    while position < header_end and ascii_space(data[position]):
        position += 1
    if position == header_end:
        return -1
    var token_end = position
    while token_end < header_end and not ascii_space(data[token_end]):
        token_end += 1
    if not valid_ascii_float(data, position, token_end):
        return -1
    labels[0] = Float32(parse_ascii_float(data, position, token_end))
    importance[0] = Float32(1.0)
    initial[0] = Float32(0.0)

    position = token_end
    while position < header_end and ascii_space(data[position]):
        position += 1
    if position < header_end:
        token_end = position
        while token_end < header_end and not ascii_space(data[token_end]):
            token_end += 1
        if not valid_ascii_float(data, position, token_end):
            return -1
        importance[0] = Float32(parse_ascii_float(data, position, token_end))
        position = token_end
        while position < header_end and ascii_space(data[position]):
            position += 1
        if position < header_end:
            token_end = position
            while token_end < header_end and not ascii_space(data[token_end]):
                token_end += 1
            if not valid_ascii_float(data, position, token_end):
                return -1
            initial[0] = Float32(parse_ascii_float(data, position, token_end))

    var count = 0
    position = header_end
    while position < length:
        var segment_start = position + 1
        var segment_end = segment_start
        while segment_end < length and data[segment_end] != UInt8(ord("|")):
            segment_end += 1
        var anonymous = (
            segment_start == segment_end or ascii_space(data[segment_start])
        )
        var feature_position = segment_start
        while feature_position < segment_end and ascii_space(data[feature_position]):
            feature_position += 1
        var namespace_seed = UInt32(0)
        var namespace_scale = Float64(1.0)
        if anonymous:
            if hash_all:
                namespace_seed = UInt32(2129959832)
        elif feature_position < segment_end:
            var namespace_end = feature_position
            while namespace_end < segment_end and not ascii_space(data[namespace_end]):
                namespace_end += 1
            var namespace_name_end = namespace_end
            for i in range(feature_position, namespace_end):
                if data[i] == UInt8(ord(":")):
                    namespace_name_end = i
                    if not valid_ascii_float(data, i + 1, namespace_end):
                        return -1
                    namespace_scale = parse_ascii_float(data, i + 1, namespace_end)
                    break
            namespace_seed = hash_string(
                data + feature_position,
                namespace_name_end - feature_position,
                UInt32(0),
                hash_all,
            )
            feature_position = namespace_end

        while feature_position < segment_end:
            while feature_position < segment_end and ascii_space(data[feature_position]):
                feature_position += 1
            if feature_position == segment_end:
                break
            var feature_end = feature_position
            while feature_end < segment_end and not ascii_space(data[feature_end]):
                feature_end += 1
            var name_end = feature_end
            var feature_value = Float64(1.0)
            var i = feature_end
            while i > feature_position:
                i -= 1
                if data[i] == UInt8(ord(":")):
                    name_end = i
                    if name_end == feature_position:
                        break
                    if not valid_ascii_float(data, i + 1, feature_end):
                        return -1
                    feature_value = parse_ascii_float(data, i + 1, feature_end)
                    break
            if name_end != feature_position:
                if count == capacity:
                    return -2
                indices[count] = UInt64(
                    hash_string(
                        data + feature_position,
                        name_end - feature_position,
                        namespace_seed,
                        hash_all,
                    )
                ) & UInt64(mask)
                values[count] = Float32(feature_value * namespace_scale)
                count += 1
            feature_position = feature_end
        position = segment_end

    if noconstant_arg == 0:
        if count == capacity:
            return -2
        indices[count] = UInt64(11_650_396) & UInt64(mask)
        values[count] = Float32(1.0)
        count += 1
    return count


@export("mvw_predict_many")
def mvw_predict_many(
    weights_addr: Int,
    indices_addr: Int,
    values_addr: Int,
    offsets_addr: Int,
    initial_addr: Int,
    result_addr: Int,
    count: Int,
    mask: Int,
    min_prediction: Float32,
    max_prediction: Float32,
    link: Int,
) abi("C"):
    var weights = F32Ptr(unsafe_from_address=weights_addr)
    var indices = U64Ptr(unsafe_from_address=indices_addr)
    var values = F32Ptr(unsafe_from_address=values_addr)
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var initial = F32Ptr(unsafe_from_address=initial_addr)
    var result = F32Ptr(unsafe_from_address=result_addr)
    for i in range(count):
        var raw = sparse_predict(
            weights,
            indices,
            values,
            Int(offsets[i]),
            Int(offsets[i + 1]),
            UInt64(mask),
            initial[i],
        )
        result[i] = apply_link(
            clip_prediction(raw, min_prediction, max_prediction), link
        )


@export("mvw_learn_many")
def mvw_learn_many(
    weights_addr: Int,
    accumulators_addr: Int,
    normalizers_addr: Int,
    indices_addr: Int,
    values_addr: Int,
    offsets_addr: Int,
    labels_addr: Int,
    importance_addr: Int,
    initial_addr: Int,
    predictions_addr: Int,
    state_addr: Int,
    bounds_addr: Int,
    count: Int,
    mask_arg: Int,
    eta: Float32,
    power_t: Float32,
    logistic_min: Float32,
    logistic_max: Float32,
    loss: Int,
    adaptive_arg: Int,
    normalized_arg: Int,
    invariant_arg: Int,
    dynamic_bounds_arg: Int,
    link: Int,
) abi("C"):
    var weights = F32Ptr(unsafe_from_address=weights_addr)
    var accumulators = F32Ptr(unsafe_from_address=accumulators_addr)
    var normalizers = F32Ptr(unsafe_from_address=normalizers_addr)
    var indices = U64Ptr(unsafe_from_address=indices_addr)
    var values = F32Ptr(unsafe_from_address=values_addr)
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var labels = F32Ptr(unsafe_from_address=labels_addr)
    var importance = F32Ptr(unsafe_from_address=importance_addr)
    var initial = F32Ptr(unsafe_from_address=initial_addr)
    var predictions = F32Ptr(unsafe_from_address=predictions_addr)
    var state = F64Ptr(unsafe_from_address=state_addr)
    var bounds = F32Ptr(unsafe_from_address=bounds_addr)
    var mask = UInt64(mask_arg)
    var adaptive = adaptive_arg != 0
    var normalized = normalized_arg != 0
    var invariant = invariant_arg != 0
    var dynamic_bounds = dynamic_bounds_arg != 0
    var t = state[0]
    var total_weight = state[1]
    var normalized_sum = state[2]

    for row in range(count):
        var label = labels[row]
        var example_weight = importance[row]
        if dynamic_bounds:
            bounds[0] = min(bounds[0], label)
            bounds[1] = max(bounds[1], label)
        var start = Int(offsets[row])
        var end = Int(offsets[row + 1])
        var prediction = clip_prediction(
            sparse_predict(
                weights, indices, values, start, end, mask, initial[row]
            ),
            bounds[0],
            bounds[1],
        )
        predictions[row] = apply_link(prediction, link)
        t += Float64(example_weight)

        var square_gradient = Float32(0.0)
        if loss == 2:
            if label * prediction < Float32(1.0):
                square_gradient = Float32(1.0)
        elif loss == 1:
            var std_label = (
                (label - logistic_min) / (logistic_max - logistic_min)
            )
            var derivative = (
                std_label * (-Float32(1.0) / (Float32(1.0) + exp(prediction)))
                + (Float32(1.0) - std_label)
                * (Float32(1.0) / (Float32(1.0) + exp(-prediction)))
            )
            square_gradient = derivative * derivative
        else:
            var error = prediction - label
            square_gradient = Float32(4.0) * error * error
        square_gradient *= example_weight
        if square_gradient == Float32(0.0):
            continue

        var pred_per_update = Float32(0.0)
        var norm_x = Float32(0.0)
        for j in range(start, end):
            var slot = Int(indices[j] & mask)
            var x = values[j]
            var x2 = x * x
            if adaptive:
                accumulators[slot] += square_gradient * x2
            if normalized:
                var x_abs = abs(x)
                if x_abs > normalizers[slot]:
                    if normalizers[slot] > Float32(0.0):
                        var rescale = normalizers[slot] / x_abs
                        if power_t == Float32(0.5):
                            if adaptive:
                                weights[slot] *= rescale
                            else:
                                weights[slot] *= rescale * rescale
                        else:
                            var inverse_rescale = x_abs / normalizers[slot]
                            weights[slot] *= pow(
                                inverse_rescale * inverse_rescale,
                                power_t - Float32(1.0),
                            )
                    normalizers[slot] = x_abs
                norm_x += x2 / (
                    normalizers[slot] * normalizers[slot]
                )
            var decay = rate_decay(
                accumulators[slot],
                normalizers[slot],
                power_t,
                adaptive,
                normalized,
            )
            pred_per_update += x2 * decay

        var update_multiplier = Float32(1.0)
        if normalized:
            normalized_sum += Float64(example_weight * norm_x)
            total_weight += Float64(example_weight)
            if power_t == Float32(0.5):
                var average = Float32(total_weight / normalized_sum)
                update_multiplier = sqrt(average) if adaptive else average
            else:
                update_multiplier = pow(
                    Float32(normalized_sum / total_weight),
                    power_t - Float32(1.0),
                )
            pred_per_update *= update_multiplier

        var update_scale = eta * example_weight
        if not adaptive:
            update_scale *= pow(Float32(t), -power_t)
        var update = unsafe_loss_update(
            prediction,
            label,
            update_scale,
            loss,
            logistic_min,
            logistic_max,
        )
        if invariant:
            update = invariant_loss_update(
                prediction,
                label,
                update_scale,
                pred_per_update,
                loss,
                logistic_min,
                logistic_max,
            )
        if normalized:
            update *= update_multiplier
        for j in range(start, end):
            var slot = Int(indices[j] & mask)
            var decay = rate_decay(
                accumulators[slot],
                normalizers[slot],
                power_t,
                adaptive,
                normalized,
            )
            weights[slot] += update * values[j] * decay

    state[0] = t
    state[1] = total_weight
    state[2] = normalized_sum


@export("mvw_learn_text")
def mvw_learn_text(
    data_addr: Int,
    length: Int,
    weights_addr: Int,
    accumulators_addr: Int,
    normalizers_addr: Int,
    indices_addr: Int,
    values_addr: Int,
    offsets_addr: Int,
    labels_addr: Int,
    importance_addr: Int,
    initial_addr: Int,
    predictions_addr: Int,
    state_addr: Int,
    bounds_addr: Int,
    capacity: Int,
    mask: Int,
    eta: Float32,
    power_t: Float32,
    logistic_min: Float32,
    logistic_max: Float32,
    loss: Int,
    adaptive_arg: Int,
    normalized_arg: Int,
    invariant_arg: Int,
    dynamic_bounds_arg: Int,
    link: Int,
    hash_all_arg: Int,
    noconstant_arg: Int,
) abi("C") -> Int:
    var count = mvw_parse_line(
        data_addr,
        length,
        indices_addr,
        values_addr,
        labels_addr,
        importance_addr,
        initial_addr,
        capacity,
        mask,
        hash_all_arg,
        noconstant_arg,
    )
    if count < 0:
        return count

    var labels = F32Ptr(unsafe_from_address=labels_addr)
    var importance = F32Ptr(unsafe_from_address=importance_addr)
    var initial = F32Ptr(unsafe_from_address=initial_addr)
    var values = F32Ptr(unsafe_from_address=values_addr)
    if (
        not isfinite(labels[0])
        or not isfinite(importance[0])
        or not isfinite(initial[0])
    ):
        return -3
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var vector_end = count - count % W
    while i < vector_end:
        if not isfinite(values.load[width=W](i)).reduce_and():
            return -3
        i += W
    while i < count:
        if not isfinite(values[i]):
            return -3
        i += 1

    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    offsets[0] = Int64(0)
    offsets[1] = Int64(count)
    mvw_learn_many(
        weights_addr,
        accumulators_addr,
        normalizers_addr,
        indices_addr,
        values_addr,
        offsets_addr,
        labels_addr,
        importance_addr,
        initial_addr,
        predictions_addr,
        state_addr,
        bounds_addr,
        1,
        mask,
        eta,
        power_t,
        logistic_min,
        logistic_max,
        loss,
        adaptive_arg,
        normalized_arg,
        invariant_arg,
        dynamic_bounds_arg,
        link,
    )
    return count

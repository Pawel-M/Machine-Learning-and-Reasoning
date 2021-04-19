import numpy as np


def broadcast(x, y):
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    print(x.shape, y.shape)
    x = np.transpose(x, axes=[0, 2, 1])
    y = np.transpose(y, axes=[2, 0, 1])
    print(x.shape, y.shape)
    x, y = np.broadcast_arrays(x, y)
    print(x.shape, y.shape)
    return x, y


def calculate_values(x, y):
    s = x + y
    print('s', s.shape)
    print(s)
    sc = np.clip(s, -1, 1)
    print('sc', sc.shape)
    print(sc)
    return sc


def calculate_correctness(x, y):
    diff = 1 - np.maximum(0, np.abs(x - y) - 1)
    print('diff', diff.shape)
    print(diff)
    prod = np.prod(diff, axis=-1)
    print('prod', prod.shape)
    print(prod)
    return prod


def calculate_values_soft(x, y, av=10):
    return np.tanh((x + y) * av)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_correctness_soft(x, y, ac=10):
    diff = 1 - sigmoid((np.abs(x - y) - 1.5) * ac)
    print('diff', diff.shape)
    print(diff)
    prod = np.prod(diff, axis=-1)
    print('prod', prod.shape)
    print(prod)
    return prod


def calculate_out(values, correctness):
    result = values * correctness[..., np.newaxis]
    print('result', result.shape)
    print(result)
    reshaped = np.reshape(result, (result.shape[0] * result.shape[1], result.shape[2]))
    print('reshaped', reshaped.shape)
    print(reshaped)
    return reshaped


def combine_mental_models(mm1, mm2):
    mm1b, mm2b = broadcast(mm1, mm2)
    print(mm1b)
    print(mm1b.shape)
    print(mm2b)
    print(mm2b.shape)

    values = calculate_values(mm1b, mm2b)
    correctness = calculate_correctness(mm1b, mm2b)
    print('values', values.shape)
    print(values)
    print('correctness', correctness.shape)
    print(correctness)
    out = calculate_out(values, correctness)
    print('out', out.shape)
    print(out)
    return out


def combine_mental_models_soft(mm1, mm2):
    mm1b, mm2b = broadcast(mm1, mm2)
    print(mm1b)
    print(mm1b.shape)
    print(mm2b)
    print(mm2b.shape)

    values = calculate_values_soft(mm1b, mm2b, av=10)
    correctness = calculate_correctness_soft(mm1b, mm2b, ac=10)
    print('values', values.shape)
    print(values)
    print('correctness', correctness.shape)
    print(correctness)
    out = calculate_out(values, correctness)
    print('out', out.shape)
    print(out)
    return out


# (a or b)      ---> [T, n], [n, T]
# (a or not b)  ---> [T, n], [n, F]
mm1 = np.array([
    [1, 0],
    [0, 1],
])
mm2 = np.array([
    [1, 0],
    [0, -1]
])

combined_mental_models = combine_mental_models(mm1, mm2)
combined_mental_models_soft = combine_mental_models_soft(mm1, mm2)
print(combined_mental_models)
print(combined_mental_models_soft)

import numpy as np


def _select_sublevel(values, sorting_permutation, level):
    sublevel = []
    for idx in sorting_permutation:
        if values[idx] <= level:
            sublevel.append(idx)
        else:
            break
    return sublevel


def _functional_weight_filtration_v2(curvatures, num_steps, s):
    """Generates a filtration from a generic threshold function s.

    The function is scaled automatically to account for minimal and
    maximal curvatures.

    :param curvatures: List of curvature values
    :param num_steps: Number of sublevel sets to generate
    :param s: A function $[0, 1] \to [0, 1]$
    :return: A list of sublevel sets
    """
    min_weight = np.min(curvatures)
    max_weight = np.max(curvatures)
    ordering = np.argsort(curvatures)
    filtration = [[]] + [
        _select_sublevel(
            curvatures,
            ordering,
            min_weight + (max_weight - min_weight) * s((t - 1) / (num_steps - 1)),
        )
        for t in range(1, num_steps + 1)
    ]
    assert len(filtration[-1]) == len(curvatures)
    return filtration


def linear_weight_filtration_v2(curvatures, num_steps):
    result = _functional_weight_filtration_v2(curvatures, num_steps, lambda t: t)
    assert len(result[-1]) == len(curvatures)
    return result


def cosine_quantile_filtration(curvatures, num_steps):
    quantiles = np.linspace(0, 1, num_steps + 1)
    quantiles = 1 - np.cos(quantiles * np.pi / 2)
    return quantile_filtration(curvatures, num_steps, quantiles)


def inverse_cosine_quantile_filtration(curvatures, num_steps):
    quantiles = np.linspace(0, 1, num_steps + 1)
    quantiles = np.cos((quantiles - 1) * np.pi / 2)
    quantiles[0] = 0
    quantiles[-1] = 1
    return quantile_filtration(curvatures, num_steps, quantiles)


def quantile_filtration(curvatures, num_steps, quantiles=None):
    quantiles = np.linspace(0, 1, num_steps + 1) if quantiles is None else quantiles
    ordering = np.argsort(curvatures)
    filtration = []
    idx = 0
    for quantile in quantiles:
        while idx < quantile * len(curvatures):
            idx += 1
            while (
                idx < len(curvatures)
                and curvatures[ordering[idx - 1]] == curvatures[ordering[idx]]
            ):
                idx += 1
        filtration.append(ordering[:idx].tolist())
    assert len(filtration[-1]) == len(curvatures) and len(filtration[0]) == 0
    return filtration

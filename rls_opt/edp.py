from collections.abc import Sequence

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def decompose_price(
    sales_dollars: np.ndarray,
    sales_units: np.ndarray,
    indexors: Sequence[np.ndarray] | Sequence[slice] | None = None,
    max_tpr_len: int = 16,
):
    """Helper function to calculate ASP, EDP and Discount for all data rows"""

    assert sales_dollars.shape == sales_units.shape

    if indexors is None:
        indexors = [slice(None, None, None)]

    # Set up output arrays
    # FIX: sales_units puede venir en 0 (día sin ventas) -> división por cero
    # producía inf/nan que se propagaban silenciosamente (numba no emite
    # RuntimeWarning) hasta contaminar edp/discount y, más adelante, los
    # coeficientes del RLS (overflow en exp() al predecir).
    asp = np.zeros_like(sales_dollars, dtype=np.float64)
    for i in range(sales_dollars.shape[0]):
        if sales_units[i] > 0:
            asp[i] = sales_dollars[i] / sales_units[i]

    edp = np.zeros_like(sales_dollars, dtype=np.float64)
    discount = np.zeros_like(sales_dollars, dtype=np.float64)

    # Loop through each Product-Geography
    for indexor in indexors:
        asp_indexor = asp[indexor]
        n = len(asp_indexor)

        # Construct EDP
        lookback_asp = np.zeros(n, dtype=np.float64)

        for i in range(max_tpr_len - 1, n):
            lookback_asp[i] = np.max(asp_indexor[i - max_tpr_len + 1 : i + 1])

        min_max_tpr_len_or_n = min(max_tpr_len, n)
        for i in range(min_max_tpr_len_or_n):
            # TODO: What if max_tpr_len < n. Then lookback_asp[min_max_tpr_len_or_n] is 0
            #  should this case be handled separately?
            lookback_asp[i] = lookback_asp[min_max_tpr_len_or_n - 1]

        edp_indexor = edp[indexor]
        for i in range(n):
            edp_indexor[i] = np.min(lookback_asp[i : min(i + max_tpr_len, n)])

        # Calculate Discount
        discount_indexor = discount[indexor]
        for i in range(n):
            if edp_indexor[i] > 0:
                discount_indexor[i] = 1 - asp_indexor[i] / edp_indexor[i]

    return asp, edp, discount

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RLSPriorBase:
    standard_error: float
    rmse_error: float

    def coefficient_value(self, *args) -> float:
        raise NotImplementedError

    def standard_error_value(self, *args) -> float:
        raise NotImplementedError

    def xxt_seed_value(self, *args) -> float:
        return (self.standard_error_value(*args) / self.rmse_error) ** 2


@dataclass(frozen=True)
class RLSPrior(RLSPriorBase):
    coefficient: float

    def coefficient_value(self) -> float:
        return self.coefficient

    def standard_error_value(self) -> float:
        return self.standard_error


@dataclass(frozen=True)
class RLSRelativePrior(RLSPriorBase):
    coefficient: float
    _max_magnitude: float = field(default=1e6, repr=False)

    def coefficient_value(self, y: np.ndarray) -> float:
        return np.clip(
            self.coefficient / y.mean(),
            a_min=-self._max_magnitude,
            a_max=self._max_magnitude,
        )

    def standard_error_value(self, y: float) -> float:
        return np.clip(
            self.standard_error / y.mean(),
            a_min=-self._max_magnitude,
            a_max=self._max_magnitude,
        )


@dataclass(frozen=True)
class RLSConstantPrior(RLSPriorBase):
    def coefficient_value(
        self, x: np.ndarray, y: np.ndarray, priors_coeffs: np.ndarray
    ) -> float:
        return (y - x @ priors_coeffs).mean()

    def standard_error_value(self, coefficient_value: float) -> float:
        return self.standard_error * coefficient_value


DEFAULT_PRIORS = {
    "INTERCEPT": RLSConstantPrior(rmse_error=0.2, standard_error=0.5),
    "NUMBER_OF_STORES_SELLING": RLSRelativePrior(
        coefficient=1, standard_error=0.3, rmse_error=0.2
    ),
    "EDP": RLSRelativePrior(coefficient=-1, standard_error=0.3, rmse_error=0.2),
    "Discount": RLSPrior(coefficient=2.0, standard_error=0.5, rmse_error=0.2),
    "Dist": RLSPrior(coefficient=0.05, standard_error=0.1, rmse_error=0.2),
    "Feat": RLSPrior(coefficient=0.001, standard_error=0.001, rmse_error=0.2),
    "Disp": RLSPrior(coefficient=0.001, standard_error=0.001, rmse_error=0.2),
    "FeatDisp": RLSPrior(coefficient=0.002, standard_error=0.002, rmse_error=0.2),
    "Trend": RLSPrior(coefficient=0.0, standard_error=0.0001, rmse_error=0.2),
    "Feb": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Mar": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Apr": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "May": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Jun": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Jul": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Aug": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Sep": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Oct": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Nov": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Dec": RLSPrior(coefficient=0.0, standard_error=0.1, rmse_error=0.2),
    "Xmas": RLSPrior(coefficient=0.0, standard_error=0.2, rmse_error=0.2),
    "Easter": RLSPrior(coefficient=0.0, standard_error=0.2, rmse_error=0.2),
    "Halloween": RLSPrior(coefficient=0.0, standard_error=0.2, rmse_error=0.2),
    "VDay": RLSPrior(coefficient=0.0, standard_error=0.2, rmse_error=0.2),
    "Constant": RLSConstantPrior(standard_error=0.5, rmse_error=0.2),
}

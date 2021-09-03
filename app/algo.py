from dataclasses import dataclass
from typing import List

import numpy as np
import sksurv.metrics


class Result:
    pass


@dataclass
class AggregatedConcordanceIndex(Result):
    mean_cindex: float
    weighted_cindex_concordant_pairs: float


class Evaluation:
    pass


@dataclass
class LocalConcordanceIndex(Evaluation):
    cindex: float
    num_concordant_pairs: int


def calculate_cindex_on_local_data(event_indicator, event_time, estimate, tied_tol=1e-8) -> LocalConcordanceIndex:
    cindex, concordant, discordant, tied_risk, tied_time = sksurv.metrics.concordance_index_censored(event_indicator,
                                                                                                     event_time,
                                                                                                     estimate,
                                                                                                     tied_tol=tied_tol)
    return LocalConcordanceIndex(
        cindex=cindex,
        num_concordant_pairs=concordant,
    )


class GlobalConcordanceIndexEvaluations(object):
    def __init__(self, evaluations: List[LocalConcordanceIndex]):
        self.c_indices = np.zeros(len(evaluations))
        self.concordant_pairs = np.zeros(len(evaluations))
        for i, evaluation in enumerate(evaluations):
            self.c_indices[i] = evaluation.cindex
            self.concordant_pairs[i] = evaluation.num_concordant_pairs

    def mean_cindex(self) -> float:
        r"""Calculate the mean concordance index.

        .. math::
            C_{global\_mean} = \frac{\sum_{k}^{n\_clients} C_k}{n\_clients}

        Reference
        ---------
        Remus, S. L. (2021). Implementation of a federated Random Survival Forest Feature Cloud app (Bachelor's thesis).

        :return: mean concordance index
        :rtype: float
        """
        return self.c_indices.mean()

    def weighted_cindex_concordant_pairs(self) -> float:
        r"""Calculate the weighted concordance index.
        Weighting is done by the number of concordant pairs of each client.

        .. math::
            C_{global\_weighted} = \frac{\sum_{k}^{n\_clients} C_k * CP_k}{\sum_{k}^{n\_clients} CP_k}

        Reference
        ---------
        Remus, S. L. (2021). Implementation of a federated Random Survival Forest Feature Cloud app (Bachelor's thesis).

        :return: weighted concordance index
        :rtype: float
        """
        return np.sum(self.c_indices * self.concordant_pairs) / np.sum(self.concordant_pairs)

    def calc(self) -> AggregatedConcordanceIndex:
        return AggregatedConcordanceIndex(
            mean_cindex=self.mean_cindex(),
            weighted_cindex_concordant_pairs=self.weighted_cindex_concordant_pairs(),
        )

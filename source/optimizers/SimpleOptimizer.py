from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count, Pool

import tqdm

from source.theory.conditional.ConditionalProbabilisticScheme import ConditionalProbabilisticScheme
from source.theory.conditional.ConditionalStatisticalCalculations import ConditionalStatisticalCalculations
from source.theory.simple.SimpleProbabilisticScheme import SimpleProbabilisticScheme
from source.theory.simple.SimpleStatisticalCalculations import SimpleStatisticalCalculations


class SimpleOptimizer:
    def __init__(self, probabilistic_scheme: SimpleProbabilisticScheme | ConditionalProbabilisticScheme):
        self._probabilistic_scheme: SimpleProbabilisticScheme | ConditionalProbabilisticScheme = probabilistic_scheme
        if type(probabilistic_scheme) is SimpleProbabilisticScheme:
            self._calc = SimpleStatisticalCalculations(self._probabilistic_scheme)
        elif type(probabilistic_scheme) is ConditionalProbabilisticScheme:
            self._calc = ConditionalStatisticalCalculations(self._probabilistic_scheme)
        else:
            raise TypeError(f"{type(probabilistic_scheme)} is not a supported!")

    def _calculate_score(self, w, variant):
        """Helper function to calculate score for a single variant"""
        return self._calc.score(w, variant), variant

    def optimize(self, w, variants, n_processes=None, display_tqdm=False):
        """
        Parallel optimization using process pool

        Args:
            w: Speculation width (number of elements in resulting vector)
            variants: List of variants to evaluate
            n_processes: Number of processes (default: number of CPU cores)
            display_tqdm: Display tqdm progress bar
        """
        if n_processes is None:
            n_processes = cpu_count()

        # Create a partial function with fixed w parameter
        calculate_func = partial(self._calculate_score, w)

        best_score = 0
        best_variant = None

        # Use Pool for parallel computation
        with Pool(processes=n_processes) as pool:
            # Use imap_unordered to process results as they become available
            results = pool.imap_unordered(calculate_func, variants)

            # Process results with progress bar
            for score, variant in tqdm.tqdm(results, total=len(variants), disable=not display_tqdm):
                if score > best_score:
                    best_score = score
                    best_variant = variant

        return best_score, best_variant

    def _greedy(self, w: int):
        variant = sorted(self._probabilistic_scheme.get_top(w))

        return self._calculate_score(w, tuple(variant))

    def greedy(self, w: int, r: int):
        """
        The best vector is chosen as top w elements by probabilities.

        Args:
            w: Speculation width (number of elements in resulting vector)
            r: Desired 'future depth' parameter
        """
        if r == 1:
            return self._greedy(w)
        else:
            priorities = defaultdict(float)
            for r_candidate in range(1, r + 1):
                future_scheme = self._probabilistic_scheme.get_scheme_for_sum(r_candidate)
                for o, p in future_scheme.compact():
                    priorities[o] += p

            priorities = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
            variant = tuple(sorted([r[0] for r in priorities[:w]]))

            if w > len(variant):
                raise ValueError(f"{w} outcomes are impossible. max is {len(priorities)}")

            return self._calculate_score(w, variant)

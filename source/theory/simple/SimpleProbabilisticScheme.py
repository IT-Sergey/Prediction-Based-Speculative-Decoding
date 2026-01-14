import numpy as np


class SimpleProbabilisticScheme:
    def __init__(self, outcomes, probabilities):
        self.outcomes = outcomes
        self.probabilities = probabilities

    def p(self, outcome):
        return self.probabilities[self.outcomes.index(outcome)] if outcome in self.outcomes else 0

    def get_top(self, n: int):
        outcomes_and_probabilities = sorted([(o, self.p(o)) for o in self.outcomes], key=lambda x: x[1], reverse=True)

        if n > len(outcomes_and_probabilities):
            raise ValueError(f"{n} outcomes are impossible. max is {len(outcomes_and_probabilities)}")

        return [r[0] for r in outcomes_and_probabilities[:n]]

    def get_scheme_for_sum(self, k: int):
        complete_vector = [self.p(o) for o in range(0, max(self.outcomes) + 1)]

        base_vector = np.hstack((np.array(complete_vector), np.zeros((k - 1) * len(complete_vector))))
        base_vector_fft = np.fft.fft(base_vector)
        probabilities = np.real(np.fft.ifft(base_vector_fft ** k))

        selected_probabilities = []
        selected_outcomes = []

        for i, outcome in enumerate(range(0, k * (max(self.outcomes)) + 1)):
            if probabilities[i] > np.finfo(float).eps and outcome >= k:
                selected_probabilities.append(probabilities[i])
                selected_outcomes.append(outcome)

        selected_probabilities = np.array(selected_probabilities) / np.sum(selected_probabilities)

        return SimpleProbabilisticScheme(selected_outcomes, selected_probabilities)

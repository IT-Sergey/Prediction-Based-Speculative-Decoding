class BasePredictor:
    @property
    def cardinality(self):
        return 0

    @property
    def requires_training(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "BasePredictor"

    def predict(self, previous, n):
        raise NotImplementedError()

    def feed(self, current):
        pass

    def reconstruct(self, data):
        raise NotImplementedError()

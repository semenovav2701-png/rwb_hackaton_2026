class AppConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):

            self.sources = [
                {"type": "demo"}
            ]


            self.aggregation_strategy = "count"
            self.decision_strategy = "truck count"

            self._initialized = True
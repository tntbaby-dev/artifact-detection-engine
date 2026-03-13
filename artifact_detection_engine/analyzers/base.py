from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    name = "base"

    @abstractmethod
    def analyze(self, audio):
        """
        Returns a dict or AnalysisResult describing artifact findings.
        """
        raise NotImplementedError
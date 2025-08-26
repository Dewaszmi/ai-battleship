from abc import ABC, abstractmethod


class Phase(ABC):
    @abstractmethod
    def handle_events(self, events):
        pass

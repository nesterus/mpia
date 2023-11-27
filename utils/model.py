from abc import ABC, abstractmethod

class Model:
    def __init__(self, *args, **kwargs):
        self.init()
    
    @abstractmethod
    def init(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def run(self):
        raise NotImplementedError

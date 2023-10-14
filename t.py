from src.mesoscaler.generic import Data


class MyData(Data[int]):
    def __init__(self):
        self.node_1 = 0
        self.node_2 = 0

    @property
    def nodes(self):
        return "node_1", "node_2"

    @property
    def data(self):
        return ((node, getattr(self, node)) for node in self.nodes)

    def __len__(self):
        return len(self.nodes)

# https://en.wikipedia.org/wiki/Disjoint-set_data_structure
class DisjointSetForest:
    class DisjointSetNode:
        def __init__(self, value):
            self.parent = self
            self.set = {value}

    def __init__(self):
        self._value_to_node = {}

    def make_set(self, value):
        if value not in self._value_to_node:
            self._value_to_node[value] = self.DisjointSetNode(value)
        return self._value_to_node[value]

    def get_forest_sets(self):
        trees = {self.find(node) for node in self._value_to_node.values()}
        return [tree.set for tree in trees]

    def find(self, x):
        """
        x can be the value or the node that holds the value.
        Supporting node as an argument is needed for the recursive call
        If x is a value that is not in the forest, returns None.

        """
        if isinstance(x, self.DisjointSetNode):
            node = x
        elif x in self._value_to_node:
            node = self._value_to_node[x]
        else:
            return None
        if node.parent == node:
            return node
        node.parent = self.find(node.parent)
        return node.parent

    def union(self, x, y):
        """
        x and y can be either the values of the nodes, or the nodes themselves
        """
        node1 = self.find(x)
        node2 = self.find(y)
        if node1 == node2:
            return node1
        if len(node1.set) < len(node2.set):
            node1, node2 = node2, node1
        node2.parent = node1
        node1.set = node1.set.union(node2.set)
        node2.set = None
        return node1

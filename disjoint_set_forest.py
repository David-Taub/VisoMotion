# https://en.wikipedia.org/wiki/Disjoint-set_data_structure
class DisjointSetForest:
    class DisjointSetNode:
        def __init__(self, value):
            self.parent = self
            self.set = {value}

    def __init__(self):
        self._value_to_tree = {}

    def make_set(self, value):
        if value not in self._value_to_tree:
            self._value_to_tree[value] = self.DisjointSetNode(value)

    def find(self, x):
        """
        x can be the value or the node that holds the value.
        Supporting node as an argument is needed for the recursive call
        If x is a value that is not in the forest, returns None.

        """
        if isinstance(x, self.DisjointSetNode):
            node = x
        elif x in self._value_to_tree:
            node = self._value_to_tree[x]
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
            return
        if len(node1.set) < len(node2.set):
            node1.parent = node2
            node2.set = node2.set.union(node1.set)
            node1.set = None
        else:
            node2.parent = node1
            node1.set = node1.set.union(node2.set)
            node2.set = None

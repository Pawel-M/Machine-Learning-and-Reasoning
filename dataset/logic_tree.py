import abc


class Node(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, values):
        pass

    @property
    @abc.abstractmethod
    def variables(self):
        pass

    @abc.abstractmethod
    def to_string(self):
        pass

    @abc.abstractmethod
    def to_string_prefix(self):
        pass


class ValueNode(Node):
    def __init__(self, variable):
        self._variable = variable
        self._variables = (variable,)

    def evaluate(self, values):
        return values[self._variable]

    @property
    def variables(self):
        return self._variables

    def to_string(self):
        return str(self._variable)

    def to_string_prefix(self):
        return self.to_string()


class NotNode(Node):
    def __init__(self, child: Node):
        self._child = child
        self._variables = ()

    def evaluate(self, values):
        return not self._child.evaluate(values)

    @property
    def variables(self):
        return self._child.variables

    def to_string(self):
        return f'not {self._child.to_string()}'

    def to_string_prefix(self):
        return f'not {self._child.to_string_prefix()}'


class OperatorNode(Node, abc.ABC):
    def __init__(self, operator_symbol, *children):
        self._operator_symbol = operator_symbol
        self._children = tuple(children)

        variables_set = set()
        for child in children:
            variables_set = variables_set.union(set(child.variables))
        self._variables = tuple(variables_set)

    @property
    def variables(self):
        return self._variables

    def to_string(self):
        string = f'( {self._children[0].to_string()}'
        for child in self._children[1:]:
            string += f' {self._operator_symbol} {child.to_string()}'
        string += ' )'
        return string

    def to_string_prefix(self):
        arity = len(self._children)
        symbol = f'{self._operator_symbol}{arity}' if arity is not 2 else f'{self._operator_symbol}'
        string = f'{symbol}'
        for child in self._children:
            string += f' {child.to_string_prefix()}'
        return string


class AndNode(OperatorNode):
    def __init__(self, *children):
        super(AndNode, self).__init__('and', *children)

    def evaluate(self, values):
        value = self._children[0].evaluate(values)
        for child in self._children[1:]:
            value = value and child.evaluate(values)

        return value


class OrNode(OperatorNode):
    def __init__(self, *children):
        super(OrNode, self).__init__('or', *children)

    def evaluate(self, values):
        value = self._children[0].evaluate(values)
        for child in self._children[1:]:
            value = value or child.evaluate(values)

        return value


if __name__ == '__main__':
    import random

    tree = AndNode(
        OrNode(
            ValueNode(1),
            ValueNode(2)),
        NotNode(
            OrNode(
                ValueNode(2),
                ValueNode(3),
                ValueNode(3))
        ))

    print(tree.to_string())
    print(tree.to_string_prefix())

    print(tree.variables)
    values = {variable: random.random() < .5 for variable in tree.variables}
    print(values)
    print(tree.evaluate(values))

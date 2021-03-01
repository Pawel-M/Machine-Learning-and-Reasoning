# create tree structure - Thomas
# generate random tree - Theodor
# generate truth table - Theodor
# finish test for one conclusion - Pawel
# save valid trees (with conclusions) to disc - Thomas

import numpy as np

# Comment Thomas: everything still needs implementation of brackets
# Comment Thomas: print_infix only works when nodes have at most 2 children and root of tree has exactly 2 children

class Node:
    # A node contains a value (in our case a symbol like "A","B",... or an operand "and", "or", "not", ...
    # A node can have a list of children which should also be of type node.
    # Note that the order of the nodes within children is important for printing the logical sentence as a string,
    # the first element in children should be the leftmost leaf of the tree
    def __init__(self, value, children=None):
        self.value = value
        if children is None:
            self.children = []
        self.children = children


def conjunction(left_child, right_child, operands=None):
    # The dictionary "operands" stores pairs of decoded operands (keys) and their english words (values).
    # The standard operands, like "and", "or", "not", "(" and ")", should always be the first five elements
    # and they should also be in this exact order.
    if operands is None:
        operands = {"and": "and", "or": "or", "not": "not", "(": "(", ")": ")"}
    operator = list(operands.keys())[list(operands.values()).index("and")]
    return Node(operator, [left_child, right_child])


def disjunction(left_child, right_child, operands=None):
    # The dictionary "operands" stores pairs of decoded operands (keys) and their english words (values).
    # The standard operands, like "and", "or", "not", "(" and ")", should always be the first five elements
    # and they should also be in this exact order.
    if operands is None:
        operands = {"and": "and", "or": "or", "not": "not", "(": "(", ")": ")"}
    operator = list(operands.keys())[list(operands.values()).index("or")]
    return Node(operator, [left_child, right_child])


def negation(child, operands=None):
    # The dictionary "operands" stores pairs of decoded operands (keys) and their english words (values).
    # The standard operands, like "and", "or", "not", "(" and ")", should always be the first five elements
    # and they should also be in this exact order.
    if operands is None:
        operands = {"and": "and", "or": "or", "not": "not", "(": "(", ")": ")"}
    operator = list(operands.keys())[list(operands.values()).index("not")]
    return Node(operator, [child])


class Tree:
    # The tree structure can be initialized empty or with a root, which is an object of type Node.
    def __init__(self, root=None):
        if root is None:
            root = []
        self.root = root

    def print_infix2(self, node=None, operands=None):
        # This method will print out the logical sentence represented in the tree as a string with infix notation.
        # The dictionary "operands" stores pairs of decoded operands (keys) and their english words (values).
        # The standard operands, like "and", "or", "not", "(" and ")", should always be the first five elements
        # and they should also be in this exact order.
        if operands is None:
            operands = {"and": "and", "or": "or", "not": "not", "(": "(", ")": ")"}
        output_string = []
        # list of strings that the method will return at the end
        if node is None:
            root = self.root
            left_string = self.print_infix2(root.children[0])
            right_string = self.print_infix2(root.children[1])
            output_string.append(left_string + " " + operands[root.value] + " " + right_string)
        # This is done to allow both calling this method directly from a tree object,
        # but also recursively within the method itself.
        # Take note that when called recursively within the method, the input is simply a node, and
        # not a tree object necessarily.
        else:
            temp = node
            if temp.children is None:
                return temp.value
            elif len(temp.children) == 1:
                # If a node has only one child and this child has no children themselves, the algorithm implicitly
                # assumes the node is a "not" operator. It then concatenates the value of the node with the value of
                # the child as a string, again using the dictionary to translate the value of the node into english.
                string_temp = operands[temp.value] + temp.children[0].value
                return string_temp
            elif len(temp.children) == 2:
                string_left = self.print_infix2(temp.children[0])
                string_right = self.print_infix2(temp.children[1])
                return string_left + operands[temp.value] + string_right
            else:
                return "No procedure defined for nodes with more than two leaves."
                # Implement further code when a node has more than two children
        print(output_string)


def generate_random_tree(total_variables, num_used_variables, max_depth):
    pass



def generate_truth_table(num_variables):
    table = list(itertools.product([False, True], repeat=num_variables))

    return table
  

def test_for_one_conclusion(tree):
    variables = tree.variables

    truth_table = get_truth_table(len(variables))
    always_true = [True for _ in range(len(variables))]
    for evaluation in truth_table:
        # evaluation = [True, False, False]
        if not tree.evaluate(evaluation):
            continue

        for i in range(len(variables)):
            if not evaluation[i]:
                always_true[i] = False

    # TODO add check for one conclusion and return
    # return True, 1, truth_table
    # return False, None


for i in range(1000):
    tree = generate_random_tree()
    valid, conclusion, _ = test_for_one_conclusion(tree)

    valid_trees = []
    if valid:
        valid_trees.append(tree, conclusion)

    # save list of trees

#this is to test what is implemented so far
if __name__ == '__main__':
    con = conjunction(Node("A"), Node("B"))
    neg = negation(Node("C"))
    first_root = disjunction(con, neg)
    first_tree = Tree(first_root)
    first_tree.print_infix2()

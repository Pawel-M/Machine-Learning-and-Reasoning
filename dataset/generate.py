# create tree structure - Thomas
# generate random tree - Theodor
# generate truth table - Theodor
# finish test for one conclusion - Pawel
# save valid trees (with conclusions) to disc - Thomas

import numpy as np


def generate_random_tree(total_variables, num_used_variables, max_depth):
    pass


def get_truth_table(variables):
    pass


# truth_tables = {
#     1: [[True], [False]],
#     2: [[False, False], [False, True], [True, False], [True, True]],
# }

def generate_truth_table(num_variables):
    #if num_variables not in truth_tables:
        #truth_tables[num_variables] = ...

    #return truth_tables[num_variables]


    # [[True], [False]] - 1 variable
    # [[False, False], [False, True], [True, False], [True, True]] - 2 variables
    # ...
    pass
    # return [[]]


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

import os
import random

import numpy as np
import pandas as pd

import dataset.logic_tree

pd.set_option('display.max_columns', 100)

OPERATION_NODES = (
    (dataset.logic_tree.NotNode, 1),
    (dataset.logic_tree.AndNode, 2),
    (dataset.logic_tree.OrNode, 2),
)


def generate_node(num_operations, variables):
    if num_operations == 0 or random.random() < .1:
        return dataset.logic_tree.ValueNode(random.choice(variables))

    operation_node, num_children = random.choice(OPERATION_NODES)
    children = list([generate_node(num_operations - 1, variables) for _ in range(num_children)])
    return operation_node(*children)


def generate_random_tree(num_operations, variables):
    return generate_node(num_operations, variables)


def get_truth_table(variables):
    num_variables = len(variables)
    max_bit_values = 2 ** num_variables

    truth_table = []
    for bit_values in range(max_bit_values):
        values = [int(x) == 1 for x in '{:0{size}b}'.format(bit_values, size=num_variables)]
        truth_row = {variables[i]: values[i] for i in range(num_variables)}
        truth_table.append(truth_row)

    return truth_table


def test_for_always_true(tree):
    variables = tree.variables

    truth_table = get_truth_table(variables)
    always_true = [True for _ in range(len(variables))]
    for truth_row in truth_table:
        if not tree.evaluate(truth_row):
            continue

        for i, variable in enumerate(variables):
            if not truth_row[variable]:
                always_true[i] = False

    always_true_array = np.array(always_true).astype(np.int)
    if np.sum(always_true_array) == 1:
        conclusion_index = np.where(always_true_array == 1)[0][0]
        return True, variables[conclusion_index]
    else:
        return False, None


def test_for_guarantees_true(tree):
    variables = tree.variables

    truth_table = get_truth_table(variables)
    count_variable_true = [0 for _ in range(len(variables))]
    num_tree_true = 0
    for truth_row in truth_table:
        if not tree.evaluate(truth_row):
            continue

        num_tree_true += 1
        for i, variable in enumerate(variables):
            if truth_row[variable]:
                count_variable_true[i] += 1

    if num_tree_true == 0:
        return False, None

    count_array = np.array(count_variable_true).astype(np.int)
    guarantees = np.where(count_array == num_tree_true)[0]

    if len(guarantees) == 1:
        return tree, variables[guarantees[0]]
    else:
        return False, None


def generate_and_test_trees(num_generations, num_operations, num_variables, test_fn, print_every=100):
    variables = tuple(range(1, num_variables + 1))
    correct_trees_infix = []
    correct_trees_prefix = []
    conclusions = []
    for i in range(num_generations):
        tree = generate_random_tree(num_operations, variables)
        correct, conclusion = test_fn(tree)
        if correct:
            infix_tree = tree.to_string()

            if infix_tree not in correct_trees_infix:
                correct_trees_infix.append(infix_tree)
                correct_trees_prefix.append(tree.to_string_prefix())
                conclusions.append(conclusion)

        if (i + 1) % print_every == 0:
            print(f'Checked {i + 1} trees, correct: {len(correct_trees_infix)}')

    return list(zip(correct_trees_infix, correct_trees_prefix, conclusions))


def generate_and_save_trees(folder, num_generations, max_depth, num_variables, test_fn):
    trees = generate_and_test_trees(num_generations, max_depth, num_variables, test_fn)
    print('found trees:', len(trees))

    for i in range(5):
        print('infix tree:', trees[i][0])
        print('prefix tree:', trees[i][1])
        print('conclusion:', trees[i][2])
        print()

    df = pd.DataFrame(trees)
    df.rename(columns={0: 'infix', 1: 'prefix', 2: 'conclusion'}, inplace=True)

    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = os.path.join(folder, f'trees_depth-{max_depth}_vars-{num_variables}.csv')
    df.to_csv(file_name)


def load_trees(folder, max_depth, num_variables):
    file_name = f'trees_depth-{max_depth}_vars-{num_variables}.csv'
    df = pd.read_csv(os.path.join(folder, file_name), index_col=0)
    return df


if __name__ == '__main__':
    # for i in range(100):
    #     tree = generate_random_tree(3, tuple(range(1, 3 + 1)))
    #     correct, conclusion = test_for_guarantees_true(tree)
    #     if correct:
    #         print('tree', tree.to_string())
    #         print('variables', tree.variables)
    #         print('conclusion', conclusion)
    #         print()

    generate_and_save_trees('../data', num_generations=1000000,
                            max_depth=2, num_variables=5,
                            test_fn=test_for_guarantees_true)

import math
import os
import random
import time
import itertools

import numpy as np
import pandas as pd

from dataset.logic_tree import NotNode, AndNode, OrNode, ValueNode

pd.set_option('display.max_columns', 100)

OPERATION_NODES = (
    (NotNode, 1),
    (AndNode, 2),
    (OrNode, 2),
)


def get_hms(seconds):
    h = math.floor(seconds / 3600)
    m = math.floor(seconds / 60) % 60
    s = int(seconds) % 60

    result = f'{s:0>2}'
    if m > 0 or h > 0:
        if h > 0:
            result = f'{h}:{m:0>2}:' + result
        else:
            result = f'{m}:' + result
    return result


def generate_node(num_operations, variables):
    if num_operations == 0 or random.random() < .1:
        return ValueNode(random.choice(variables))

    operation_node, num_children = random.choice(OPERATION_NODES)
    children = list([generate_node(num_operations - 1, variables) for _ in range(num_children)])
    return operation_node(*children)


def generate_random_tree(num_operations, variables, root_node_cls=None):
    if root_node_cls is None:
        return generate_node(num_operations, variables)

    children = []
    for i in range(root_node_cls.accepts_children):
        children.append(generate_node(num_operations - 1, variables))

    return root_node_cls(*children)


def get_truth_table(variables):
    num_variables = len(variables)
    max_bit_values = 2 ** num_variables

    truth_table = []
    for bit_values in range(max_bit_values):
        values = [int(x) == 1 for x in '{:0{size}b}'.format(bit_values, size=num_variables)]
        truth_row = {variables[i]: values[i] for i in range(num_variables)}
        truth_table.append(truth_row)

    return truth_table


def test_for_always_true(tree, num_variables):
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


def test_for_guarantees_true(tree, num_variables):
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


def generate_and_test_trees(num_generations, num_operations, num_variables, test_fn, generate_fn=generate_random_tree,
                            existing=None, print_every=100):
    variables = tuple(range(1, num_variables + 1))

    if existing is None:
        correct_trees_infix = []
        correct_trees_prefix = []
        conclusions = []
    else:
        correct_trees_infix, correct_trees_prefix, conclusions = existing

    num_correct = len(correct_trees_infix)
    start_time = time.time()
    for i in range(num_generations):
        tree = generate_fn(num_operations, variables)
        correct, conclusion = test_fn(tree, num_variables)
        if correct:
            infix_tree = tree.to_string()

            if infix_tree not in correct_trees_infix:
                correct_trees_infix.append(infix_tree)
                correct_trees_prefix.append(tree.to_string_prefix())
                conclusions.append(conclusion)

        if (i + 1) % print_every == 0:
            last_num_correct = num_correct
            num_correct = len(correct_trees_infix)
            percent_correct = int(1000 * (num_correct - last_num_correct) / print_every) / 10
            elapsed_iterations = (i + 1) / num_generations
            percent_trees = int(1000 * elapsed_iterations) / 10
            elapsed_time = time.time() - start_time
            eta = (1 - elapsed_iterations) * elapsed_time / elapsed_iterations
            print(
                f'Checked {i + 1} ({percent_trees}%) trees, correct: {num_correct}, recently correct: {percent_correct}%, eta: {get_hms(eta)}s')

    return list(zip(correct_trees_infix, correct_trees_prefix, conclusions))


def generate_and_save_trees(folder, num_generations, max_depth, num_variables, test_fn,
                            generate_fn=generate_random_tree, base_name='trees', load_existing=False,
                            print_every=100):
    existing = None
    if load_existing:
        try:
            trees_df = load_trees(folder, max_depth, num_variables, base_name)
            infix_trees = trees_df['infix'].tolist()
            prefix_trees = trees_df['prefix'].tolist()
            conclusions = trees_df['conclusion'].tolist()
            existing = infix_trees, prefix_trees, conclusions

        except IOError:
            print('Cannot load existing tree - file does not exist.')
            existing = None

    trees = generate_and_test_trees(num_generations, max_depth, num_variables,
                                    test_fn, generate_fn,
                                    existing, print_every)
    print('found trees:', len(trees))

    for i in range(15):
        print('infix tree:', trees[i][0])
        print('prefix tree:', trees[i][1])
        print('conclusion:', trees[i][2])
        print()

    df = pd.DataFrame(trees)
    df.rename(columns={0: 'infix', 1: 'prefix', 2: 'conclusion'}, inplace=True)

    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = os.path.join(folder, f'{base_name}_depth-{max_depth}_vars-{num_variables}.csv')
    df.to_csv(file_name, sep=';')


def load_trees(folder, max_depth, num_variables, base_name='trees'):
    file_name = f'{base_name}_depth-{max_depth}_vars-{num_variables}.csv'
    df = pd.read_csv(os.path.join(folder, file_name), index_col=0, sep=';')
    return df


def test_for_mental_models(tree, num_variables, type_one=True, allow_only_one_mental_model=False):
    variables = tree.variables

    truth_table = get_truth_table(variables)

    type_two_mental_models = []
    num_tree_true = 0
    for truth_row in truth_table:
        if not tree.evaluate(truth_row):
            continue

        num_tree_true += 1
        mental_model = np.zeros(num_variables)
        for variable in range(num_variables):
            if (variable + 1) in truth_row:
                mental_model[variable] = 1 if truth_row[variable + 1] else -1

        type_two_mental_models.append(mental_model)

    if num_tree_true == 0:
        return False, None

    if type_one:
        type_one_mental_models = list(type_two_mental_models)
        changed = True
        while changed:
            changed = False
            next_mental_models = []
            for i in range(len(type_one_mental_models)):
                first_mental_model = type_one_mental_models[i]
                mental_model_added = False
                for j in range(len(type_one_mental_models)):
                    if i == j:
                        continue

                    second_mental_model = type_one_mental_models[j]
                    differences = np.abs(first_mental_model - second_mental_model)

                    if np.count_nonzero(differences) == 1:
                        if np.sum(differences) == 2:
                            combined_mental_model = (first_mental_model + second_mental_model) / 2
                        else:
                            if np.sum(first_mental_model) < np.sum(second_mental_model):
                                combined_mental_model = first_mental_model
                            else:
                                combined_mental_model = second_mental_model

                        for existing_mental_model in next_mental_models:
                            if np.count_nonzero(combined_mental_model - existing_mental_model) == 0:
                                break
                        else:
                            next_mental_models.append(combined_mental_model)
                            changed = True
                        mental_model_added = True
                        # break

                if not mental_model_added:
                    for existing_mental_model in next_mental_models:
                        if np.count_nonzero(first_mental_model - existing_mental_model) == 0:
                            break
                    else:
                        next_mental_models.append(first_mental_model)

            if changed:
                type_one_mental_models = next_mental_models

        mental_models = type_one_mental_models
    else:
        mental_models = type_two_mental_models

    string_mental_models = []
    for mental_model in mental_models:
        string_mental_model = ''
        for value in mental_model:
            if value == 1:
                string_mental_model += 'T'
            elif value == -1:
                string_mental_model += 'F'
            else:
                string_mental_model += 'n'

        string_mental_models.append(string_mental_model)

    if allow_only_one_mental_model:
        if len(mental_models) == 1 and np.count_nonzero(mental_models[0]) > 0:
            return tree, string_mental_models[0]

        return False, None

    return tree, ','.join(string_mental_models)


if __name__ == '__main__':
    import functools
    from dataset.logic_tree import OperatorNode


    class SeparatorNode(OperatorNode):
        accepts_children = 2

        def __init__(self, *children):
            super(SeparatorNode, self).__init__('sep', *children)

        def evaluate(self, values):
            value = self._children[0].evaluate(values)
            for child in self._children[1:]:
                value = value and child.evaluate(values)

            return value

        def to_string(self):
            string = f'{self._children[0].to_string()}'
            for child in self._children[1:]:
                string += f' {self._operator_symbol} {child.to_string()}'
            return string


    test_for_one_mental_models = functools.partial(test_for_mental_models, allow_only_one_mental_model=True)
    test_for_mental_models_type_two = functools.partial(test_for_mental_models, type_one=False)
    test_for_one_mental_models_type_two = functools.partial(test_for_mental_models, type_one=False,
                                                            allow_only_one_mental_model=True)
    generate_random_sep_tree = functools.partial(generate_random_tree, root_node_cls=SeparatorNode)

    generate_and_save_trees('./data', 1000000, 2, 5,
                            test_for_one_mental_models, generate_random_sep_tree,
                            base_name='and_trees_single_mms')



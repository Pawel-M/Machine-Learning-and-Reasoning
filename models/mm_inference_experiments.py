def find_common_models(set_a, set_b):
    # the function simply returns all states that are possible in both mental models
    return set_a.intersection(set_b)

# Model 1 represents the sentence (A or B).
# This sentence can imply three different states:
# 1) A = True, B = False
# 2) A = False, B = True
# 3) A = True, B = True
# Representing these states in vectors:
# 1) (1, 0)
# 2) (0, 1)
# 3) (1, 1)

# Here, mm1 is the vector-representation of the sentence (A or B)
# Here, mm2 is the vector-representation of the sentence (A or not B)

mm1 = {(1, 0), (0, 1), (1, 1)}
mm2 = {(1, 0), (1, 1), (0, 0)}
common_models = find_common_models(mm1, mm2)
print(common_models)

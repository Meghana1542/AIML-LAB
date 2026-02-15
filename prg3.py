facts = {"A"}
rules = [
    ({"A"}, "B"),
    ({"B"}, "C"),
    ({"C"}, "D")
]

def forward_chaining(facts, rules):
    inferred = set(facts)
    added = True
    while added:
        added = False
        for condition, result in rules:
            if condition.issubset(inferred) and result not in inferred:
                inferred.add(result)
                added = True
    return sorted(list(inferred - facts))

def backward_chaining(goal, facts, rules, visited=None):
    if visited is None:
        visited = set()
    if goal in facts:
        return True
    if goal in visited:
        return False
    visited.add(goal)
    for condition, result in rules:
        if result == goal:
            if all(backward_chaining(g, facts, rules, visited) for g in condition):
                return True
    return False

def resolve(clause1, clause2):
    for literal in clause1:
        if ("~" + literal) in clause2:
            new_clause = (clause1 - {literal}) | (clause2 - {("~" + literal)})
            return new_clause
        if literal.startswith("~") and literal[1:] in clause2:
            new_clause = (clause1 - {literal}) | (clause2 - {literal[1:]})
            return new_clause
    return None

def resolution(kb, query):
    clauses = kb + [{ "~" + query }]
    new = []
    while True:
        n = len(clauses)
        for i in range(n):
            for j in range(i+1, n):
                resolvent = resolve(clauses[i], clauses[j])
                if resolvent == set():
                    return True
                if resolvent and resolvent not in clauses:
                    new.append(resolvent)
        if all(c in clauses for c in new):
            return False
        clauses.extend(new)

forward_result = forward_chaining(facts, rules)
backward_result = backward_chaining("D", facts, rules)

knowledge_base = [
    {"A", "B"},
    {"~A"},
    {"~B", "C"}
]

resolution_result = resolution(knowledge_base, "C")

print("forward_chaining_result:", forward_result)
print("backward_chaining_result:", backward_result)
print("resolution_result:", resolution_result)

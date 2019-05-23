import sys
"""
Based on Guo Li
"""

class BinPackPartitioner:
    def __init__(self):
        pass
        
    def bin_pack(self, sizes, budget):
        lst = list(reversed(sorted([(size, name) for name, size in sizes.items()])))
        budget = max(budget, lst[0][0])
        budgets = []
        indexes = dict()
        for size, name in lst:
            ok = False
            for i, b in enumerate(budgets):
                if b >= size:
                    budgets[i] -= size
                    indexes[name] = i
                    ok = True
                    break
            if not ok:
                budgets.append(budget - size)
                indexes[name] = len(budgets) - 1
        return indexes
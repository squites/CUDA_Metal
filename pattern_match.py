# A single pattern matching rule. 
# This is the container for 1 rule
class Rule:
    def __init__(self, name, fpattern, fbuilder):
        self.name = name
        self.fpattern = fpattern # pattern function to call: node -> dict
        self.fbuilder = fbuilder # builder function to call: dict -> new_node

    def match(self):
        """ does the pattern match? if yes: Build. if no: return None"""
        pass

# applies rules to the subtree. This needs to hold lists of rules.
class Rewriter:
    """ takes a list of rules and applies them to the subtree """
    def __init__(self, rules):
        self.rules = rules

    def rewrite(self, root):
        pass

    def rewrite_once(self):
        pass


# pattern functions:
def pat_GlobalThreadIdx(node):
    pass

def pat_Something(node):
    pass

# builder functions:
def build_GlobalThreadIdx(params):
    """ create node GlobalThreadIdx(params) """     
    pass

# create the rules
rule1 = Rule(
    "global_thread_idx",
    pat_GlobalThreadIdx,
    build_GlobalThreadIdx
)

# Calling the rewriter
# e.g.:
# expr = Add(operands=[Mul(operands=[ThreadIdx(dim='x')]), Mul(operands=[BlockIdx(dim='x'), BlockDim(dim='x')])])

# Create the rewriter
# rewriter = Rewriter([rule1, rule2, rule3, ...])

# Apply the rules
# result = rewriter.rewrite(expr)
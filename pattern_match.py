from ast_builder import ThreadIdx, BlockIdx, BlockDim, GlobalThreadIdx

# adding high-level semantic nodes to the expressions
# Add(operands=[Mul(operands=[ThreadIdx(dim='x')]), Mul(operands=[BlockIdx(dim='x'), BlockDim(dim='x')])]) 
# -> GlobalThreadIdx()
# Move this to new .py file!
class Rule:
    def __init__(self, name, fpattern, fbuilder):
        self.name = name
        self.fpattern = fpattern
        self.fbuilder = fbuilder
    
    def match(self, node):
        print("matching: ", node)
        binds = self.fpattern(node)
        if binds is not None:
            return self.fbuilder(binds)
        return None

class Rewriter:
    def __init__(self, rules):
        self.rules = rules

    def rewrite(self, node):
        print("Rewriting... ", node)
        if not hasattr(node, "operands"):
            print("LEAF!")
            return node
        nodeops = [self.rewrite(child) for child in node.operands]
        print("nodeops: ", nodeops)

        for rule in self.rules:
            print("RULE: ", rule.name)
            x = rule.match(node)
            print("x: ", x)
            if x is not None:
                print("MATCH!")
                return x
        print("NO MATCH!")
        return node

# pattern functions:
def pat_GlobalThreadIdx(node):
    print("Pattern function: ", node)
    if not isinstance(node, Add):
        return None
    if len(node.operands) != 2:
        return None
    l, r = node.operands
    print(f"l {l}\nr {r}")
    if not isinstance(l, Mul) or not isinstance(r, Mul):
        return None

    def isthread(x):
        #print("isthread:", x)
        return (len(x.operands) == 1 and isinstance(x.operands[0], ThreadIdx))

    def isblock(x):
        #print("isblock: ", x) 
        if len(x.operands) != 2:
            return False
        a,b = x.operands
        return (isinstance(a, BlockIdx) and isinstance(b, BlockDim)) or (isinstance(a, BlockDim) and isinstance(b, BlockIdx))

    if isthread(l) and isblock(r):
        return {
            "dim": l.operands[0].dim
        }
    else:
        return None

# builder functions:
def build_GlobalThreadIdx(binds):
    """ create node GlobalThreadIdx(params) """     
    print("Building Node: ", binds)
    return GlobalThreadIdx(dim=binds["dim"])

def IRrewrite(subtree):
    print("IR subtree: ", subtree)
    
    # rules
    rule1 = Rule(
        "GlobalThreadIdx",  # name
        pat_GlobalThreadIdx,  # pattern function
        build_GlobalThreadIdx # builder function
    )

    rewriter = Rewriter([rule1])
    new_tree = rewriter.rewrite(subtree)
    print("RESULT: ", new_tree)
    return new_tree

# Calling the rewriter
# e.g.:
# expr = Add(operands=[Mul(operands=[ThreadIdx(dim='x')]), Mul(operands=[BlockIdx(dim='x'), BlockDim(dim='x')])])

# Create the rewriter
# rewriter = Rewriter([rule1, rule2, rule3, ...])

# Apply the rules
# result = rewriter.rewrite(expr)
import sys
import re
from pycparser import c_parser, c_ast, c_generator


class TACConverter:
    def __init__(self):
        self.tac = []
        self.temp_count = 0
        self.generator = c_generator.CGenerator()
        self.label_map = {}
        self.next_label_id = 0
        self.loop_stack = []  # Стек для хранения информации о текущих циклах
        self.curr_temp_var_id = 0
        self.curr_var_values = {}

    def new_temp(self):
        name = f"t{self.temp_count}"
        self.temp_count += 1
        return name

    def new_label(self):
        label = f"L{self.next_label_id}"
        self.next_label_id += 1
        return label

    def to_rpn(self, expr):
        if isinstance(expr, c_ast.Constant):
            return expr.value
        elif isinstance(expr, c_ast.ID):
            return expr.name
        elif isinstance(expr, c_ast.ArrayRef):
            array_name = self.to_rpn(expr.name)
            subscript = self.to_rpn(expr.subscript)
            return f"{array_name} {subscript} []"
        elif isinstance(expr, c_ast.UnaryOp):
            operand = self.to_rpn(expr.expr)
            op = expr.op

            if op == "+" or op == "-":
                op = "_" + op
            if op == "&":
                raise ValueError(f"Address-of operator (&) not supported: {self.generator.visit(expr)}")
            elif op == "p++":
                temp_var_name = "'t" + str(self.curr_temp_var_id)
                self.curr_temp_var_id += 1
                self.tac.append(('assign', temp_var_name, f"{operand}"))
                self.tac.append(('assign', operand, f"{operand} 1 +"))
                return temp_var_name
            elif op == "++":
                temp_var_name = "'t" + str(self.curr_temp_var_id)
                self.curr_temp_var_id += 1
                self.tac.append(('assign', temp_var_name, f"{operand} 1 +"))
                self.tac.append(('assign', operand, f"{operand} 1 +"))
                return temp_var_name
            elif op == "p--":
                temp_var_name = "'t" + str(self.curr_temp_var_id)
                self.curr_temp_var_id += 1
                self.tac.append(('assign', temp_var_name, f"{operand}"))
                self.tac.append(('assign', operand, f"{operand} 1 -"))
                return temp_var_name
            elif op == "--":
                temp_var_name = "'t" + str(self.curr_temp_var_id)
                self.curr_temp_var_id += 1
                self.tac.append(('assign', temp_var_name, f"{operand} 1 -"))
                self.tac.append(('assign', operand, f"{operand} 1 -"))
                return temp_var_name

            return f"{operand} {op}"
        elif isinstance(expr, c_ast.BinaryOp):
            left = self.to_rpn(expr.left)
            right = self.to_rpn(expr.right)
            op = expr.op
            return f"{left} {right} {op}"
        elif isinstance(expr, c_ast.TernaryOp):
            cond = self.to_rpn(expr.cond)
            result_iftrue = self.to_rpn(expr.iftrue)
            result_iffalse = self.to_rpn(expr.iffalse)

            return f"{cond} {result_iftrue} {result_iffalse} ?:"
        elif isinstance(expr, c_ast.Assignment):
            left = self.to_rpn(expr.lvalue)
            right = self.to_rpn(expr.rvalue)
            op = expr.op
            return f"{left} {right} {op}"
        else:
            return self.generator.visit(expr)

    def add_assignment(self, lhs, rhs, assign_type):
        lhs_rpn = lhs
        if isinstance(lhs, c_ast.ArrayRef):
            lhs_rpn = self.to_rpn(lhs)
        elif isinstance(lhs, c_ast.ID):
            lhs_rpn = lhs.name

        rhs_rpn = self.to_rpn(rhs)
        if assign_type != "=":
            rhs_rpn = lhs_rpn + " " + rhs_rpn + " " + assign_type[0]

        if not isinstance(lhs, c_ast.ArrayRef):
            self.tac.append(('assign', lhs_rpn, rhs_rpn))
        else:
            self.tac.append(('array_assign', lhs_rpn, rhs_rpn))

    def add_if(self, cond, true_label):
        cond_rpn = self.to_rpn(cond)
        self.tac.append(('if', cond_rpn, true_label))

    def add_goto(self, label):
        self.tac.append(('goto', label))

    def add_label(self, label):
        self.label_map[label] = len(self.tac)

    def add_return(self, expr):
        if expr:
            expr_rpn = self.to_rpn(expr)
            self.tac.append(('return', expr_rpn))
        else:
            self.tac.append(('return', ''))

    def visit(self, node):
        if isinstance(node, c_ast.Assignment):
            self.add_assignment(node.lvalue, node.rvalue, node.op)
        elif isinstance(node, c_ast.UnaryOp):
            rhs_rpn = self.to_rpn(node.expr)
            if node.op == "p++" or node.op == "++":
                self.tac.append(('assign', self.to_rpn(node.expr), f"{rhs_rpn} {1} +"))
            elif node.op == "p--" or node.op == "--":
                self.tac.append(('assign', self.to_rpn(node.expr), f"{rhs_rpn} {1} -"))

        elif isinstance(node, c_ast.If):
            true_label = self.new_label()
            end_label = self.new_label()

            # Condition and branch
            self.add_if(node.cond, true_label)

            # False branch
            if node.iffalse:
                self.visit(node.iffalse)
            self.add_goto(end_label)

            # True branch
            self.add_label(true_label)
            self.visit(node.iftrue)

            # End label
            self.add_label(end_label)

        elif isinstance(node, c_ast.While):
            start_label = self.new_label()
            body_label = self.new_label()
            end_label = self.new_label()

            # Push loop info to stack (end_label for break, start_label for continue)
            self.loop_stack.append((end_label, start_label))

            # Loop start
            self.add_label(start_label)
            self.add_if(node.cond, body_label)
            self.add_goto(end_label)

            # Loop body
            self.add_label(body_label)
            self.visit(node.stmt)
            self.add_goto(start_label)

            # End label
            self.add_label(end_label)
            self.loop_stack.pop()  # Pop loop info after processing

        elif isinstance(node, c_ast.For):
            init_label = self.new_label()
            start_label = self.new_label()
            body_label = self.new_label()
            next_label = self.new_label()
            end_label = self.new_label()

            # Push loop info (end_label for break, next_label for continue)
            self.loop_stack.append((end_label, next_label))

            # Initialization
            if node.init:
                self.visit(node.init)
            self.add_label(init_label)

            # Condition
            self.add_label(start_label)
            if node.cond:
                self.add_if(node.cond, body_label)
                self.add_goto(end_label)
            else:
                # If no condition, always enter loop
                self.add_goto(body_label)

            # Loop body
            self.add_label(body_label)
            self.visit(node.stmt)

            # Increment
            self.add_label(next_label)
            if node.next:
                self.visit(node.next)
            self.add_goto(start_label)

            # End label
            self.add_label(end_label)
            self.loop_stack.pop()  # Pop loop info after processing

        elif isinstance(node, c_ast.DoWhile):
            body_label = self.new_label()
            cond_label = self.new_label()
            end_label = self.new_label()

            # Push loop info (end_label for break, cond_label for continue)
            self.loop_stack.append((end_label, cond_label))

            # Loop body
            self.add_label(body_label)
            self.visit(node.stmt)

            # Condition
            self.add_label(cond_label)
            if node.cond:
                self.add_if(node.cond, body_label)
                self.add_goto(end_label)
            else:
                # If no condition, infinite loop
                self.add_goto(body_label)

            # End label
            self.add_label(end_label)
            self.loop_stack.pop()  # Pop loop info after processing

        elif isinstance(node, c_ast.Break):
            if self.loop_stack:
                end_label = self.loop_stack[-1][0]
                self.add_goto(end_label)
            # Else: break outside loop - error, but we'll ignore

        elif isinstance(node, c_ast.Continue):
            if self.loop_stack:
                continue_label = self.loop_stack[-1][1]
                self.add_goto(continue_label)
            # Else: continue outside loop - error, but we'll ignore

        elif isinstance(node, c_ast.Compound):
            if node.block_items:
                for item in node.block_items:
                    self.visit(item)
        elif isinstance(node, c_ast.Return):
            self.add_return(node.expr)
        elif isinstance(node, c_ast.Decl):
            if node.init:
                self.add_assignment(node.name, node.init, "=")
        elif isinstance(node, c_ast.DeclList):
            for decl in node.decls:
                self.visit(decl)


def build_cfg(tac, label_map):
    nodes = []
    edges = []
    final_node = len(tac)

    # Create nodes for each TAC instruction
    nodes = list(range(len(tac)))

    # Build edges
    for i, instr in enumerate(tac):
        if instr[0] == 'assign':
            # Assignment always goes to next instruction
            if i < len(tac):
                edges.append((i, i + 1, f"{instr[2]} {instr[1]} ="))

        elif instr[0] == 'array_assign':
            # Assignment always goes to next instruction
            if i < len(tac):
                edges.append((i, i + 1, f"{instr[2]} {instr[1][:-3]} []="))

        elif instr[0] == 'if':
            cond = instr[1]
            true_label = instr[2]
            true_index = label_map[true_label]

            # True branch
            edges.append((i, true_index, cond))

            # False branch (next instruction)
            if i + 1 < len(tac):
                edges.append((i, i + 1, f"{cond} !"))

        elif instr[0] == 'goto':
            target_label = instr[1]
            target_index = label_map[target_label]
            edges.append((i, target_index, "skip"))

        elif instr[0] == 'return':
            expr = instr[1]
            action = f"{expr} return" if expr else "return"
            edges.append((i, final_node, action))

    return nodes, edges, final_node


def generate_dot(nodes, edges, final_node):
    dot = ["digraph PG {"]
    dot.append("  node [shape=circle];")
    dot.append(f"  q{final_node} [shape=doublecircle];")

    for node in nodes:
        dot.append(f"  q{node};")
    dot.append(f"  q{final_node};")

    for src, dst, action in edges:
        dot.append(f'  q{src} -> q{dst} [label="{action}"];')

    dot.append("}")
    return "\n".join(dot)


def optimize(nodes, edges, final_node):
    new_nodes = []
    new_edges = []
    new_final_node = final_node

    for e in edges:
        if e[2] == "skip":
            continue
        elif "return" in e[2]:
            continue

        ends = [e[1]]

        push_forward = True
        while push_forward:
            push_forward = False

            new_ends = []
            for end in ends:
                tails = [e1[1] for e1 in edges if e1[0] == end and (e1[2] == "skip" or "return" in e1[2])]

                if tails:
                    push_forward = True
                    new_ends += tails
                else:
                    new_ends.append(end)

            ends = new_ends

        for end in ends:
            new_edges.append((e[0], end, e[2]))

    # new_edges = edges

    for e in new_edges:
        new_nodes.append(e[0])
        new_nodes.append(e[1])

    new_nodes = list(set(new_nodes))

    # Remove false begin branches
    nodes_to_delete = []

    buffer_nodes = [n for n in new_nodes if n != 0 and not n in [n2 for n1, n2, n3 in new_edges]]
    while buffer_nodes:
        nodes_to_delete += buffer_nodes
        buffer_nodes = [n for n in new_nodes if n != 0 and not n in nodes_to_delete and not [e for e in new_edges if
                                                                                             e[1] == n and not e[
                                                                                                                   0] in nodes_to_delete]]

    new_nodes = [n for n in new_nodes if not n in nodes_to_delete]
    new_edges = [e for e in new_edges if e[0] in new_nodes and e[1] in new_nodes]

    return new_nodes, new_edges, new_final_node


def c_to_pg(c_code, func_name='main'):
    parser = c_parser.CParser()
    ast = parser.parse(c_code)
    converter = TACConverter()

    if isinstance(ast, c_ast.FileAST):
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef) and ext.decl.name == func_name:
                converter.visit(ext.body)

    tac = converter.tac
    nodes, edges, final_node = build_cfg(tac, converter.label_map)
    nodes, edges, final_node = optimize(nodes, edges, final_node)

    dot = generate_dot(nodes, edges, final_node)
    return dot


def clear_code(code_text):
    code_text = re.sub('\/\*[\w\W]*?\*\/', '', code_text)
    # code_text = re.sub(r'return[\w\W]*?;', '', code_text)
    code_text = re.sub('\/\/.*', '', code_text)
    code_text = re.sub('^[ \t]*#(include|pragma|if|else|elif|endif|warning|error).*', '', code_text)
    code_text = re.sub('^[ \t]*#(define|ifdef|ifndef|elif|elifdef|elifndef)(.*\\\n)*.*', '', code_text)

    return code_text


if __name__ == "__main__":
    code_file_path = "file.c"
    if len(sys.argv) >= 2:
        code_file_path = sys.argv[1]

    func_name = "main"
    if len(sys.argv) >= 3:
        func_name = sys.argv[2]

    save_pg_file_path = "pg.dot"
    if len(sys.argv) >= 4:
        save_pg_file_path = sys.argv[3]

    with open(code_file_path, "r") as file:
        c_code = file.read()
        c_code = clear_code(c_code)
        pg_dot = c_to_pg(c_code, func_name)
        with open(save_pg_file_path, 'w') as f:
            f.write(pg_dot)
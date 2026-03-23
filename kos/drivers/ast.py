"""
KOS V2.0 — AST Driver (Code Parser).

Parses source code structure (functions, variables, imports)
into graph topology. Links function definitions to their
parameter and return type concepts.
"""


class ASTDriver:
    def __init__(self, kernel, lexicon):
        self.kernel = kernel
        self.lexicon = lexicon

    def ingest_code(self, code_str: str):
        """
        Simulates Tree-Sitter parsing logic linking functions to concepts.
        Example: def calculate_efficiency(solar_cell): ...
        """
        import ast as python_ast

        try:
            tree = python_ast.parse(code_str)
        except SyntaxError:
            return

        for node in python_ast.walk(tree):
            if isinstance(node, python_ast.FunctionDef):
                # Split function name by underscores into concepts
                parts = node.name.lower().split('_')
                func_concepts = [self.lexicon.get_or_create_id(p)
                                 for p in parts if len(p) > 2]

                # Link function concepts together
                for c1 in func_concepts:
                    for c2 in func_concepts:
                        if c1 != c2:
                            self.kernel.add_connection(
                                c1, c2, 0.7,
                                f"[AST] Function: {node.name}")

                # Link function to its arguments
                for arg in node.args.args:
                    arg_parts = arg.arg.lower().split('_')
                    arg_concepts = [self.lexicon.get_or_create_id(p)
                                    for p in arg_parts if len(p) > 2]
                    for fc in func_concepts:
                        for ac in arg_concepts:
                            self.kernel.add_connection(
                                fc, ac, 0.9,
                                f"[AST] {node.name}({arg.arg})")

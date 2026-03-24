"""
KASM Interpreter — Executes a KASM AST against the VSA engine.

Pipeline:  .kasm source -> Lexer -> Parser -> AST -> Interpreter -> VSA results

The interpreter walks the AST and maps each node to the corresponding
KASMEngine operation (NODE, BIND, SUPERPOSE, PERMUTE, RESONATE, etc.)
"""

import sys
import time
import numpy as np
from typing import Optional, TextIO

from .vsa import KASMEngine
from .lexer import tokenize
from .parser import (
    Parser, ProgramNode, NodeDeclNode, BindNode, SuperposeNode,
    PermuteNode, ResonateNode, UnbindNode, CleanupNode, PrintNode,
    DimNode, SeedNode, IdentNode, BindExprNode, SuperposeExprNode,
    PermuteExprNode, Expr
)


class RuntimeError(Exception):
    """KASM runtime error."""
    pass


class KASMInterpreter:
    """
    Executes a KASM program.

    Usage:
        interpreter = KASMInterpreter()
        interpreter.run_file("example.kasm")
        # or
        interpreter.run_source("NODE sun, planet\\nRESONATE sun <=> planet")
    """

    def __init__(self, dimensions: int = 10_000, seed: Optional[int] = None,
                 output: Optional[TextIO] = None):
        self.engine = KASMEngine(dimensions=dimensions, seed=seed)
        self.output = output or sys.stdout
        self._pending_dim: Optional[int] = None
        self._pending_seed: Optional[int] = None

    def _emit(self, text: str):
        """Write output."""
        self.output.write(text + "\n")
        self.output.flush()

    # ── Public API ───────────────────────────────────────────────────

    def run_source(self, source: str) -> KASMEngine:
        """Parse and execute KASM source code. Returns the engine state."""
        tokens = tokenize(source)
        parser = Parser(tokens)
        ast = parser.parse()
        self.execute(ast)
        return self.engine

    def run_file(self, filepath: str) -> KASMEngine:
        """Parse and execute a .kasm file. Returns the engine state."""
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        return self.run_source(source)

    # ── AST Execution ────────────────────────────────────────────────

    def execute(self, program: ProgramNode):
        """Execute a parsed KASM program."""
        # First pass: collect DIM and SEED directives
        for stmt in program.statements:
            if isinstance(stmt, DimNode):
                self._pending_dim = stmt.dimensions
            elif isinstance(stmt, SeedNode):
                self._pending_seed = stmt.seed

        # Reinitialize engine if DIM or SEED were specified
        if self._pending_dim is not None or self._pending_seed is not None:
            dim = self._pending_dim or self.engine.D
            seed = self._pending_seed
            self.engine = KASMEngine(dimensions=dim, seed=seed)

        # Second pass: execute all statements
        t_start = time.perf_counter()

        for stmt in program.statements:
            if isinstance(stmt, (DimNode, SeedNode)):
                continue  # already handled
            self._exec_statement(stmt)

        elapsed = (time.perf_counter() - t_start) * 1000
        self._emit(f"\n[KASM] Execution complete in {elapsed:.2f} ms")
        self._emit(f"[KASM] Symbols: {len(self.engine.symbols)} | "
                   f"Dimensions: {self.engine.D:,} | "
                   f"Memory: {self.engine.stats()['memory_mb']} MB")

    def _exec_statement(self, stmt):
        """Dispatch a statement to its handler."""
        if isinstance(stmt, NodeDeclNode):
            self._exec_node(stmt)
        elif isinstance(stmt, BindNode):
            self._exec_bind(stmt)
        elif isinstance(stmt, SuperposeNode):
            self._exec_superpose(stmt)
        elif isinstance(stmt, PermuteNode):
            self._exec_permute(stmt)
        elif isinstance(stmt, ResonateNode):
            self._exec_resonate(stmt)
        elif isinstance(stmt, UnbindNode):
            self._exec_unbind(stmt)
        elif isinstance(stmt, CleanupNode):
            self._exec_cleanup(stmt)
        elif isinstance(stmt, PrintNode):
            self._exec_print(stmt)
        else:
            raise RuntimeError(f"Unknown statement type: {type(stmt).__name__}")

    # ── Statement Handlers ───────────────────────────────────────────

    def _exec_node(self, stmt: NodeDeclNode):
        for name in stmt.names:
            if name in self.engine.symbols:
                raise RuntimeError(f"Symbol '{name}' already defined")
            self.engine.node(name)
        names_str = ", ".join(stmt.names)
        self._emit(f"  NODE  {names_str}")

    def _exec_bind(self, stmt: BindNode):
        vec = self._eval_expr(stmt.expr)
        self.engine.store(stmt.target, vec)
        self._emit(f"  BIND  {stmt.target} = <{self.engine.D}-D vector>")

    def _exec_superpose(self, stmt: SuperposeNode):
        vec = self._eval_expr(stmt.expr)
        self.engine.store(stmt.target, vec)
        self._emit(f"  SUPER {stmt.target} = <{self.engine.D}-D vector>")

    def _exec_permute(self, stmt: PermuteNode):
        vec = self._eval_expr(stmt.expr)
        self.engine.store(stmt.target, vec)
        self._emit(f"  PERM  {stmt.target} = <{self.engine.D}-D vector>")

    def _exec_resonate(self, stmt: ResonateNode):
        vec_a = self._eval_expr(stmt.left)
        vec_b = self._eval_expr(stmt.right)
        score = self.engine.resonate(vec_a, vec_b)

        left_name = stmt.left.name if isinstance(stmt.left, IdentNode) else "expr"
        right_name = stmt.right.name if isinstance(stmt.right, IdentNode) else "expr"

        bar_len = int(abs(score) * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)

        self._emit(f"  <=>   {left_name} <=> {right_name}")
        self._emit(f"        Score: {score:+.4f}  [{bar}]")

        if abs(score) > 0.30:
            self._emit(f"        >> STRONG MATCH")
        elif abs(score) > 0.15:
            self._emit(f"        >> Weak similarity")
        else:
            self._emit(f"        >> Orthogonal (unrelated)")

        if stmt.store_as:
            # Store the score... but we need a vector. Store vec_b scaled?
            # For now, we just print the result.
            self._emit(f"        (stored as '{stmt.store_as}')")

    def _exec_unbind(self, stmt: UnbindNode):
        vec = self._eval_expr(stmt.expr)
        self.engine.store(stmt.target, vec)
        self._emit(f"  UNBND {stmt.target} = <{self.engine.D}-D vector>")

    def _exec_cleanup(self, stmt: CleanupNode):
        vec = self._eval_expr(stmt.expr)
        matches = self.engine.cleanup(vec, threshold=0.05)

        # Exclude the query variable itself and intermediate bindings
        # Only show NODE-declared atomic concepts and SUPERPOSE composites
        query_name = stmt.expr.name if isinstance(stmt.expr, IdentNode) else None
        filtered = [(n, s) for n, s in matches
                    if n != query_name and abs(s) < 0.99]

        self._emit(f"  CLEAN Nearest known symbols:")
        if not filtered:
            self._emit(f"        (no matches above threshold)")
        else:
            for i, (name, score) in enumerate(filtered[:8]):
                rank_marker = " <<" if i == 0 else ""
                bar_len = int(abs(score) * 40)
                bar = "#" * bar_len
                self._emit(f"        {i+1}. {name:25s} {score:+.4f} [{bar}]{rank_marker}")

        if stmt.store_as and matches:
            best_name = matches[0][0]
            self.engine.store(stmt.store_as, self.engine.get(best_name).copy())
            self._emit(f"        -> stored '{matches[0][0]}' as '{stmt.store_as}'")

    def _exec_print(self, stmt: PrintNode):
        if stmt.is_string:
            self._emit(f"\n  >>> {stmt.value}")
        else:
            expr = stmt.value
            if isinstance(expr, IdentNode):
                name = expr.name
                if name in self.engine.symbols:
                    vec = self.engine.get(name)
                    self._emit(f"  INFO  {name}: D={len(vec)}, "
                               f"sum={int(vec.sum())}, "
                               f"dtype={vec.dtype}")
                else:
                    self._emit(f"  INFO  {name}: (undefined)")
            else:
                self._emit(f"  INFO  <expression>")

    # ── Expression Evaluator ─────────────────────────────────────────

    def _eval_expr(self, expr: Expr) -> np.ndarray:
        """Recursively evaluate an expression to a vector."""
        if isinstance(expr, IdentNode):
            if expr.name not in self.engine.symbols:
                raise RuntimeError(f"Undefined symbol: '{expr.name}'")
            return self.engine.get(expr.name)

        elif isinstance(expr, BindExprNode):
            left = self._eval_expr(expr.left)
            right = self._eval_expr(expr.right)
            return self.engine.bind(left, right)

        elif isinstance(expr, SuperposeExprNode):
            vecs = [self._eval_expr(op) for op in expr.operands]
            return self.engine.superpose(*vecs)

        elif isinstance(expr, PermuteExprNode):
            vec = self._eval_expr(expr.operand)
            shifts = expr.shifts if expr.direction == 'right' else -expr.shifts
            return self.engine.permute(vec, shifts)

        else:
            raise RuntimeError(f"Unknown expression type: {type(expr).__name__}")


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    """Run a .kasm file from the command line."""
    if len(sys.argv) < 2:
        print("Usage: python -m kasm.interpreter <file.kasm>")
        print("       python -m kasm.interpreter --eval 'NODE a, b'")
        sys.exit(1)

    if sys.argv[1] == '--eval':
        source = ' '.join(sys.argv[2:])
        interpreter = KASMInterpreter()
        interpreter.run_source(source)
    else:
        filepath = sys.argv[1]
        interpreter = KASMInterpreter()
        interpreter.run_file(filepath)


if __name__ == "__main__":
    main()

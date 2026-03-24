"""
KASM Parser — Transforms token stream into an Abstract Syntax Tree (AST).

AST Node Types:
    ProgramNode     — root: list of statements
    NodeDeclNode    — NODE name1, name2, ...
    BindNode        — BIND target = expr * expr
    SuperposeNode   — SUPERPOSE target = expr + expr + ...
    PermuteNode     — PERMUTE target = expr >> n
    ResonateNode    — RESONATE expr <=> expr [-> name]
    UnbindNode      — UNBIND target = expr * expr
    CleanupNode     — CLEANUP expr [-> name]
    PrintNode       — PRINT expr_or_string
    DimNode         — DIM integer
    SeedNode        — SEED integer
    IdentNode       — variable reference
    BindExprNode    — inline a * b expression
    SuperposeExprNode — inline a + b + c expression
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from .lexer import Token, TokenType, LexerError


# ── AST Nodes ────────────────────────────────────────────────────────

@dataclass
class IdentNode:
    name: str

@dataclass
class BindExprNode:
    left: 'Expr'
    right: 'Expr'

@dataclass
class SuperposeExprNode:
    operands: List['Expr']

@dataclass
class PermuteExprNode:
    operand: 'Expr'
    shifts: int
    direction: str  # 'right' or 'left'

# Union of all expression types
Expr = Union[IdentNode, BindExprNode, SuperposeExprNode, PermuteExprNode]

@dataclass
class NodeDeclNode:
    names: List[str]

@dataclass
class BindNode:
    target: str
    expr: Expr

@dataclass
class SuperposeNode:
    target: str
    expr: Expr

@dataclass
class PermuteNode:
    target: str
    expr: Expr

@dataclass
class UnbindNode:
    target: str
    expr: Expr

@dataclass
class ResonateNode:
    left: Expr
    right: Expr
    store_as: Optional[str] = None

@dataclass
class CleanupNode:
    expr: Expr
    store_as: Optional[str] = None

@dataclass
class PrintNode:
    value: Union[str, Expr]
    is_string: bool = False

@dataclass
class DimNode:
    dimensions: int

@dataclass
class SeedNode:
    seed: int

# Statement union
Statement = Union[
    NodeDeclNode, BindNode, SuperposeNode, PermuteNode,
    UnbindNode, ResonateNode, CleanupNode, PrintNode,
    DimNode, SeedNode
]

@dataclass
class ProgramNode:
    statements: List[Statement] = field(default_factory=list)


# ── Parser ───────────────────────────────────────────────────────────

class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(f"Parse error at line {token.line}, col {token.col}: {message}")
        self.token = token


class Parser:
    """
    Recursive descent parser for KASM grammar v0.1.

    Grammar (simplified EBNF):
        program     = { statement NEWLINE }
        statement   = node_decl | bind_stmt | superpose_stmt | permute_stmt
                    | resonate_stmt | unbind_stmt | cleanup_stmt | print_stmt
                    | dim_stmt | seed_stmt
        node_decl   = "NODE" IDENT { "," IDENT }
        bind_stmt   = "BIND" IDENT "=" expr "*" expr
        superpose_stmt = "SUPERPOSE" IDENT "=" expr { "+" expr }
        permute_stmt = "PERMUTE" IDENT "=" expr (">>" | "<<") INTEGER
        resonate_stmt = "RESONATE" expr "<=>" expr [ "->" IDENT ]
        unbind_stmt = "UNBIND" IDENT "=" expr "*" expr
        cleanup_stmt = "CLEANUP" expr [ "->" IDENT ]
        print_stmt  = "PRINT" (STRING | expr)
        dim_stmt    = "DIM" INTEGER
        seed_stmt   = "SEED" INTEGER
        expr        = atom { "*" atom }     // bind has highest precedence
        atom        = IDENT | "(" expr ")"
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, token_type: TokenType) -> Token:
        tok = self.peek()
        if tok.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {tok.type.name} ('{tok.value}')",
                tok
            )
        return self.advance()

    def at(self, *types: TokenType) -> bool:
        return self.peek().type in types

    def skip_newlines(self):
        while self.at(TokenType.NEWLINE):
            self.advance()

    # ── Top-level ────────────────────────────────────────────────────

    def parse(self) -> ProgramNode:
        program = ProgramNode()
        self.skip_newlines()

        while not self.at(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                program.statements.append(stmt)
            # Consume statement terminator (newline or EOF)
            if self.at(TokenType.NEWLINE):
                self.skip_newlines()
            elif not self.at(TokenType.EOF):
                raise ParseError(
                    f"Expected newline or EOF after statement, got {self.peek().type.name}",
                    self.peek()
                )

        return program

    # ── Statement dispatch ───────────────────────────────────────────

    def parse_statement(self) -> Optional[Statement]:
        tok = self.peek()

        if tok.type == TokenType.NODE:
            return self.parse_node_decl()
        elif tok.type == TokenType.BIND:
            return self.parse_bind()
        elif tok.type == TokenType.SUPERPOSE:
            return self.parse_superpose()
        elif tok.type == TokenType.PERMUTE:
            return self.parse_permute()
        elif tok.type == TokenType.RESONATE:
            return self.parse_resonate()
        elif tok.type == TokenType.UNBIND:
            return self.parse_unbind()
        elif tok.type == TokenType.CLEANUP:
            return self.parse_cleanup()
        elif tok.type == TokenType.PRINT:
            return self.parse_print()
        elif tok.type == TokenType.DIM:
            return self.parse_dim()
        elif tok.type == TokenType.SEED:
            return self.parse_seed()
        elif tok.type == TokenType.NEWLINE:
            return None  # blank line
        else:
            raise ParseError(f"Unexpected token: {tok.type.name} ('{tok.value}')", tok)

    # ── Statement parsers ────────────────────────────────────────────

    def parse_node_decl(self) -> NodeDeclNode:
        self.expect(TokenType.NODE)
        names = [self.expect(TokenType.IDENT).value]
        while self.at(TokenType.COMMA):
            self.advance()  # skip comma
            names.append(self.expect(TokenType.IDENT).value)
        return NodeDeclNode(names=names)

    def parse_bind(self) -> BindNode:
        self.expect(TokenType.BIND)
        target = self.expect(TokenType.IDENT).value
        self.expect(TokenType.EQUALS)
        expr = self.parse_expr()
        # Ensure it's a bind expression
        if not isinstance(expr, BindExprNode):
            raise ParseError("BIND requires a * expression", self.peek())
        return BindNode(target=target, expr=expr)

    def parse_superpose(self) -> SuperposeNode:
        self.expect(TokenType.SUPERPOSE)
        target = self.expect(TokenType.IDENT).value
        self.expect(TokenType.EQUALS)
        expr = self.parse_superpose_expr()
        return SuperposeNode(target=target, expr=expr)

    def parse_permute(self) -> PermuteNode:
        self.expect(TokenType.PERMUTE)
        target = self.expect(TokenType.IDENT).value
        self.expect(TokenType.EQUALS)
        operand = self.parse_atom()
        if self.at(TokenType.SHIFT_RIGHT):
            self.advance()
            direction = 'right'
        elif self.at(TokenType.SHIFT_LEFT):
            self.advance()
            direction = 'left'
        else:
            raise ParseError("PERMUTE requires >> or <<", self.peek())
        shifts = int(self.expect(TokenType.INTEGER).value)
        return PermuteNode(
            target=target,
            expr=PermuteExprNode(operand=operand, shifts=shifts, direction=direction)
        )

    def parse_resonate(self) -> ResonateNode:
        self.expect(TokenType.RESONATE)
        left = self.parse_atom()
        self.expect(TokenType.ARROW_DOUBLE)
        right = self.parse_atom()
        store_as = None
        if self.at(TokenType.ARROW_RIGHT):
            self.advance()
            store_as = self.expect(TokenType.IDENT).value
        return ResonateNode(left=left, right=right, store_as=store_as)

    def parse_unbind(self) -> UnbindNode:
        self.expect(TokenType.UNBIND)
        target = self.expect(TokenType.IDENT).value
        self.expect(TokenType.EQUALS)
        expr = self.parse_expr()
        if not isinstance(expr, BindExprNode):
            raise ParseError("UNBIND requires a * expression", self.peek())
        return UnbindNode(target=target, expr=expr)

    def parse_cleanup(self) -> CleanupNode:
        self.expect(TokenType.CLEANUP)
        expr = self.parse_atom()
        store_as = None
        if self.at(TokenType.ARROW_RIGHT):
            self.advance()
            store_as = self.expect(TokenType.IDENT).value
        return CleanupNode(expr=expr, store_as=store_as)

    def parse_print(self) -> PrintNode:
        self.expect(TokenType.PRINT)
        if self.at(TokenType.STRING):
            tok = self.advance()
            return PrintNode(value=tok.value, is_string=True)
        else:
            expr = self.parse_atom()
            return PrintNode(value=expr, is_string=False)

    def parse_dim(self) -> DimNode:
        self.expect(TokenType.DIM)
        val = int(self.expect(TokenType.INTEGER).value)
        return DimNode(dimensions=val)

    def parse_seed(self) -> SeedNode:
        self.expect(TokenType.SEED)
        val = int(self.expect(TokenType.INTEGER).value)
        return SeedNode(seed=val)

    # ── Expression parsers ───────────────────────────────────────────

    def parse_superpose_expr(self) -> Expr:
        """Parse: expr + expr + expr ..."""
        first = self.parse_expr()
        if not self.at(TokenType.PLUS):
            return first
        operands = [first]
        while self.at(TokenType.PLUS):
            self.advance()
            operands.append(self.parse_expr())
        if isinstance(first, SuperposeExprNode):
            return first  # shouldn't happen but safety
        return SuperposeExprNode(operands=operands)

    def parse_expr(self) -> Expr:
        """Parse: atom * atom (bind chain)"""
        left = self.parse_atom()
        while self.at(TokenType.STAR):
            self.advance()
            right = self.parse_atom()
            left = BindExprNode(left=left, right=right)
        return left

    def parse_atom(self) -> Expr:
        """Parse: IDENT | ( expr )"""
        if self.at(TokenType.LPAREN):
            self.advance()
            expr = self.parse_superpose_expr()
            self.expect(TokenType.RPAREN)
            return expr
        elif self.at(TokenType.IDENT):
            return IdentNode(name=self.advance().value)
        else:
            raise ParseError(
                f"Expected identifier or '(', got {self.peek().type.name}",
                self.peek()
            )

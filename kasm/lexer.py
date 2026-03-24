"""
KASM Lexer — Tokenizes .kasm source code into a stream of typed tokens.

Token types map directly to KASM grammar v0.1:
    Keywords:  NODE, BIND, SUPERPOSE, PERMUTE, RESONATE, UNBIND, CLEANUP, PRINT, DIM, SEED
    Operators: *, +, <=>, >>, <<, =, ->
    Literals:  STRING ("..."), INTEGER (digits)
    Names:     IDENT (variable/concept names)
    Structure: COMMA, LPAREN, RPAREN, NEWLINE, EOF
"""

from enum import Enum, auto
from typing import List, NamedTuple


class TokenType(Enum):
    # Keywords
    NODE = auto()
    BIND = auto()
    SUPERPOSE = auto()
    PERMUTE = auto()
    RESONATE = auto()
    UNBIND = auto()
    CLEANUP = auto()
    PRINT = auto()
    DIM = auto()
    SEED = auto()

    # Operators
    STAR = auto()       # *  (bind)
    PLUS = auto()       # +  (superpose)
    ARROW_DOUBLE = auto()  # <=>  (resonate)
    SHIFT_RIGHT = auto()   # >>   (permute right)
    SHIFT_LEFT = auto()    # <<   (permute left)
    EQUALS = auto()     # =  (assignment)
    ARROW_RIGHT = auto()   # ->  (store result)

    # Literals
    STRING = auto()     # "..."
    INTEGER = auto()    # 42

    # Names
    IDENT = auto()      # variable names

    # Structure
    COMMA = auto()
    LPAREN = auto()
    RPAREN = auto()
    NEWLINE = auto()
    EOF = auto()


KEYWORDS = {
    'NODE': TokenType.NODE,
    'BIND': TokenType.BIND,
    'SUPERPOSE': TokenType.SUPERPOSE,
    'PERMUTE': TokenType.PERMUTE,
    'RESONATE': TokenType.RESONATE,
    'UNBIND': TokenType.UNBIND,
    'CLEANUP': TokenType.CLEANUP,
    'PRINT': TokenType.PRINT,
    'DIM': TokenType.DIM,
    'SEED': TokenType.SEED,
}


class Token(NamedTuple):
    type: TokenType
    value: str
    line: int
    col: int


class LexerError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"Lexer error at line {line}, col {col}: {message}")
        self.line = line
        self.col = col


def tokenize(source: str) -> List[Token]:
    """
    Tokenize KASM source code into a list of tokens.

    Handles:
        - Single-line comments (//)
        - Multi-line comments (/* ... */)
        - All KASM operators including multi-char (<==>, >>, <<, ->)
        - String literals with basic escape sequences
        - Integer literals
        - Identifiers and keywords
        - Significant newlines (statement separators)
    """
    tokens: List[Token] = []
    i = 0
    line = 1
    col = 1
    length = len(source)

    def peek(offset=0) -> str:
        pos = i + offset
        return source[pos] if pos < length else '\0'

    def advance(n=1) -> str:
        nonlocal i, col
        ch = source[i:i+n]
        i += n
        col += n
        return ch

    while i < length:
        ch = source[i]

        # ── Newlines ──
        if ch == '\n':
            # Only emit NEWLINE if last token isn't already a NEWLINE
            if tokens and tokens[-1].type != TokenType.NEWLINE:
                tokens.append(Token(TokenType.NEWLINE, '\\n', line, col))
            i += 1
            line += 1
            col = 1
            continue

        # ── Whitespace (non-newline) ──
        if ch in ' \t\r':
            i += 1
            col += 1
            continue

        # ── Single-line comment ──
        if ch == '/' and peek(1) == '/':
            while i < length and source[i] != '\n':
                i += 1
                col += 1
            continue

        # ── Multi-line comment ──
        if ch == '/' and peek(1) == '*':
            start_line, start_col = line, col
            i += 2
            col += 2
            while i < length:
                if source[i] == '*' and peek(1) == '/':
                    i += 2
                    col += 2
                    break
                if source[i] == '\n':
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1
            else:
                raise LexerError("Unterminated multi-line comment", start_line, start_col)
            continue

        # ── Multi-character operators ──
        if ch == '<' and peek(1) == '=' and peek(2) == '>':
            tokens.append(Token(TokenType.ARROW_DOUBLE, '<=>', line, col))
            advance(3)
            continue

        if ch == '>' and peek(1) == '>':
            tokens.append(Token(TokenType.SHIFT_RIGHT, '>>', line, col))
            advance(2)
            continue

        if ch == '<' and peek(1) == '<':
            tokens.append(Token(TokenType.SHIFT_LEFT, '<<', line, col))
            advance(2)
            continue

        if ch == '-' and peek(1) == '>':
            tokens.append(Token(TokenType.ARROW_RIGHT, '->', line, col))
            advance(2)
            continue

        # ── Single-character operators ──
        if ch == '*':
            tokens.append(Token(TokenType.STAR, '*', line, col))
            advance()
            continue
        if ch == '+':
            tokens.append(Token(TokenType.PLUS, '+', line, col))
            advance()
            continue
        if ch == '=':
            tokens.append(Token(TokenType.EQUALS, '=', line, col))
            advance()
            continue
        if ch == ',':
            tokens.append(Token(TokenType.COMMA, ',', line, col))
            advance()
            continue
        if ch == '(':
            tokens.append(Token(TokenType.LPAREN, '(', line, col))
            advance()
            continue
        if ch == ')':
            tokens.append(Token(TokenType.RPAREN, ')', line, col))
            advance()
            continue

        # ── String literals ──
        if ch == '"':
            start_col = col
            advance()  # skip opening quote
            string_val = []
            while i < length and source[i] != '"':
                if source[i] == '\\' and i + 1 < length:
                    next_ch = source[i + 1]
                    if next_ch == 'n':
                        string_val.append('\n')
                    elif next_ch == 't':
                        string_val.append('\t')
                    elif next_ch == '"':
                        string_val.append('"')
                    elif next_ch == '\\':
                        string_val.append('\\')
                    else:
                        string_val.append(next_ch)
                    i += 2
                    col += 2
                elif source[i] == '\n':
                    raise LexerError("Unterminated string (newline in string)", line, start_col)
                else:
                    string_val.append(source[i])
                    i += 1
                    col += 1

            if i >= length:
                raise LexerError("Unterminated string", line, start_col)
            advance()  # skip closing quote
            tokens.append(Token(TokenType.STRING, ''.join(string_val), line, start_col))
            continue

        # ── Integer literals ──
        if ch.isdigit():
            start_col = col
            num = []
            while i < length and source[i].isdigit():
                num.append(source[i])
                i += 1
                col += 1
            tokens.append(Token(TokenType.INTEGER, ''.join(num), line, start_col))
            continue

        # ── Identifiers and keywords ──
        if ch.isalpha() or ch == '_':
            start_col = col
            ident = []
            while i < length and (source[i].isalnum() or source[i] == '_'):
                ident.append(source[i])
                i += 1
                col += 1
            word = ''.join(ident)
            tok_type = KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tok_type, word, line, start_col))
            continue

        raise LexerError(f"Unexpected character: '{ch}'", line, col)

    # Ensure stream ends with NEWLINE + EOF
    if tokens and tokens[-1].type != TokenType.NEWLINE:
        tokens.append(Token(TokenType.NEWLINE, '\\n', line, col))
    tokens.append(Token(TokenType.EOF, '', line, col))

    return tokens

# KASM Language Specification v0.1

## Overview
KASM (Knowledge Assembly) is a domain-specific language for Vector Symbolic Architectures.
It is to topological algebra what SQL is to relational algebra.

## File Extension
`.kasm`

## Comments
```
// Single-line comment
/* Multi-line comment */
```

## Data Types
- **Vector**: 10,000-D bipolar vector {-1, +1}^D (the only type)
- **Scalar**: Cosine similarity result (float, output only)

## Statements

### NODE — Create atomic concept
```
NODE name
NODE name1, name2, name3    // batch creation
```

### BIND — Role-filler association (element-wise multiply)
```
BIND result = expr * expr
```

### SUPERPOSE — Bundle into composite (element-wise add + threshold)
```
SUPERPOSE result = expr + expr + expr
```

### PERMUTE — Sequence encoding (circular shift)
```
PERMUTE result = expr >> n     // shift right by n
PERMUTE result = expr << n     // shift left by n
```

### RESONATE — Cosine similarity query
```
RESONATE expr <=> expr          // prints similarity score
RESONATE expr <=> expr -> name  // stores result and prints
```

### UNBIND — Inverse query (same as BIND, semantic alias)
```
UNBIND result = composite * key
```

### CLEANUP — Find nearest known symbol
```
CLEANUP expr                    // prints top matches
CLEANUP expr -> name            // stores best match vector
```

### PRINT — Output
```
PRINT "literal string"
PRINT name                     // prints vector stats
```

### DIM — Set dimensions (must be first statement if used)
```
DIM 10000
```

### SEED — Set random seed for reproducibility
```
SEED 42
```

## Expressions
Expressions can reference named vectors or inline operations:
```
BIND x = a * b
SUPERPOSE s = x + (c * d) + e    // inline bind inside superpose
```

## Operator Precedence
1. `*` (BIND) — highest
2. `+` (SUPERPOSE)
3. `>>`, `<<` (PERMUTE)
4. `<=>` (RESONATE) — lowest

## Example Program
```
// Analogical Reasoning: Solar System <=> Atom
SEED 42
DIM 10000

NODE sun, planet, gravity
NODE nucleus, electron, electromagnetism
NODE role_center, role_orbiter, role_force

BIND r_sun    = sun * role_center
BIND r_planet = planet * role_orbiter
BIND r_grav   = gravity * role_force

BIND r_nuc  = nucleus * role_center
BIND r_elec = electron * role_orbiter
BIND r_em   = electromagnetism * role_force

SUPERPOSE solar_system = r_sun + r_planet + r_grav
SUPERPOSE atom = r_nuc + r_elec + r_em

PRINT "=== Direct Comparison ==="
RESONATE solar_system <=> atom

PRINT "=== Analogical Mapping ==="
BIND mapping = solar_system * atom
UNBIND answer = mapping * sun
CLEANUP answer

PRINT "=== Reverse ==="
UNBIND reverse = mapping * nucleus
CLEANUP reverse
```

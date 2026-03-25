//! KOS V8.0 — Rust Core Engine (4-Stage Retrieval Pipeline)
//!
//! Arena-based spreading activation graph + SIMD-ready hyperdimensional VSA.
//! Compiled to Python via PyO3.
//!
//! V8 changes:
//!   - Typed edges: (target_idx, weight, myelin, edge_type)
//!   - Hub penalty: 1/(1+ln(degree)) on propagation
//!   - Beam search: bounded frontier (width=32, depth=5)
//!   - Tier tracking: last_queried timestamp per node
//!   - Format v2: backward-compatible binary persistence

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::io::{BufWriter, BufReader, Write, Read};
use std::fs::File;

// ── Edge Type Constants (match Python edge_types.py) ────────────
const ET_GENERIC: u8 = 0;
const ET_IS_A: u8 = 1;
const ET_CAUSES: u8 = 2;
const ET_PART_OF: u8 = 3;
const ET_OBSERVED_WITH: u8 = 4;
const ET_CONTRADICTS: u8 = 5;
const ET_SUPPORTS: u8 = 6;
const ET_DERIVED_FROM: u8 = 7;
const ET_PROCEDURE_STEP: u8 = 8;
const ET_TEMPORAL_BEFORE: u8 = 9;
const ET_TEMPORAL_AFTER: u8 = 10;
const ET_LOCATED_IN: u8 = 11;
const ET_HAS_PROPERTY: u8 = 12;

/// Trust multiplier per edge type (used in beam scoring)
fn edge_trust(et: u8) -> f64 {
    match et {
        ET_IS_A => 0.9,
        ET_CAUSES => 0.85,
        ET_PART_OF => 0.8,
        ET_SUPPORTS => 0.8,
        ET_LOCATED_IN => 0.8,
        ET_DERIVED_FROM => 0.7,
        ET_HAS_PROPERTY => 0.7,
        ET_PROCEDURE_STEP => 0.9,
        ET_TEMPORAL_BEFORE | ET_TEMPORAL_AFTER => 0.6,
        ET_GENERIC => 0.5,
        ET_OBSERVED_WITH => 0.4,
        ET_CONTRADICTS => 0.3,
        _ => 0.5,
    }
}

// ── Arena Node ──────────────────────────────────────────────────────

struct ArenaNode {
    id: String,
    activation: f64,
    fuel: f64,
    last_tick: u64,
    last_queried: u64,  // V8: tier tracking
    temporal_decay: f64,
    max_energy: f64,
    connections: Vec<(usize, f32, u32, u8)>, // V8: (target_idx, weight, myelin, edge_type)
    vsa_base: Vec<i8>,
    vsa_state: Vec<i8>,
    binding_count: u32,
}

impl ArenaNode {
    fn new(id: String, td: f64, me: f64, dim: usize, rng: &mut ChaCha8Rng) -> Self {
        let base: Vec<i8> = (0..dim)
            .map(|_| if rng.random_bool(0.5) { 1i8 } else { -1i8 })
            .collect();
        ArenaNode {
            id,
            activation: 0.0,
            fuel: 0.0,
            last_tick: 0,
            last_queried: 0,
            temporal_decay: td,
            max_energy: me,
            connections: Vec::new(),
            vsa_state: base.clone(),
            vsa_base: base,
            binding_count: 0,
        }
    }

    fn decay(&mut self, tick: u64) {
        if tick > self.last_tick {
            let f = self.temporal_decay.powi((tick - self.last_tick) as i32);
            self.activation *= f;
            self.fuel *= f;
            self.last_tick = tick;
        }
    }

    fn receive(&mut self, energy: f64, tick: u64) {
        self.decay(tick);
        self.activation = (self.activation + energy).clamp(-self.max_energy, self.max_energy);
        if energy > 0.0 && self.activation > 0.0 {
            self.fuel = (self.fuel + energy).clamp(0.0, self.max_energy);
        }
    }

    /// Hub penalty: 1/(1+ln(degree)). High-degree nodes propagate less energy.
    fn hub_penalty(&self) -> f64 {
        let d = self.connections.len() as f64;
        if d <= 1.0 { 1.0 } else { 1.0 / (1.0 + d.ln()) }
    }

    /// Tier bias based on recency of last query involvement.
    fn tier_bias(&self, current_tick: u64) -> f64 {
        if self.last_queried == 0 { return 0.5; } // never queried = cold
        let age = current_tick.saturating_sub(self.last_queried);
        if age < 50 { 1.5 }       // hot
        else if age < 200 { 1.0 }  // warm
        else { 0.5 }               // cold
    }
}

// ── Priority Queue ──────────────────────────────────────────────────

#[derive(PartialEq)]
struct PQ {
    fuel: f64,
    tie: u64,
    idx: usize,
}
impl Eq for PQ {}
impl PartialOrd for PQ {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for PQ {
    fn cmp(&self, o: &Self) -> Ordering {
        self.fuel.partial_cmp(&o.fuel).unwrap_or(Ordering::Equal)
            .then(self.tie.cmp(&o.tie))
    }
}

// ── VSA Ops ─────────────────────────────────────────────────────────

#[inline]
fn bind(a: &[i8], b: &[i8]) -> Vec<i8> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

fn superpose(vecs: &[&[i8]], rng: &mut ChaCha8Rng) -> Vec<i8> {
    let d = vecs[0].len();
    let mut r = vec![0i32; d];
    for v in vecs { for i in 0..d { r[i] += v[i] as i32; } }
    r.iter().map(|&s| if s > 0 { 1 } else if s < 0 { -1 } else {
        if rng.random_bool(0.5) { 1 } else { -1 }
    }).collect()
}

#[inline]
fn resonate(a: &[i8], b: &[i8]) -> f64 {
    let dot: i64 = a.iter().zip(b).map(|(x, y)| (*x as i64) * (*y as i64)).sum();
    dot as f64 / a.len() as f64
}

// ── Binary format helpers ───────────────────────────────────────────

const MAGIC: u32 = 0x4B4F5338; // "KOS8"
const FORMAT_VERSION: u32 = 2;

fn write_u8(w: &mut impl Write, v: u8) -> std::io::Result<()> { w.write_all(&[v]) }
fn write_u16(w: &mut impl Write, v: u16) -> std::io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn write_f32(w: &mut impl Write, v: f32) -> std::io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn write_f64(w: &mut impl Write, v: f64) -> std::io::Result<()> { w.write_all(&v.to_le_bytes()) }

fn read_u8(r: &mut impl Read) -> std::io::Result<u8> { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0]) }
fn read_u16(r: &mut impl Read) -> std::io::Result<u16> { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
fn read_u32(r: &mut impl Read) -> std::io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn read_u64(r: &mut impl Read) -> std::io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn read_f32(r: &mut impl Read) -> std::io::Result<f32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b)) }
fn read_f64(r: &mut impl Read) -> std::io::Result<f64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(f64::from_le_bytes(b)) }

// ── RustKernel V8 ───────────────────────────────────────────────────

#[pyclass]
struct RustKernel {
    arena: Vec<ArenaNode>,
    idx: HashMap<String, usize>,
    prov: HashMap<(usize, usize), Vec<String>>,
    tick: u64,
    max_ticks: u64,
    tie: u64,
    dim: usize,
    td: f64,
    me: f64,
    rng: ChaCha8Rng,
    // V8: retrieval cache (seed_hash → (tick, results))
    cache: HashMap<u64, (u64, Vec<(String, f64)>)>,
    cache_ttl: u64,
}

#[pymethods]
impl RustKernel {
    #[new]
    #[pyo3(signature = (dim=10000, temporal_decay=0.7, max_energy=3.0, seed=42))]
    fn new(dim: usize, temporal_decay: f64, max_energy: f64, seed: u64) -> Self {
        RustKernel {
            arena: Vec::with_capacity(10_000),
            idx: HashMap::new(),
            prov: HashMap::new(),
            tick: 0, max_ticks: 15, tie: 0,
            dim, td: temporal_decay, me: max_energy,
            rng: ChaCha8Rng::seed_from_u64(seed),
            cache: HashMap::new(),
            cache_ttl: 10,
        }
    }

    fn add_node(&mut self, name: String) -> usize {
        if let Some(&i) = self.idx.get(&name) { return i; }
        let i = self.arena.len();
        self.arena.push(ArenaNode::new(name.clone(), self.td, self.me, self.dim, &mut self.rng));
        self.idx.insert(name, i);
        self.invalidate_cache();
        i
    }

    /// V8: add_connection with edge_type parameter
    #[pyo3(signature = (src, tgt, w, text=None, edge_type=0))]
    fn add_connection(&mut self, src: String, tgt: String, w: f32,
                      text: Option<String>, edge_type: u8) {
        let si = self.add_node(src);
        let ti = self.add_node(tgt);
        let exists = self.arena[si].connections.iter_mut().find(|(t,_,_,_)| *t == ti);
        if let Some(e) = exists { e.1 = w; e.3 = edge_type; } else {
            self.arena[si].connections.push((ti, w, 0, edge_type));
        }
        if let Some(t) = text {
            let k = if si <= ti { (si, ti) } else { (ti, si) };
            self.prov.entry(k).or_default().push(t);
        }
        // VSA bind
        let b = bind(&self.arena[si].vsa_base, &self.arena[ti].vsa_base);
        let old = self.arena[si].vsa_state.clone();
        let s = superpose(&[old.as_slice(), b.as_slice()], &mut self.rng);
        self.arena[si].vsa_state = s;
        self.arena[si].binding_count += 1;
        self.invalidate_cache();
    }

    // ═══════════════════════════════════════════════════════════════
    // RETRIEVAL: Legacy query (backward compatible, now with hub penalty)
    // ═══════════════════════════════════════════════════════════════

    fn query(&mut self, seeds: Vec<String>, top_k: usize) -> Vec<(String, f64)> {
        // Check cache
        let cache_key = self.hash_seeds(&seeds);
        if let Some((cached_tick, cached_results)) = self.cache.get(&cache_key) {
            if self.tick.saturating_sub(*cached_tick) < self.cache_ttl {
                return cached_results.clone();
            }
        }

        self.tick += 1;
        let t = self.tick;
        let mut active: HashSet<usize> = HashSet::new();
        let mut pq: BinaryHeap<PQ> = BinaryHeap::new();

        for n in &seeds {
            if let Some(&i) = self.idx.get(n) {
                self.arena[i].receive(3.0, t);
                self.arena[i].last_queried = t;
                self.tie += 1;
                pq.push(PQ { fuel: self.arena[i].fuel, tie: self.tie, idx: i });
                active.insert(i);
            }
        }

        let mut ticks = 0u64;
        while let Some(e) = pq.pop() {
            if ticks >= self.max_ticks { break; }
            let ni = e.idx;
            if self.arena[ni].fuel < 0.05 { continue; }
            ticks += 1;

            self.arena[ni].decay(t);
            let fuel = self.arena[ni].fuel;
            let hub_pen = self.arena[ni].hub_penalty(); // V8: hub penalty

            let mut edges: Vec<(usize, f64, u8)> = self.arena[ni].connections.iter()
                .map(|(tgt, w, m, et)| (*tgt, (*w as f64) * (1.0 + (*m as f64) * 0.01), *et))
                .collect();
            edges.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
            edges.truncate(500);

            let out: Vec<(usize, f64)> = edges.iter()
                .filter_map(|(tgt, ew, et)| {
                    let trust = edge_trust(*et);
                    let tier = self.arena[*tgt].tier_bias(t);
                    let p = fuel * ew * 0.8 * hub_pen * trust * tier;
                    if p.abs() >= 0.05 { Some((*tgt, p)) } else { None }
                }).collect();

            // Myelin reinforcement
            for (tgt, _) in &out {
                if let Some(e) = self.arena[ni].connections.iter_mut().find(|(t,_,_,_)| t == tgt) {
                    e.2 += 1;
                }
            }
            self.arena[ni].fuel = 0.0;

            for (ti, pe) in out {
                self.arena[ti].receive(pe, t);
                self.arena[ti].last_queried = t;
                active.insert(ti);
                if self.arena[ti].fuel >= 0.05 {
                    self.tie += 1;
                    pq.push(PQ { fuel: self.arena[ti].fuel, tie: self.tie, idx: ti });
                }
            }
        }

        let seed_idx: HashSet<usize> = seeds.iter().filter_map(|n| self.idx.get(n).copied()).collect();
        let mut res: Vec<(String, f64)> = Vec::new();
        for &i in &active {
            self.arena[i].decay(t);
            let a = self.arena[i].activation;
            if a > 0.1 && !seed_idx.contains(&i) {
                res.push((self.arena[i].id.clone(), a));
            }
            self.arena[i].fuel = 0.0;
            self.arena[i].activation = 0.0;
        }
        res.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        res.truncate(top_k);

        // Store in cache
        self.cache.insert(cache_key, (self.tick, res.clone()));
        res
    }

    // ═══════════════════════════════════════════════════════════════
    // V8: BEAM SEARCH — Bounded frontier retrieval
    // ═══════════════════════════════════════════════════════════════

    /// Beam search: expand top beam_width nodes at each depth level.
    /// Returns (node_name, score) for the top-k results.
    #[pyo3(signature = (seeds, top_k=20, beam_width=32, max_depth=5, allowed_edge_types=None))]
    fn query_beam(&mut self, seeds: Vec<String>, top_k: usize,
                  beam_width: usize, max_depth: usize,
                  allowed_edge_types: Option<Vec<u8>>) -> Vec<(String, f64)> {
        self.tick += 1;
        let t = self.tick;
        let allowed: Option<HashSet<u8>> = allowed_edge_types.map(|v| v.into_iter().collect());

        // Stage A: Candidate seeds
        let mut beam: Vec<(usize, f64)> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        for n in &seeds {
            if let Some(&i) = self.idx.get(n) {
                beam.push((i, 3.0));
                visited.insert(i);
                self.arena[i].last_queried = t;
            }
        }

        // Stage B: Bounded expansion (depth levels)
        let mut all_activated: Vec<(usize, f64)> = beam.clone();

        for _depth in 0..max_depth {
            let mut next_beam: Vec<(usize, f64)> = Vec::new();

            for &(ni, energy) in &beam {
                if energy.abs() < 0.05 { continue; }

                let hub_pen = self.arena[ni].hub_penalty();

                // Collect connections to avoid borrow conflict
                let conns: Vec<(usize, f32, u32, u8)> = self.arena[ni].connections.clone();

                for &(tgt, w, m, et) in &conns {
                    // Filter by allowed edge types
                    if let Some(ref allowed_set) = allowed {
                        if !allowed_set.contains(&et) { continue; }
                    }

                    let trust = edge_trust(et);
                    let tier = self.arena[tgt].tier_bias(t);
                    let myelin_boost = 1.0 + (m as f64) * 0.01;
                    let score = energy * (w as f64) * myelin_boost * 0.8
                                * hub_pen * trust * tier;

                    if score.abs() >= 0.05 {
                        next_beam.push((tgt, score));
                        self.arena[tgt].last_queried = t;

                        // Myelin reinforcement
                        if let Some(e) = self.arena[ni].connections.iter_mut()
                            .find(|(t,_,_,_)| *t == tgt) {
                            e.2 += 1;
                        }
                    }
                }
            }

            // Keep top beam_width candidates (beam pruning)
            next_beam.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
            next_beam.truncate(beam_width);

            // Deduplicate: merge scores for already-visited nodes
            let mut deduped: Vec<(usize, f64)> = Vec::new();
            for (idx, score) in next_beam {
                if !visited.contains(&idx) {
                    visited.insert(idx);
                    deduped.push((idx, score));
                    all_activated.push((idx, score));
                }
            }

            if deduped.is_empty() { break; }
            beam = deduped;
        }

        // Stage C: Score and rank
        let seed_idx: HashSet<usize> = seeds.iter()
            .filter_map(|n| self.idx.get(n).copied()).collect();

        let mut res: Vec<(String, f64)> = all_activated.iter()
            .filter(|(i, _)| !seed_idx.contains(i))
            .map(|(i, score)| (self.arena[*i].id.clone(), *score))
            .collect();

        // Merge duplicates (sum scores)
        let mut merged: HashMap<String, f64> = HashMap::new();
        for (name, score) in &res {
            *merged.entry(name.clone()).or_insert(0.0) += score;
        }
        res = merged.into_iter().collect();
        res.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
        res.truncate(top_k);
        res
    }

    // ═══════════════════════════════════════════════════════════════
    // V8: CAUSAL LANE — Only traverse CAUSES edges
    // ═══════════════════════════════════════════════════════════════

    #[pyo3(signature = (seeds, top_k=10, max_depth=7))]
    fn query_causal(&mut self, seeds: Vec<String>, top_k: usize,
                    max_depth: usize) -> Vec<(String, f64)> {
        self.query_beam(seeds, top_k, 16, max_depth,
                       Some(vec![ET_CAUSES, ET_TEMPORAL_BEFORE, ET_TEMPORAL_AFTER]))
    }

    // ═══════════════════════════════════════════════════════════════
    // V8: ENERGY INJECTION + CONTINUOUS TICK (for organism loop)
    // ═══════════════════════════════════════════════════════════════

    /// Inject energy into a specific node without running full propagation.
    fn inject_energy(&mut self, node: String, amount: f64) -> bool {
        if let Some(&i) = self.idx.get(&node) {
            self.arena[i].receive(amount, self.tick);
            true
        } else {
            false
        }
    }

    /// Continuous tick: propagate one step, don't reset. Returns triggered action nodes.
    /// Action nodes are detected by name prefix "action_".
    fn tick_continuous(&mut self, propagation_factor: f64, threshold: f64) -> Vec<(String, f64)> {
        self.tick += 1;
        let t = self.tick;
        let mut triggered: Vec<(String, f64)> = Vec::new();

        // Collect nodes with fuel > threshold
        let active_indices: Vec<usize> = (0..self.arena.len())
            .filter(|&i| {
                self.arena[i].decay(t);
                self.arena[i].fuel > threshold
            })
            .collect();

        for ni in active_indices {
            let fuel = self.arena[ni].fuel;
            let hub_pen = self.arena[ni].hub_penalty();

            // Collect edges to propagate
            let edges: Vec<(usize, f64)> = self.arena[ni].connections.iter()
                .map(|(tgt, w, m, et)| {
                    let trust = edge_trust(*et);
                    let myelin_boost = 1.0 + (*m as f64) * 0.01;
                    (*tgt, fuel * (*w as f64) * myelin_boost * propagation_factor * hub_pen * trust)
                })
                .filter(|(_, e)| e.abs() >= threshold)
                .collect();

            self.arena[ni].fuel *= 0.5; // Partial drain, not full reset

            for (ti, energy) in edges {
                self.arena[ti].receive(energy, t);

                // Check if this is an action node that breached threshold
                if self.arena[ti].activation > 2.0 && self.arena[ti].id.starts_with("action_") {
                    triggered.push((self.arena[ti].id.clone(), self.arena[ti].activation));
                    self.arena[ti].activation = 0.0; // Reset after firing
                    self.arena[ti].fuel = 0.0;
                }
            }
        }

        triggered
    }

    /// Get count of active nodes (fuel > 0.01). Used for adaptive tick rate.
    fn active_node_count(&self) -> usize {
        self.arena.iter().filter(|n| n.fuel > 0.01).count()
    }

    // ═══════════════════════════════════════════════════════════════
    // PROVENANCE + EDGE INFO
    // ═══════════════════════════════════════════════════════════════

    fn get_provenance(&self, a: String, b: String) -> Vec<String> {
        let (ia, ib) = match (self.idx.get(&a), self.idx.get(&b)) {
            (Some(&x), Some(&y)) => if x <= y { (x, y) } else { (y, x) },
            _ => return vec![],
        };
        self.prov.get(&(ia, ib)).cloned().unwrap_or_default()
    }

    /// V8: Get edge info including type: returns (weight, myelin, edge_type) or None.
    fn get_edge(&self, src: &str, tgt: &str) -> Option<(f32, u32, u8)> {
        let si = self.idx.get(src)?;
        let ti = self.idx.get(tgt)?;
        self.arena[*si].connections.iter()
            .find(|(t,_,_,_)| t == ti)
            .map(|&(_, w, m, et)| (w, m, et))
    }

    /// V8: Get all neighbors with edge types: [(target_name, weight, myelin, edge_type), ...]
    fn get_neighbors(&self, node: &str) -> PyResult<Vec<(String, f32, u32, u8)>> {
        let ni = *self.idx.get(node).ok_or_else(|| PyValueError::new_err(format!("Unknown: {node}")))?;
        Ok(self.arena[ni].connections.iter()
            .map(|&(tgt, w, m, et)| (self.arena[tgt].id.clone(), w, m, et))
            .collect())
    }

    /// Get node degree (number of connections).
    fn get_degree(&self, node: &str) -> Option<usize> {
        self.idx.get(node).map(|&i| self.arena[i].connections.len())
    }

    // ═══════════════════════════════════════════════════════════════
    // VSA OPERATIONS
    // ═══════════════════════════════════════════════════════════════

    fn resonate(&self, a: &str, b: &str) -> PyResult<f64> {
        let ia = self.idx.get(a).ok_or_else(|| PyValueError::new_err(format!("Unknown: {a}")))?;
        let ib = self.idx.get(b).ok_or_else(|| PyValueError::new_err(format!("Unknown: {b}")))?;
        Ok(resonate(&self.arena[*ia].vsa_state, &self.arena[*ib].vsa_state))
    }

    fn resonate_search(&self, q: &str, top_k: usize, threshold: f64) -> PyResult<Vec<(String, f64)>> {
        let qi = *self.idx.get(q).ok_or_else(|| PyValueError::new_err(format!("Unknown: {q}")))?;
        let qv = &self.arena[qi].vsa_state;
        let mut r: Vec<(String, f64)> = self.arena.iter().enumerate()
            .filter(|(i, _)| *i != qi)
            .filter_map(|(_, n)| {
                let s = resonate(qv, &n.vsa_state);
                if s.abs() >= threshold { Some((n.id.clone(), s)) } else { None }
            }).collect();
        r.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
        r.truncate(top_k);
        Ok(r)
    }

    // ═══════════════════════════════════════════════════════════════
    // BATCH OPERATIONS
    // ═══════════════════════════════════════════════════════════════

    /// Batch add edges with types: [(src, tgt, weight, edge_type), ...]
    fn batch_add_edges(&mut self, edges: Vec<(String, String, f32)>) {
        for (src, tgt, w) in edges {
            let si = self.add_node(src);
            let ti = self.add_node(tgt);
            let exists = self.arena[si].connections.iter_mut().find(|(t,_,_,_)| *t == ti);
            if let Some(e) = exists { e.1 = w; } else {
                self.arena[si].connections.push((ti, w, 0, ET_GENERIC));
            }
        }
        self.invalidate_cache();
    }

    /// Batch add typed edges: [(src, tgt, weight, edge_type), ...]
    fn batch_add_typed_edges(&mut self, edges: Vec<(String, String, f32, u8)>) {
        for (src, tgt, w, et) in edges {
            let si = self.add_node(src);
            let ti = self.add_node(tgt);
            let exists = self.arena[si].connections.iter_mut().find(|(t,_,_,_)| *t == ti);
            if let Some(e) = exists { e.1 = w; e.3 = et; } else {
                self.arena[si].connections.push((ti, w, 0, et));
            }
        }
        self.invalidate_cache();
    }

    // ═══════════════════════════════════════════════════════════════
    // LEARNING PRIMITIVES
    // ═══════════════════════════════════════════════════════════════

    fn myelinate(&mut self, src: &str, tgt: &str, delta: i32) -> PyResult<bool> {
        let si = *self.idx.get(src).ok_or_else(|| PyValueError::new_err(format!("Unknown: {src}")))?;
        let ti = *self.idx.get(tgt).ok_or_else(|| PyValueError::new_err(format!("Unknown: {tgt}")))?;
        if let Some(e) = self.arena[si].connections.iter_mut().find(|(t,_,_,_)| *t == ti) {
            let new_val = (e.2 as i64 + delta as i64).clamp(0, u32::MAX as i64) as u32;
            e.2 = new_val;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn adjust_weight(&mut self, src: &str, tgt: &str, delta: f32) -> PyResult<bool> {
        let si = *self.idx.get(src).ok_or_else(|| PyValueError::new_err(format!("Unknown: {src}")))?;
        let ti = *self.idx.get(tgt).ok_or_else(|| PyValueError::new_err(format!("Unknown: {tgt}")))?;
        if let Some(e) = self.arena[si].connections.iter_mut().find(|(t,_,_,_)| *t == ti) {
            e.1 = (e.1 + delta).clamp(-1.0, 1.0);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn prune_weak_edges(&mut self, min_weight: f32) -> usize {
        let mut removed = 0;
        for node in &mut self.arena {
            let before = node.connections.len();
            node.connections.retain(|&(_, w, _, _)| w.abs() >= min_weight);
            removed += before - node.connections.len();
        }
        if removed > 0 { self.invalidate_cache(); }
        removed
    }

    fn prune_unmyelinated(&mut self, max_edges_per_node: usize) -> usize {
        let mut removed = 0;
        for node in &mut self.arena {
            if node.connections.len() > max_edges_per_node {
                node.connections.sort_by(|a, b| a.2.cmp(&b.2)
                    .then(a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(Ordering::Equal)));
                let excess = node.connections.len() - max_edges_per_node;
                node.connections.drain(0..excess);
                removed += excess;
            }
        }
        if removed > 0 { self.invalidate_cache(); }
        removed
    }

    fn decay_myelin(&mut self, factor: f64) {
        for node in &mut self.arena {
            for edge in &mut node.connections {
                edge.2 = ((edge.2 as f64) * factor) as u32;
            }
        }
    }

    fn top_myelinated(&self, n: usize) -> Vec<(String, String, f32, u32)> {
        let mut all: Vec<(String, String, f32, u32)> = Vec::new();
        for node in &self.arena {
            for &(tgt, w, m, _) in &node.connections {
                if m > 0 {
                    all.push((node.id.clone(), self.arena[tgt].id.clone(), w, m));
                }
            }
        }
        all.sort_by(|a, b| b.3.cmp(&a.3));
        all.truncate(n);
        all
    }

    // ═══════════════════════════════════════════════════════════════
    // PERSISTENCE: Binary save/load (V2 format with edge types)
    // ═══════════════════════════════════════════════════════════════

    fn save(&self, path: String) -> PyResult<usize> {
        let f = File::create(&path).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        let mut w = BufWriter::new(f);
        let mut bytes = 0usize;

        // Header
        write_u32(&mut w, MAGIC).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        write_u32(&mut w, FORMAT_VERSION).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        write_u32(&mut w, self.dim as u32).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        write_u32(&mut w, self.arena.len() as u32).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        write_u64(&mut w, self.tick).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        write_f64(&mut w, self.td).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        write_f64(&mut w, self.me).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        bytes += 40;

        for node in &self.arena {
            let id_bytes = node.id.as_bytes();
            write_u16(&mut w, id_bytes.len() as u16).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            w.write_all(id_bytes).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            bytes += 2 + id_bytes.len();

            // VSA vectors
            let base_u8: Vec<u8> = node.vsa_base.iter().map(|&v| v as u8).collect();
            w.write_all(&base_u8).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            let state_u8: Vec<u8> = node.vsa_state.iter().map(|&v| v as u8).collect();
            w.write_all(&state_u8).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            bytes += self.dim * 2;

            write_u32(&mut w, node.binding_count).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            write_u64(&mut w, node.last_queried).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            bytes += 12;

            // V2: edges with type
            write_u32(&mut w, node.connections.len() as u32).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            bytes += 4;
            for &(tgt_idx, weight, myelin, edge_type) in &node.connections {
                write_u32(&mut w, tgt_idx as u32).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                write_f32(&mut w, weight).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                write_u32(&mut w, myelin).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                write_u8(&mut w, edge_type).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                bytes += 13; // 4+4+4+1
            }
        }

        // Provenance
        write_u32(&mut w, self.prov.len() as u32).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        bytes += 4;
        for (&(a, b), texts) in &self.prov {
            write_u32(&mut w, a as u32).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            write_u32(&mut w, b as u32).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            write_u16(&mut w, texts.len() as u16).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            bytes += 10;
            for t in texts {
                let tb = t.as_bytes();
                write_u32(&mut w, tb.len() as u32).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                w.write_all(tb).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                bytes += 4 + tb.len();
            }
        }

        w.flush().map_err(|e| PyIOError::new_err(format!("{e}")))?;
        Ok(bytes)
    }

    fn load(&mut self, path: String) -> PyResult<usize> {
        let f = File::open(&path).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        let mut r = BufReader::new(f);

        let magic = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        // Accept both V1 (KOS7) and V2 (KOS8) formats
        let is_v1 = magic == 0x4B4F5337;
        let is_v2 = magic == MAGIC;
        if !is_v1 && !is_v2 {
            return Err(PyValueError::new_err("Not a KOS brain file"));
        }
        let version = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        let dim = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
        let node_count = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
        let tick = read_u64(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        let td = read_f64(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?;
        let me = read_f64(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?;

        self.arena.clear();
        self.idx.clear();
        self.prov.clear();
        self.cache.clear();
        self.dim = dim;
        self.tick = tick;
        self.td = td;
        self.me = me;
        self.arena.reserve(node_count);

        for i in 0..node_count {
            let id_len = read_u16(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
            let mut id_buf = vec![0u8; id_len];
            r.read_exact(&mut id_buf).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            let id = String::from_utf8(id_buf).map_err(|e| PyValueError::new_err(format!("{e}")))?;

            let mut base_u8 = vec![0u8; dim];
            r.read_exact(&mut base_u8).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            let vsa_base: Vec<i8> = base_u8.iter().map(|&v| v as i8).collect();

            let mut state_u8 = vec![0u8; dim];
            r.read_exact(&mut state_u8).map_err(|e| PyIOError::new_err(format!("{e}")))?;
            let vsa_state: Vec<i8> = state_u8.iter().map(|&v| v as i8).collect();

            let binding_count = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?;

            // V2: read last_queried (V1: default to 0)
            let last_queried = if is_v2 && version >= 2 {
                read_u64(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?
            } else { 0 };

            let edge_count = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
            let mut connections = Vec::with_capacity(edge_count);
            for _ in 0..edge_count {
                let tgt = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
                let weight = read_f32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                let myelin = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                // V2: read edge type (V1: default to GENERIC)
                let edge_type = if is_v2 && version >= 2 {
                    read_u8(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))?
                } else { ET_GENERIC };
                connections.push((tgt, weight, myelin, edge_type));
            }

            self.idx.insert(id.clone(), i);
            self.arena.push(ArenaNode {
                id,
                activation: 0.0,
                fuel: 0.0,
                last_tick: 0,
                last_queried,
                temporal_decay: td,
                max_energy: me,
                connections,
                vsa_base,
                vsa_state,
                binding_count,
            });
        }

        // Provenance
        let prov_count = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
        for _ in 0..prov_count {
            let a = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
            let b = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
            let text_count = read_u16(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
            let mut texts = Vec::with_capacity(text_count);
            for _ in 0..text_count {
                let tlen = read_u32(&mut r).map_err(|e| PyIOError::new_err(format!("{e}")))? as usize;
                let mut tbuf = vec![0u8; tlen];
                r.read_exact(&mut tbuf).map_err(|e| PyIOError::new_err(format!("{e}")))?;
                texts.push(String::from_utf8(tbuf).map_err(|e| PyValueError::new_err(format!("{e}")))?);
            }
            self.prov.insert((a, b), texts);
        }

        Ok(node_count)
    }

    fn export_graph(&self) -> Vec<(String, Vec<(String, f32, u32, u8)>)> {
        self.arena.iter().map(|node| {
            let edges: Vec<(String, f32, u32, u8)> = node.connections.iter()
                .map(|&(tgt_idx, w, m, et)| (self.arena[tgt_idx].id.clone(), w, m, et))
                .collect();
            (node.id.clone(), edges)
        }).collect()
    }

    fn node_count(&self) -> usize { self.arena.len() }
    fn edge_count(&self) -> usize { self.arena.iter().map(|n| n.connections.len()).sum() }
    fn has_node(&self, name: &str) -> bool { self.idx.contains_key(name) }
    fn node_names(&self) -> Vec<String> { self.arena.iter().map(|n| n.id.clone()).collect() }

    fn stats(&self) -> HashMap<String, f64> {
        let n = self.arena.len();
        let e: usize = self.arena.iter().map(|n| n.connections.len()).sum();
        let total_myelin: u64 = self.arena.iter()
            .flat_map(|n| n.connections.iter().map(|e| e.2 as u64))
            .sum();
        let mut m = HashMap::new();
        m.insert("nodes".into(), n as f64);
        m.insert("edges".into(), e as f64);
        m.insert("vsa_dim".into(), self.dim as f64);
        m.insert("vsa_mb".into(), (n * 2 * self.dim) as f64 / 1048576.0);
        m.insert("arena_contiguous".into(), 1.0);
        m.insert("total_myelin".into(), total_myelin as f64);
        m.insert("tick".into(), self.tick as f64);
        m.insert("cache_size".into(), self.cache.len() as f64);
        m
    }
}

// Private helper methods (not exposed to Python)
impl RustKernel {
    fn hash_seeds(&self, seeds: &[String]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV-1a
        for s in seeds {
            for b in s.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= 0xff;
        }
        h
    }

    fn invalidate_cache(&mut self) {
        self.cache.clear();
    }
}

// ── Standalone RustVSA ──────────────────────────────────────────────

#[pyclass]
struct RustVSA {
    dim: usize,
    syms: HashMap<String, Vec<i8>>,
    rng: ChaCha8Rng,
}

#[pymethods]
impl RustVSA {
    #[new]
    #[pyo3(signature = (dim=10000, seed=42))]
    fn new(dim: usize, seed: u64) -> Self {
        RustVSA { dim, syms: HashMap::new(), rng: ChaCha8Rng::seed_from_u64(seed) }
    }

    fn node(&mut self, name: String) {
        let v: Vec<i8> = (0..self.dim).map(|_| if self.rng.random_bool(0.5) { 1 } else { -1 }).collect();
        self.syms.insert(name, v);
    }

    fn node_batch(&mut self, names: Vec<String>) {
        for name in names { self.node(name); }
    }

    fn bind_named(&mut self, result_name: String, a: &str, b: &str) -> PyResult<()> {
        let va = self.syms.get(a).ok_or_else(|| PyValueError::new_err(format!("Unknown: {a}")))?;
        let vb = self.syms.get(b).ok_or_else(|| PyValueError::new_err(format!("Unknown: {b}")))?;
        let r = bind(va, vb);
        self.syms.insert(result_name, r);
        Ok(())
    }

    fn superpose_named(&mut self, result_name: String, names: Vec<String>) -> PyResult<()> {
        let vecs: Vec<&[i8]> = names.iter()
            .map(|n| self.syms.get(n.as_str())
                .map(|v| v.as_slice())
                .ok_or_else(|| PyValueError::new_err(format!("Unknown: {n}"))))
            .collect::<PyResult<Vec<_>>>()?;
        let r = superpose(&vecs, &mut self.rng);
        self.syms.insert(result_name, r);
        Ok(())
    }

    fn resonate_named(&self, a: &str, b: &str) -> PyResult<f64> {
        let va = self.syms.get(a).ok_or_else(|| PyValueError::new_err(format!("Unknown: {a}")))?;
        let vb = self.syms.get(b).ok_or_else(|| PyValueError::new_err(format!("Unknown: {b}")))?;
        Ok(resonate(va, vb))
    }

    fn permute_named(&mut self, result_name: String, source: &str, shifts: i32) -> PyResult<()> {
        let v = self.syms.get(source).ok_or_else(|| PyValueError::new_err(format!("Unknown: {source}")))?;
        let n = v.len() as i32;
        let s = ((shifts % n) + n) % n;
        let mut r = vec![0i8; v.len()];
        for i in 0..v.len() { r[((i as i32 + s) % n) as usize] = v[i]; }
        self.syms.insert(result_name, r);
        Ok(())
    }

    fn cleanup_named(&self, query_name: &str, threshold: f64) -> PyResult<Vec<(String, f64)>> {
        let qv = self.syms.get(query_name)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown: {query_name}")))?;
        let mut r: Vec<(String, f64)> = self.syms.iter()
            .filter(|(n, _)| n.as_str() != query_name)
            .filter_map(|(n, v)| {
                let s = resonate(qv, v);
                if s.abs() >= threshold && s.abs() < 0.99 {
                    Some((n.clone(), s))
                } else {
                    None
                }
            }).collect();
        r.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
        Ok(r)
    }

    fn analogy(&mut self,
               system_a: Vec<String>,
               system_b: Vec<String>,
               query: &str,
    ) -> PyResult<Vec<(String, f64)>> {
        let a_refs: Vec<&[i8]> = system_a.iter()
            .map(|n| self.syms.get(n.as_str()).map(|v| v.as_slice())
                .ok_or_else(|| PyValueError::new_err(format!("Unknown: {n}"))))
            .collect::<PyResult<Vec<_>>>()?;
        let b_refs: Vec<&[i8]> = system_b.iter()
            .map(|n| self.syms.get(n.as_str()).map(|v| v.as_slice())
                .ok_or_else(|| PyValueError::new_err(format!("Unknown: {n}"))))
            .collect::<PyResult<Vec<_>>>()?;

        let sys_a = superpose(&a_refs, &mut self.rng);
        let sys_b = superpose(&b_refs, &mut self.rng);
        let mapping = bind(&sys_a, &sys_b);

        let qv = self.syms.get(query)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown: {query}")))?;
        let answer = bind(&mapping, qv);

        let mut r: Vec<(String, f64)> = self.syms.iter()
            .filter(|(n, _)| n.as_str() != query)
            .filter_map(|(n, v)| {
                let s = resonate(&answer, v);
                if s.abs() >= 0.05 && s.abs() < 0.99 {
                    Some((n.clone(), s))
                } else {
                    None
                }
            }).collect();
        r.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
        r.truncate(10);
        Ok(r)
    }

    fn symbol_count(&self) -> usize { self.syms.len() }
    fn dimensions(&self) -> usize { self.dim }
    fn has_symbol(&self, name: &str) -> bool { self.syms.contains_key(name) }
    fn symbol_names(&self) -> Vec<String> { self.syms.keys().cloned().collect() }
}

// ── Module ──────────────────────────────────────────────────────────

#[pymodule]
fn kos_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustKernel>()?;
    m.add_class::<RustVSA>()?;
    Ok(())
}

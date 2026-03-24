//! KOS V5.0 — Rust Core Engine
//!
//! Arena-based spreading activation graph + SIMD-ready hyperdimensional VSA.
//! Compiled to Python via PyO3.
//!
//! Architecture:
//!   - All nodes stored in contiguous Vec<ArenaNode> (cache-friendly)
//!   - Edges are integer indices (no pointers, no borrow checker issues)
//!   - VSA vectors are flat i8 arrays (SIMD-friendly BIND/SUPERPOSE)
//!   - Spreading activation via priority queue on the arena

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

// ── Arena Node ──────────────────────────────────────────────────────

struct ArenaNode {
    id: String,
    activation: f64,
    fuel: f64,
    last_tick: u64,
    temporal_decay: f64,
    max_energy: f64,
    connections: Vec<(usize, f32, u32)>, // (target_idx, weight, myelin)
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

// ── RustKernel (Arena Graph + VSA) ──────────────────────────────────

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
        }
    }

    fn add_node(&mut self, name: String) -> usize {
        if let Some(&i) = self.idx.get(&name) { return i; }
        let i = self.arena.len();
        self.arena.push(ArenaNode::new(name.clone(), self.td, self.me, self.dim, &mut self.rng));
        self.idx.insert(name, i);
        i
    }

    fn add_connection(&mut self, src: String, tgt: String, w: f32, text: Option<String>) {
        let si = self.add_node(src);
        let ti = self.add_node(tgt);
        // Edge
        let exists = self.arena[si].connections.iter_mut().find(|(t,_,_)| *t == ti);
        if let Some(e) = exists { e.1 = w; } else {
            self.arena[si].connections.push((ti, w, 0));
        }
        // Provenance
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
    }

    fn query(&mut self, seeds: Vec<String>, top_k: usize) -> Vec<(String, f64)> {
        self.tick += 1;
        let t = self.tick;
        let mut active: HashSet<usize> = HashSet::new();
        let mut pq: BinaryHeap<PQ> = BinaryHeap::new();

        for n in &seeds {
            if let Some(&i) = self.idx.get(n) {
                self.arena[i].receive(3.0, t);
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

            // Top-500 edges
            let mut edges: Vec<(usize, f64)> = self.arena[ni].connections.iter()
                .map(|(tgt, w, m)| (*tgt, (*w as f64) * (1.0 + (*m as f64) * 0.01)))
                .collect();
            edges.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
            edges.truncate(500);

            let out: Vec<(usize, f64)> = edges.iter()
                .filter_map(|(tgt, ew)| {
                    let p = fuel * ew * 0.8;
                    if p.abs() >= 0.05 { Some((*tgt, p)) } else { None }
                }).collect();

            // Myelin
            for (tgt, _) in &out {
                if let Some(e) = self.arena[ni].connections.iter_mut().find(|(t,_,_)| t == tgt) {
                    e.2 += 1;
                }
            }
            self.arena[ni].fuel = 0.0;

            for (ti, pe) in out {
                self.arena[ti].receive(pe, t);
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
        res
    }

    fn get_provenance(&self, a: String, b: String) -> Vec<String> {
        let (ia, ib) = match (self.idx.get(&a), self.idx.get(&b)) {
            (Some(&x), Some(&y)) => if x <= y { (x, y) } else { (y, x) },
            _ => return vec![],
        };
        self.prov.get(&(ia, ib)).cloned().unwrap_or_default()
    }

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

    fn node_count(&self) -> usize { self.arena.len() }
    fn edge_count(&self) -> usize { self.arena.iter().map(|n| n.connections.len()).sum() }
    fn has_node(&self, name: &str) -> bool { self.idx.contains_key(name) }
    fn node_names(&self) -> Vec<String> { self.arena.iter().map(|n| n.id.clone()).collect() }

    fn stats(&self) -> HashMap<String, f64> {
        let n = self.arena.len();
        let e: usize = self.arena.iter().map(|n| n.connections.len()).sum();
        let mut m = HashMap::new();
        m.insert("nodes".into(), n as f64);
        m.insert("edges".into(), e as f64);
        m.insert("vsa_dim".into(), self.dim as f64);
        m.insert("vsa_mb".into(), (n * 2 * self.dim) as f64 / 1048576.0);
        m.insert("arena_contiguous".into(), 1.0);
        m
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

    fn node(&mut self, name: String) -> Vec<i8> {
        let v: Vec<i8> = (0..self.dim).map(|_| if self.rng.random_bool(0.5) { 1 } else { -1 }).collect();
        self.syms.insert(name, v.clone());
        v
    }

    fn bind(&self, a: Vec<i8>, b: Vec<i8>) -> PyResult<Vec<i8>> {
        if a.len() != b.len() { return Err(PyValueError::new_err("Dim mismatch")); }
        Ok(bind(&a, &b))
    }

    fn superpose(&mut self, vecs: Vec<Vec<i8>>) -> PyResult<Vec<i8>> {
        let refs: Vec<&[i8]> = vecs.iter().map(|v| v.as_slice()).collect();
        Ok(superpose(&refs, &mut self.rng))
    }

    fn resonate(&self, a: Vec<i8>, b: Vec<i8>) -> PyResult<f64> {
        if a.len() != b.len() { return Err(PyValueError::new_err("Dim mismatch")); }
        Ok(resonate(&a, &b))
    }

    fn permute(&self, v: Vec<i8>, shifts: i32) -> Vec<i8> {
        let n = v.len() as i32;
        let s = ((shifts % n) + n) % n;
        let mut r = vec![0i8; v.len()];
        for i in 0..v.len() { r[((i as i32 + s) % n) as usize] = v[i]; }
        r
    }

    fn cleanup(&self, q: Vec<i8>, threshold: f64) -> Vec<(String, f64)> {
        let mut r: Vec<(String, f64)> = self.syms.iter()
            .filter_map(|(n, v)| {
                let s = resonate(&q, v);
                if s.abs() >= threshold { Some((n.clone(), s)) } else { None }
            }).collect();
        r.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
        r
    }

    fn store(&mut self, name: String, v: Vec<i8>) { self.syms.insert(name, v); }
    fn get(&self, name: &str) -> PyResult<Vec<i8>> {
        self.syms.get(name).cloned().ok_or_else(|| PyValueError::new_err(format!("Unknown: {name}")))
    }
    fn symbol_count(&self) -> usize { self.syms.len() }
    fn dimensions(&self) -> usize { self.dim }
}

// ── Module ──────────────────────────────────────────────────────────

#[pymodule]
fn kos_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustKernel>()?;
    m.add_class::<RustVSA>()?;
    Ok(())
}

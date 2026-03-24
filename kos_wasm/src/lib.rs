//! KOS V7.0 — WebAssembly Build
//!
//! Runs the KOS graph engine + VSA entirely in the browser.
//! No server. No API. No data leaves the user's machine.
//!
//! Compiled with: wasm-pack build --target web

use wasm_bindgen::prelude::*;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;

// ── Arena Node ──────────────────────────────────────────

struct Node {
    id: String,
    activation: f64,
    fuel: f64,
    connections: Vec<(usize, f32, u32)>, // (target, weight, myelin)
    temporal_decay: f64,
    max_energy: f64,
    last_tick: u64,
}

impl Node {
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

#[derive(PartialEq)]
struct PQ { fuel: f64, idx: usize }
impl Eq for PQ {}
impl PartialOrd for PQ {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for PQ {
    fn cmp(&self, o: &Self) -> Ordering {
        self.fuel.partial_cmp(&o.fuel).unwrap_or(Ordering::Equal)
    }
}

// ── VSA Operations ──────────────────────────────────────

fn vsa_bind(a: &[i8], b: &[i8]) -> Vec<i8> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

fn vsa_resonate(a: &[i8], b: &[i8]) -> f64 {
    let dot: i64 = a.iter().zip(b).map(|(x, y)| (*x as i64) * (*y as i64)).sum();
    dot as f64 / a.len() as f64
}

fn vsa_superpose(vecs: &[&[i8]], rng: &mut SmallRng) -> Vec<i8> {
    let d = vecs[0].len();
    let mut r = vec![0i32; d];
    for v in vecs {
        for i in 0..d { r[i] += v[i] as i32; }
    }
    r.iter().map(|&s| {
        if s > 0 { 1 } else if s < 0 { -1 } else {
            if rng.gen_bool(0.5) { 1 } else { -1 }
        }
    }).collect()
}

// ── WASM KOS Engine ─────────────────────────────────────

#[wasm_bindgen]
pub struct WasmKOS {
    arena: Vec<Node>,
    idx: HashMap<String, usize>,
    provenance: HashMap<String, Vec<String>>,
    tick: u64,
    dim: usize,
    rng: SmallRng,
    // VSA symbols
    vsa_symbols: HashMap<String, Vec<i8>>,
}

#[wasm_bindgen]
impl WasmKOS {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize) -> Self {
        WasmKOS {
            arena: Vec::with_capacity(1000),
            idx: HashMap::new(),
            provenance: HashMap::new(),
            tick: 0,
            dim,
            rng: SmallRng::seed_from_u64(42),
            vsa_symbols: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, name: &str) -> usize {
        if let Some(&i) = self.idx.get(name) { return i; }
        let i = self.arena.len();
        self.arena.push(Node {
            id: name.to_string(),
            activation: 0.0, fuel: 0.0,
            connections: Vec::new(),
            temporal_decay: 0.7, max_energy: 3.0,
            last_tick: 0,
        });
        self.idx.insert(name.to_string(), i);
        i
    }

    pub fn add_connection(&mut self, src: &str, tgt: &str, weight: f32, text: &str) {
        let si = self.add_node(src);
        let ti = self.add_node(tgt);
        let exists = self.arena[si].connections.iter_mut().find(|(t,_,_)| *t == ti);
        if let Some(e) = exists { e.1 = weight; } else {
            self.arena[si].connections.push((ti, weight, 0));
        }
        if !text.is_empty() {
            let key = format!("{}|{}", src.min(tgt), src.max(tgt));
            self.provenance.entry(key).or_default().push(text.to_string());
        }
    }

    pub fn query(&mut self, seeds_json: &str, top_k: usize) -> String {
        self.tick += 1;
        let t = self.tick;
        let seeds: Vec<String> = serde_json::from_str(seeds_json).unwrap_or_default();

        let mut active = std::collections::HashSet::new();
        let mut pq = BinaryHeap::new();

        for s in &seeds {
            if let Some(&i) = self.idx.get(s.as_str()) {
                self.arena[i].receive(3.0, t);
                pq.push(PQ { fuel: self.arena[i].fuel, idx: i });
                active.insert(i);
            }
        }

        let mut ticks = 0u64;
        while let Some(e) = pq.pop() {
            if ticks >= 15 { break; }
            let ni = e.idx;
            if self.arena[ni].fuel < 0.05 { continue; }
            ticks += 1;

            self.arena[ni].decay(t);
            let fuel = self.arena[ni].fuel;

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
                    pq.push(PQ { fuel: self.arena[ti].fuel, idx: ti });
                }
            }
        }

        let seed_idx: std::collections::HashSet<usize> =
            seeds.iter().filter_map(|n| self.idx.get(n.as_str()).copied()).collect();
        let mut results: Vec<(String, f64)> = Vec::new();
        for &i in &active {
            self.arena[i].decay(t);
            let a = self.arena[i].activation;
            if a > 0.1 && !seed_idx.contains(&i) {
                results.push((self.arena[i].id.clone(), a));
            }
            self.arena[i].fuel = 0.0;
            self.arena[i].activation = 0.0;
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(top_k);

        serde_json::to_string(&results).unwrap_or_default()
    }

    pub fn node_count(&self) -> usize { self.arena.len() }
    pub fn edge_count(&self) -> usize { self.arena.iter().map(|n| n.connections.len()).sum() }

    // ── VSA Operations ──────────────────────────────────

    pub fn vsa_node(&mut self, name: &str) {
        let v: Vec<i8> = (0..self.dim)
            .map(|_| if self.rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();
        self.vsa_symbols.insert(name.to_string(), v);
    }

    pub fn vsa_bind(&mut self, result: &str, a: &str, b: &str) -> bool {
        let va = match self.vsa_symbols.get(a) { Some(v) => v.clone(), None => return false };
        let vb = match self.vsa_symbols.get(b) { Some(v) => v, None => return false };
        let r = vsa_bind(&va, vb);
        self.vsa_symbols.insert(result.to_string(), r);
        true
    }

    pub fn vsa_resonate(&self, a: &str, b: &str) -> f64 {
        match (self.vsa_symbols.get(a), self.vsa_symbols.get(b)) {
            (Some(va), Some(vb)) => vsa_resonate(va, vb),
            _ => 0.0,
        }
    }

    pub fn vsa_superpose(&mut self, result: &str, names_json: &str) -> bool {
        let names: Vec<String> = serde_json::from_str(names_json).unwrap_or_default();
        let vecs: Vec<&[i8]> = names.iter()
            .filter_map(|n| self.vsa_symbols.get(n).map(|v| v.as_slice()))
            .collect();
        if vecs.is_empty() { return false; }
        let r = vsa_superpose(&vecs, &mut self.rng);
        self.vsa_symbols.insert(result.to_string(), r);
        true
    }

    pub fn get_provenance(&self, a: &str, b: &str) -> String {
        let key = format!("{}|{}", a.min(b), a.max(b));
        serde_json::to_string(
            self.provenance.get(&key).unwrap_or(&Vec::new())
        ).unwrap_or_default()
    }
}

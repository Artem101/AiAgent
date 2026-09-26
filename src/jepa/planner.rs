//! Inference-time latent trajectory search. No tokens, no decoding: the search runs entirely
//! in `ℝ^{d_s}` through the learned world model.
//!
//! **MPPI** (Model Predictive Path Integral), per iteration `i`:
//!
//! ```text
//!   A_m   = clip(U + ε_m, −1, 1),   ε_m ~ N(0, σ_i²),  ε_0 = 0 (keeps the nominal)
//!   s_{m,t+1} = P_φ(s_{m,t}, a_{m,t}),   s_{m,0} = s_0                  (M parallel rollouts)
//!   E_m   = ‖s_{m,H} − g‖²/d_s + λ_a · ‖A_m‖²/(H·d_a)                  (energy)
//!   w_m   = softmax_m(−(E_m − min E) / λ)
//!   U     ← Σ_m w_m A_m
//! ```
//!
//! Rollouts run in parallel with rayon; sample `m` of iteration `i` draws from its own RNG
//! stream `(seed, i, m)`, so the result is bit-identical for any thread count. All buffers
//! live in [`MppiWorkspace`]; [`JEPAPlanner::plan_into`] allocates nothing.
//!
//! **Policy prior.** When a learned proposal `π(s, ĝ) → a` is attached
//! ([`JEPAPlanner::with_policy`]), the nominal `U` is warm-started from the policy rollout
//! instead of zero actions (amortised planning, as in TD-MPC). MPPI keeps the best
//! trajectory seen, so its energy is never worse than the warm start.
//!
//! **Latent tree search** (`PlannerConfig::tree_beam > 0`) replaces the single policy rollout
//! as the warm start — a beam search over "thoughts" in latent space:
//!
//! ```text
//!   beam = {s_0}
//!   for t in 0..H:                                     (depth = one thought step)
//!     for every node b in the beam, k in 0..K:
//!       a = clip(tanh π([s_t^b ; ĝ]) + ε_{t,b,k}),  ε_{·,·,0} = 0      (proposal k)
//!       s_{t+1} = P_φ(s_t^b, a)
//!       complete greedily with π up to H → a full hypothesis, score = E(hypothesis)
//!     keep the B hypotheses with the lowest energy, prune the rest (dead ends)
//! ```
//!
//! Child `k = 0` of the best node reproduces its parent's completion, so the best energy never
//! increases with depth and the result is never worse than the plain policy rollout. The
//! search is sequential and allocation-free; the per-depth statistics and the surviving
//! hypotheses stay in the workspace ([`MppiWorkspace::tree_depths`],
//! [`MppiWorkspace::hypotheses`]).
//!
//! **Latent GD** ([`GradientPlanner`]) refines the MPPI actions by back-propagating the same
//! energy through the (differentiable, candle) world model.

use candle_core::{bail, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rayon::prelude::*;

use super::world_model::{PackedWorldModel, WorldModel};
use crate::arena::Arena;
use crate::config::PlannerConfig;
use crate::kernels::inplace::{copy_from_slice, host_read};
use crate::kernels::{parallel_for, rng::Rng, tanh_inplace, PackedMlp};
use crate::types::{LatentPlan, LatentState};

/// Diagnostics of one planning call.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlanStats {
    /// Energy of the returned trajectory.
    pub energy: f32,
    /// `‖s_H − g‖² / d_s` of the returned trajectory.
    pub terminal_error: f32,
    /// Energy of the warm-start trajectory (policy rollout, or zero actions).
    pub initial_energy: f32,
    /// Effective sample size `1 / Σ w_m²` of the last MPPI iteration.
    pub effective_samples: f32,
    /// Energy of the plain policy rollout (the greedy chain of thoughts).
    pub greedy_energy: f32,
    /// Hypotheses expanded and pruned by the tree search (0 when it is off).
    pub tree_nodes: u32,
    pub tree_pruned: u32,
}

impl std::fmt::Display for PlanStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.tree_nodes > 0 {
            write!(
                f,
                "{} hypotheses, {} pruned | energy: greedy {:.4} → tree {:.4} → final {:.4} | ESS {:.1}",
                self.tree_nodes,
                self.tree_pruned,
                self.greedy_energy,
                self.initial_energy,
                self.energy,
                self.effective_samples
            )
        } else {
            write!(
                f,
                "energy {:.4} (warm start {:.4}), ESS {:.1}",
                self.energy, self.initial_energy, self.effective_samples
            )
        }
    }
}

/// Per-sample view handed to a rollout task: `(m, ((((actions, states), cat), hidden), cost))`.
type SampleSlot<'a> = (usize, ((((&'a mut [f32], &'a mut [f32]), &'a mut [f32]), &'a mut [f32]), &'a mut f32));

/// Pre-allocated MPPI buffers.
#[derive(Debug)]
pub struct MppiWorkspace {
    actions: Box<[f32]>,
    states: Box<[f32]>,
    cat: Box<[f32]>,
    hidden: Box<[f32]>,
    costs: Box<[f32]>,
    weights: Box<[f32]>,
    nominal: Box<[f32]>,
    s0: Box<[f32]>,
    goal: Box<[f32]>,
    best: Box<[f32]>,
    best_actions: Box<[f32]>,
    pol_in: Box<[f32]>,
    pol_hidden: Box<[f32]>,
    tree: TreeBuffers,
    /// The resulting plan `[H + 1, d_s]`.
    pub plan: Tensor,
}

/// Buffers of the latent tree search (empty when it is off).
#[derive(Debug)]
struct TreeBuffers {
    /// Surviving hypotheses: full trajectories `B × (H + 1) × d_s`, actions `B × H × d_a`,
    /// energies `B` (sorted, best first).
    beam_states: Box<[f32]>,
    beam_actions: Box<[f32]>,
    beam_cost: Box<[f32]>,
    beam_len: usize,
    /// Children of one depth: `B·K` hypotheses.
    cand_states: Box<[f32]>,
    cand_actions: Box<[f32]>,
    cand_cost: Box<[f32]>,
    order: Box<[u32]>,
    /// Per depth: `[expanded, best kept, worst kept, best pruned (NaN if none)]`.
    depths: Box<[f32]>,
}

/// Statistics of one depth of the tree search.
#[derive(Debug, Clone, Copy)]
pub struct DepthStats {
    pub expanded: usize,
    pub best_kept: f32,
    pub worst_kept: f32,
    /// Lowest energy among the pruned hypotheses (`None` if nothing was pruned).
    pub best_pruned: Option<f32>,
}

impl MppiWorkspace {
    /// Nominal (optimised) action sequence `[H · d_a]`.
    pub fn nominal_actions(&self) -> &[f32] {
        &self.nominal
    }

    /// Per-depth statistics of the last tree search (empty when it is off).
    pub fn tree_depths(&self) -> impl Iterator<Item = DepthStats> + '_ {
        let n = if self.tree.beam_len > 0 { self.tree.depths.len() / 4 } else { 0 };
        self.tree.depths.chunks_exact(4).take(n).map(|d| DepthStats {
            expanded: d[0] as usize,
            best_kept: d[1],
            worst_kept: d[2],
            best_pruned: (!d[3].is_nan()).then_some(d[3]),
        })
    }

    /// Surviving hypotheses of the last tree search, best first: `(energy, trajectory [H + 1, d_s])`.
    pub fn hypotheses(&self) -> impl Iterator<Item = (f32, &[f32])> + '_ {
        let t = &self.tree;
        let len = if t.beam_len > 0 { t.beam_states.len() / t.beam_cost.len().max(1) } else { 0 };
        t.beam_cost.iter().zip(t.beam_states.chunks_exact(len.max(1))).take(t.beam_len).map(|(&c, s)| (c, s))
    }
}

/// MPPI trajectory optimiser over a packed world model (see the module docs).
pub struct JEPAPlanner {
    /// Transition model used for every rollout.
    pub world_model: PackedWorldModel,
    /// Number of actions `H` per trajectory.
    pub horizon: usize,
    /// Sampled trajectories `M` per iteration.
    pub num_samples: usize,
    /// Softmax temperature λ of the path-integral weights.
    pub temperature: f64,
    /// Remaining planner settings (noise, iterations, action cost, …).
    pub config: PlannerConfig,
    /// Seed used by the allocating [`JEPAPlanner::plan`] convenience method.
    pub seed: u64,
    /// Optional proposal policy `a = tanh(π([s ; ĝ]))` used to warm-start the nominal.
    pub policy: Option<PackedMlp>,
}

impl JEPAPlanner {
    pub fn new(world_model: PackedWorldModel, horizon: usize, config: &PlannerConfig) -> Self {
        Self {
            world_model,
            horizon,
            num_samples: config.num_samples,
            temperature: config.temperature,
            config: config.clone(),
            seed: 0x5EED,
            policy: None,
        }
    }

    /// Attaches a learned proposal policy (see module docs).
    pub fn with_policy(mut self, policy: PackedMlp) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Allocates every buffer [`JEPAPlanner::plan_into`] needs.
    pub fn workspace(&self, arena: &mut Arena) -> Result<MppiWorkspace> {
        let (m, h) = (self.num_samples, self.horizon);
        let (ds, da, dh) = (self.world_model.d_state, self.world_model.d_action, self.world_model.d_hidden());
        Ok(MppiWorkspace {
            actions: arena.host(m * h * da),
            states: arena.host(m * (h + 1) * ds),
            cat: arena.host(m * (ds + da)),
            hidden: arena.host(m * dh),
            costs: arena.host(m),
            weights: arena.host(m),
            nominal: arena.host(h * da),
            s0: arena.host(ds),
            goal: arena.host(ds),
            best: arena.host((h + 1) * ds),
            best_actions: arena.host(h * da),
            pol_in: arena.host(2 * ds),
            pol_hidden: arena.host(self.policy.as_ref().map_or(0, |p| p.l1.d_out)),
            tree: {
                let (b, k) =
                    if self.tree_enabled() { (self.config.tree_beam, self.config.tree_branch.max(1)) } else { (0, 0) };
                TreeBuffers {
                    beam_states: arena.host(b * (h + 1) * ds),
                    beam_actions: arena.host(b * h * da),
                    beam_cost: arena.host(b),
                    beam_len: 0,
                    cand_states: arena.host(b * k * (h + 1) * ds),
                    cand_actions: arena.host(b * k * h * da),
                    cand_cost: arena.host(b * k),
                    order: arena.host_u32(b * k),
                    depths: arena.host(if b > 0 { 4 * h } else { 0 }),
                }
            },
            plan: arena.tensor((h + 1, ds))?,
        })
    }

    /// Whether the latent tree search runs (it needs the policy prior).
    pub fn tree_enabled(&self) -> bool {
        self.config.tree_beam > 0 && self.config.policy_prior && self.policy.is_some()
    }

    /// Completes a hypothesis greedily with the policy: `states[..=from]` and `actions[..from]`
    /// are given, the rest up to depth `H` is filled in. Returns its energy.
    #[allow(clippy::too_many_arguments)]
    fn complete(
        &self,
        pi: &PackedMlp,
        from: usize,
        states: &mut [f32],
        actions: &mut [f32],
        goal: &[f32],
        pol_in: &mut [f32],
        pol_hidden: &mut [f32],
        cat: &mut [f32],
        hidden: &mut [f32],
    ) -> f32 {
        let wm = &self.world_model;
        let (h, ds, da) = (self.horizon, wm.d_state, wm.d_action);
        for t in from..h {
            pol_in[..ds].copy_from_slice(&states[t * ds..(t + 1) * ds]);
            pol_in[ds..].copy_from_slice(goal);
            let a = &mut actions[t * da..(t + 1) * da];
            pi.forward(pol_in, pol_hidden, a);
            tanh_inplace(a);
            let (done, rest) = states.split_at_mut((t + 1) * ds);
            wm.step(&done[t * ds..], a, cat, hidden, &mut rest[..ds]);
        }
        self.energy(&states[h * ds..], goal, actions)
    }

    /// `E = ‖s_H − g‖²/d_s + λ_a ‖A‖²/(H d_a)`.
    #[inline]
    pub fn energy(&self, terminal: &[f32], goal: &[f32], actions: &[f32]) -> f32 {
        let term = terminal.iter().zip(goal).map(|(s, g)| (s - g) * (s - g)).sum::<f32>() / goal.len() as f32;
        let act = actions.iter().map(|a| a * a).sum::<f32>() / actions.len().max(1) as f32;
        term + self.config.action_cost as f32 * act
    }

    /// Runs MPPI from `s0` towards the energy target `goal`; the plan is written to `ws.plan`.
    pub fn plan_into(
        &self,
        s0: &LatentState,
        goal: &LatentState,
        ws: &mut MppiWorkspace,
        seed: u64,
    ) -> Result<PlanStats> {
        let wm = &self.world_model;
        let (m, h, ds, da, dh) = (self.num_samples, self.horizon, wm.d_state, wm.d_action, wm.d_hidden());
        if s0.dim() != ds || goal.dim() != ds {
            bail!("planner: states must have width {ds}")
        }
        host_read(s0.tensor(), |s| ws.s0.copy_from_slice(s))?;
        host_read(goal.tensor(), |g| ws.goal.copy_from_slice(g))?;

        let MppiWorkspace {
            actions,
            states,
            cat,
            hidden,
            costs,
            weights,
            nominal,
            s0,
            goal,
            best,
            best_actions,
            pol_in,
            pol_hidden,
            tree,
            plan,
        } = ws;

        // warm start: latent tree search, policy rollout, or zero actions
        let (mut tree_nodes, mut tree_pruned) = (0u32, 0u32);
        tree.beam_len = 0;
        let mut greedy_energy = f32::NAN;
        match &self.policy {
            Some(pi) if self.tree_enabled() => {
                let (b_max, k_max) = (self.config.tree_beam, self.config.tree_branch.max(1));
                let (sl, al) = ((h + 1) * ds, h * da);
                let (cat1, hid1) = (&mut cat[..ds + da], &mut hidden[..dh]);
                // root: the greedy chain from s_0
                tree.beam_states[..ds].copy_from_slice(s0);
                tree.beam_cost[0] = self.complete(
                    pi,
                    0,
                    &mut tree.beam_states[..sl],
                    &mut tree.beam_actions[..al],
                    goal,
                    pol_in,
                    pol_hidden,
                    cat1,
                    hid1,
                );
                greedy_energy = tree.beam_cost[0];
                tree.beam_len = 1;
                let sigma = self.config.tree_noise as f32;
                for t in 0..h {
                    let mut n = 0;
                    for b in 0..tree.beam_len {
                        for k in 0..k_max {
                            let (cs, ca) = (
                                &mut tree.cand_states[n * sl..(n + 1) * sl],
                                &mut tree.cand_actions[n * al..(n + 1) * al],
                            );
                            cs[..(t + 1) * ds].copy_from_slice(&tree.beam_states[b * sl..b * sl + (t + 1) * ds]);
                            ca[..t * da].copy_from_slice(&tree.beam_actions[b * al..b * al + t * da]);
                            // proposal k: the policy's action, perturbed for k > 0
                            pol_in[..ds].copy_from_slice(&cs[t * ds..(t + 1) * ds]);
                            pol_in[ds..].copy_from_slice(goal);
                            let a = &mut ca[t * da..(t + 1) * da];
                            pi.forward(pol_in, pol_hidden, a);
                            tanh_inplace(a);
                            if k > 0 {
                                let mut rng = Rng::stream(seed ^ 0x07EE_5EED, t as u64, (b * k_max + k) as u64);
                                for x in a.iter_mut() {
                                    *x = (*x + sigma * rng.normal()).clamp(-1.0, 1.0);
                                }
                            }
                            let (done, rest) = cs.split_at_mut((t + 1) * ds);
                            wm.step(&done[t * ds..], a, cat1, hid1, &mut rest[..ds]);
                            tree.cand_cost[n] = self.complete(pi, t + 1, cs, ca, goal, pol_in, pol_hidden, cat1, hid1);
                            n += 1;
                        }
                    }
                    // keep the B best hypotheses (ties → lower index, deterministic)
                    let order = &mut tree.order[..n];
                    for (i, o) in order.iter_mut().enumerate() {
                        *o = i as u32;
                    }
                    let cost = &tree.cand_cost;
                    order.sort_unstable_by(|&x, &y| cost[x as usize].total_cmp(&cost[y as usize]).then(x.cmp(&y)));
                    let keep = b_max.min(n);
                    let d = &mut tree.depths[4 * t..4 * t + 4];
                    d[0] = n as f32;
                    d[1] = cost[order[0] as usize];
                    d[2] = cost[order[keep - 1] as usize];
                    d[3] = if n > keep { cost[order[keep] as usize] } else { f32::NAN };
                    tree_nodes += n as u32;
                    tree_pruned += (n - keep) as u32;
                    for (i, &c) in order[..keep].iter().enumerate() {
                        let c = c as usize;
                        tree.beam_states[i * sl..(i + 1) * sl].copy_from_slice(&tree.cand_states[c * sl..(c + 1) * sl]);
                        tree.beam_actions[i * al..(i + 1) * al]
                            .copy_from_slice(&tree.cand_actions[c * al..(c + 1) * al]);
                        tree.beam_cost[i] = tree.cand_cost[c];
                    }
                    tree.beam_len = keep;
                }
                nominal.copy_from_slice(&tree.beam_actions[..al]);
                best.copy_from_slice(&tree.beam_states[..sl]);
            }
            Some(pi) if self.config.policy_prior => {
                best[..ds].copy_from_slice(s0);
                for t in 0..h {
                    pol_in[..ds].copy_from_slice(&best[t * ds..(t + 1) * ds]);
                    pol_in[ds..].copy_from_slice(goal);
                    let a = &mut nominal[t * da..(t + 1) * da];
                    pi.forward(pol_in, pol_hidden, a);
                    tanh_inplace(a);
                    let (done, rest) = best.split_at_mut((t + 1) * ds);
                    wm.step(&done[t * ds..], a, &mut cat[..ds + da], &mut hidden[..dh], &mut rest[..ds]);
                }
            }
            _ => {
                nominal.fill(0.0);
                wm.rollout(s0, nominal, best, &mut cat[..ds + da], &mut hidden[..dh]);
            }
        }
        let initial_energy = self.energy(&best[h * ds..], goal, nominal);
        if greedy_energy.is_nan() {
            greedy_energy = initial_energy;
        }
        let mut best_cost = initial_energy;
        best_actions.copy_from_slice(nominal);
        let mut ess = m as f32;

        let par = m > 1 && parallel_for(m * h * (ds + da + ds) * dh);
        for it in 0..self.config.iterations {
            let std = (self.config.noise_std * self.config.noise_decay.powi(it as i32)) as f32;
            let (nominal_r, s0_r, goal_r) = (&**nominal, &**s0, &**goal);
            let simulate = |(mi, ((((acts, st), cat), hid), cost)): SampleSlot<'_>| {
                if mi == 0 {
                    acts.fill(0.0);
                } else {
                    Rng::stream(seed, it as u64, mi as u64).fill_normal(acts, std);
                }
                for (a, &u) in acts.iter_mut().zip(nominal_r) {
                    *a = (u + *a).clamp(-1.0, 1.0);
                }
                wm.rollout(s0_r, acts, st, cat, hid);
                *cost = self.energy(&st[h * ds..], goal_r, acts);
            };
            if par {
                actions
                    .par_chunks_mut(h * da)
                    .zip(states.par_chunks_mut((h + 1) * ds))
                    .zip(cat.par_chunks_mut(ds + da))
                    .zip(hidden.par_chunks_mut(dh))
                    .zip(costs.par_iter_mut())
                    .enumerate()
                    .with_min_len(4)
                    .for_each(simulate);
            } else {
                actions
                    .chunks_mut(h * da)
                    .zip(states.chunks_mut((h + 1) * ds))
                    .zip(cat.chunks_mut(ds + da))
                    .zip(hidden.chunks_mut(dh))
                    .zip(costs.iter_mut())
                    .enumerate()
                    .for_each(simulate);
            }

            // path-integral weights (sequential reduction → deterministic)
            let (mut imin, mut cmin) = (0, f32::INFINITY);
            for (i, &c) in costs.iter().enumerate() {
                if c < cmin {
                    (imin, cmin) = (i, c);
                }
            }
            if cmin < best_cost {
                best_cost = cmin;
                best_actions.copy_from_slice(&actions[imin * h * da..(imin + 1) * h * da]);
            }
            let cmean = costs.iter().sum::<f32>() / m as f32;
            let lambda = if self.config.normalize_costs {
                (self.temperature as f32 * (cmean - cmin)).max(1e-12)
            } else {
                self.temperature as f32
            };
            let mut z = 0.0;
            for (w, &c) in weights.iter_mut().zip(costs.iter()) {
                *w = (-(c - cmin) / lambda).exp();
                z += *w;
            }
            let mut sq = 0.0;
            for w in weights.iter_mut() {
                *w /= z;
                sq += *w * *w;
            }
            ess = 1.0 / sq;
            nominal.fill(0.0);
            for (w, acts) in weights.iter().zip(actions.chunks_exact(h * da)) {
                crate::kernels::axpy(nominal, *w, acts);
            }
        }

        // final trajectory: the MPPI mean, unless an explicit sample was better
        wm.rollout(s0, nominal, best, &mut cat[..ds + da], &mut hidden[..dh]);
        if self.energy(&best[h * ds..], goal, nominal) > best_cost {
            nominal.copy_from_slice(best_actions);
            wm.rollout(s0, nominal, best, &mut cat[..ds + da], &mut hidden[..dh]);
        }
        let energy = self.energy(&best[h * ds..], goal, nominal);
        let terminal_error = self.energy(&best[h * ds..], goal, &[]);
        copy_from_slice(plan, best)?;
        Ok(PlanStats {
            energy,
            terminal_error,
            initial_energy,
            effective_samples: ess,
            greedy_energy,
            tree_nodes,
            tree_pruned,
        })
    }

    /// Allocating convenience with the spec signature: plans from `initial_state [d_s]`
    /// towards `energy_target [d_s]`.
    pub fn plan(&self, initial_state: &Tensor, energy_target: &Tensor, device: &Device) -> Result<LatentPlan> {
        let ds = self.world_model.d_state;
        let mut arena = Arena::new(&Device::Cpu);
        let mut ws = self.workspace(&mut arena)?;
        let host = |t: &Tensor| -> Result<LatentState> {
            LatentState::new(t.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?.contiguous()?, ds)
        };
        self.plan_into(&host(initial_state)?, &host(energy_target)?, &mut ws, self.seed)?;
        LatentPlan::new(ws.plan.to_device(device)?, self.horizon + 1, ds)
    }
}

/// Gradient-based latent trajectory optimisation ("latent GD").
pub struct GradientPlanner {
    pub steps: usize,
    pub lr: f64,
    pub action_cost: f64,
}

/// Output of [`GradientPlanner::refine`].
pub struct GradientPlan {
    /// `[H, d_a]`.
    pub actions: Tensor,
    /// `[H + 1, d_s]`.
    pub trajectory: Tensor,
    pub energy: f32,
}

impl GradientPlanner {
    /// Minimises the MPPI energy over `a = tanh(u)` with Adam, starting from `init_actions`.
    pub fn refine(&self, wm: &WorldModel, s0: &Tensor, goal: &Tensor, init_actions: &Tensor) -> Result<GradientPlan> {
        let a0 = init_actions.clamp(-0.999f32, 0.999f32)?;
        // atanh(a) = ½ ln((1 + a) / (1 − a))
        let u0 = (((&a0 + 1.0)? / (a0.neg()? + 1.0)?)?.log()? * 0.5)?;
        let u = Var::from_tensor(&u0)?;
        let mut opt =
            AdamW::new(vec![u.clone()], ParamsAdamW { lr: self.lr, weight_decay: 0.0, ..Default::default() })?;
        let s0b = s0.unsqueeze(0)?;
        let energy = |u: &Tensor| -> Result<(Tensor, Tensor, Tensor)> {
            let a = u.tanh()?;
            let traj = wm.rollout(&s0b, &a.unsqueeze(0)?)?.squeeze(0)?;
            let h = traj.dims()[0] - 1;
            let term = traj.get(h)?.sub(goal)?.sqr()?.mean_all()?;
            let e = (term + (a.sqr()?.mean_all()? * self.action_cost)?)?;
            Ok((e, a, traj))
        };
        for _ in 0..self.steps {
            let (e, _, _) = energy(u.as_tensor())?;
            opt.backward_step(&e)?;
        }
        let (e, a, traj) = energy(u.as_tensor())?;
        Ok(GradientPlan { actions: a.detach(), trajectory: traj.detach(), energy: e.to_scalar::<f32>()? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use crate::nn::ParamStore;
    use candle_core::DType;

    fn setup() -> Result<(WorldModel, JEPAPlanner)> {
        let cfg = EngineConfig::tiny(10, 8, 8);
        let mut ps = ParamStore::new(&Device::Cpu, 5);
        let j = &cfg.jepa;
        let wm = WorldModel::new(&mut ps, "wm", j.d_state, j.d_action, j.d_hidden)?;
        let planner = JEPAPlanner::new(wm.pack(DType::F32)?, j.horizon, &cfg.planner);
        Ok((wm, planner))
    }

    #[test]
    fn packed_rollout_matches_graph() -> Result<()> {
        let (wm, planner) = setup()?;
        let dev = Device::Cpu;
        let (ds, da, h) = (wm.d_state, wm.d_action, planner.horizon);
        let s0 = Tensor::randn(0f32, 1.0, (1, ds), &dev)?;
        let acts = Tensor::randn(0f32, 0.5, (1, h, da), &dev)?;
        let want = wm.rollout(&s0, &acts)?.flatten_all()?.to_vec1::<f32>()?;
        let mut got = vec![0f32; (h + 1) * ds];
        let (mut cat, mut hid) = (vec![0f32; ds + da], vec![0f32; planner.world_model.d_hidden()]);
        planner.world_model.rollout(
            &s0.flatten_all()?.to_vec1::<f32>()?,
            &acts.flatten_all()?.to_vec1::<f32>()?,
            &mut got,
            &mut cat,
            &mut hid,
        );
        let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "rollout mismatch {err}");
        Ok(())
    }

    #[test]
    fn mppi_reduces_energy_and_is_deterministic() -> Result<()> {
        let (wm, planner) = setup()?;
        let dev = Device::Cpu;
        let ds = wm.d_state;
        // A reachable goal: the terminal state of a random action sequence.
        let s0 = Tensor::randn(0f32, 1.0, ds, &dev)?;
        let hidden_actions = Tensor::rand(-0.8f32, 0.8, (1, planner.horizon, wm.d_action), &dev)?;
        let goal = wm.rollout(&s0.unsqueeze(0)?, &hidden_actions)?.get(0)?.get(planner.horizon)?;
        let (s0, goal) = (LatentState::new(s0, ds)?, LatentState::new(goal, ds)?);

        let mut arena = Arena::new(&dev);
        let mut ws = planner.workspace(&mut arena)?;
        let stats = planner.plan_into(&s0, &goal, &mut ws, 1)?;
        assert!(stats.energy < 0.5 * stats.initial_energy, "{stats:?}");
        let first = ws.plan.to_vec2::<f32>()?;

        crate::kernels::set_parallel(false);
        let stats2 = planner.plan_into(&s0, &goal, &mut ws, 1)?;
        crate::kernels::set_parallel(true);
        assert_eq!(first, ws.plan.to_vec2::<f32>()?, "MPPI must not depend on thread scheduling");
        assert_eq!(stats.energy, stats2.energy);

        // Latent GD polishes the MPPI solution further.
        let gd = GradientPlanner { steps: 30, lr: 0.05, action_cost: planner.config.action_cost };
        let init = Tensor::from_slice(ws.nominal_actions(), (planner.horizon, wm.d_action), &dev)?;
        let refined = gd.refine(&wm, s0.tensor(), goal.tensor(), &init)?;
        assert!(refined.energy <= stats.energy * 1.001, "{} vs {}", refined.energy, stats.energy);

        // spec-style allocating API
        let plan = planner.plan(s0.tensor(), goal.tensor(), &dev)?;
        assert_eq!(plan.len(), planner.horizon + 1);
        Ok(())
    }

    #[test]
    fn tree_search_prunes_and_never_loses_to_greedy() -> Result<()> {
        let dev = Device::Cpu;
        let mut cfg = EngineConfig::tiny(10, 8, 8);
        cfg.jepa.horizon = 8;
        cfg.planner.tree_beam = 3;
        cfg.planner.tree_branch = 4;
        cfg.planner.iterations = 0; // look at the tree alone
        let (j, mut ps) = (&cfg.jepa, ParamStore::new(&dev, 5));
        let wm = WorldModel::new(&mut ps, "wm", j.d_state, j.d_action, j.d_hidden)?;
        let policy = ps.mlp("pi", 2 * j.d_state, j.d_hidden, j.d_action)?.pack(DType::F32)?;
        let planner = JEPAPlanner::new(wm.pack(DType::F32)?, j.horizon, &cfg.planner).with_policy(policy);
        assert!(planner.tree_enabled());
        let (ds, h) = (j.d_state, j.horizon);
        let mut ws = planner.workspace(&mut Arena::new(&dev))?;
        for trial in 0..5u64 {
            let s0 = LatentState::new(Tensor::randn(0f32, 1.0, ds, &dev)?, ds)?;
            let goal = LatentState::new(Tensor::randn(0f32, 1.0, ds, &dev)?, ds)?;
            let stats = planner.plan_into(&s0, &goal, &mut ws, trial)?;
            // K children at the root, B·K at every other depth; all but B are pruned
            assert_eq!(stats.tree_nodes as usize, 4 + 3 * 4 * (h - 1));
            assert_eq!(stats.tree_pruned as usize, stats.tree_nodes as usize - 3 * h);
            assert!(stats.initial_energy <= stats.greedy_energy, "{stats:?}");
            let depths: Vec<_> = ws.tree_depths().collect();
            assert_eq!(depths.len(), h);
            for w in depths.windows(2) {
                assert!(w[1].best_kept <= w[0].best_kept, "best hypothesis never gets worse: {depths:?}");
            }
            let hyps: Vec<f32> = ws
                .hypotheses()
                .map(|(e, traj)| {
                    assert_eq!(traj.len(), (h + 1) * ds);
                    e
                })
                .collect();
            assert_eq!(hyps.len(), 3);
            assert!(hyps.windows(2).all(|w| w[0] <= w[1]));
            assert_eq!(hyps[0], stats.initial_energy);
            // deterministic
            let first = ws.plan.to_vec2::<f32>()?;
            planner.plan_into(&s0, &goal, &mut ws, trial)?;
            assert_eq!(first, ws.plan.to_vec2::<f32>()?);
        }
        Ok(())
    }
}

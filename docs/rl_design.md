# Reinforcement-learning design

## Environment (`reinforcement_learning/environment/route_env.py`)

| Element | Definition |
|---|---|
| Graph | Directed Hyderabad road graph: nodes = intersections, edges = road segments |
| Agent position | current intersection `u` |
| Goal | destination intersection `g` (snapped from the requested point) |
| Actions | outgoing edges of `u`. Edges not leaving `u` are invalid (penalised, state unchanged). The immediate U-turn to the previous node is removed unless it is the only exit. |
| Transition | move to `v`; simulation clock advances by the edge travel time `tt_e(t)`, so congestion observed next is time-dependent |
| Termination | reach `g` · dead end · `max_steps` (truncation) |

### State
`s = (u, g, d(u,g), b(t), c̄_local(t), elapsed time, visited set)`
* `d(u,g)` straight-line distance in km
* `b(t)` network traffic level bucket ∈ {0,1,2,3} (thresholds 0.2 / 0.38 / 0.56 on the network level)
* `c̄_local` mean congestion of outgoing edges

Q-Learning indexes the table by `(g, preference)` → `Q[b, e]`; the DQN consumes the continuous features (below).

### Reward
For edge `e = (u → v)` taken at time `t`:

```
r = − w_time · tt_e(t)[min]
    − w_dist · len_e[km]
    − w_cong · c_e(t) · tt_e(t)[min]
    + w_prog · (Φ(v) − Φ(u)),       Φ(n) = − d(n, g)[km]   (= distance gained)
    + R_goal            if v = g          (20)
    − P_revisit         if v visited      (3)
    − P_invalid         invalid action    (5)
    − P_timeout / P_dead_end              (15)
```

| Term | Why |
|---|---|
| time | the "fastest" objective; uses time-dependent, congestion-adjusted travel time |
| distance | the "shortest" objective |
| congestion × time | penalises minutes spent in congestion (stress, fuel, risk) beyond pure time |
| progress (potential-based shaping) | dense signal toward the goal; potential-based form (Ng et al., 1999) keeps the optimal policy unchanged |
| goal bonus | makes reaching the destination strictly better than wandering |
| revisit / invalid / timeout | discourage loops and impossible moves |

Weights per preference (`PREFERENCES`):

| preference | w_time | w_dist | w_cong |
|---|---|---|---|
| fastest | 1.0 | 0.1 | 0.6 |
| shortest | 0.15 | 1.2 | 0.1 |
| balanced | 0.6 | 0.6 | 0.35 |

`w_prog = 2.0`. The task is episodic and undiscounted for Q-learning (γ = 1, a stochastic-shortest-path problem): with γ < 1 the discounted return tolerates detours (a path twice as long costs < 2× after discounting), which measurably produced longer routes. The DQN uses γ = 0.995 for numerical stability. Dijkstra uses the same `w_time / w_dist / w_cong` for its edge weights, so all algorithms optimise one objective and route scores are comparable.

## Q-Learning (`reinforcement_learning/q_learning/agent.py`)
* Goal-conditioned tabular Q: one table per (preference, destination landmark); state-action pair = edge `e` plus traffic bucket `b`.
* `Q(b,e) = Qbase[e] + Qres[b,e]` – a shared table plus a traffic-level residual (two binary tilings). A plain `[4 × E]` table split the experience four ways and left most entries with 1–2 updates; the shared part fixed that while keeping traffic-dependent behaviour.
* ε-greedy behaviour (ε 1.0 → 0.05, ×0.998 per episode), α₀ = 0.3 decaying per entry `α_n = max(0.05, α₀/√(1+n/5))` to average stochastic rewards, γ = 1.
* Update: `δ = r + γ max_{e'∈A(v)} Q(b',e') − Q(b,e)`; `Qbase[e] += ½α_n δ`, `Qres[b,e] += ½α_n δ`; no bootstrap on terminal states.
* Initial values: a deliberately pessimistic, potential-consistent estimate `Q0(u→v) = R_goal − c_km·(len_e + 1.6·d(v,g)) + w_prog·d(u,g)` (cost per km under heavy congestion). Optimistic zeros made the greedy policy chase rarely-visited segments into long detours.
* Backward replay: after each episode the transitions are replayed in reverse (2 passes) – same update rule, faster credit assignment along 100-step routes.
* Each episode randomises scenario, clock, noise and incidents (`TrafficModel.randomize`) and starts from a random intersection within 18 km.
* Inference: greedy with loop avoidance. Steps at intersections where no candidate segment was ever updated are decided by the initial estimate and counted in `fallback_steps`, which is shown in the UI.
* Destinations that are not pre-trained landmarks get an **on-demand** table (≤ `QL_ONDEMAND_EPISODES`, ≤ `QL_ONDEMAND_BUDGET_S` seconds), cached in memory (LRU 40).
* Saved as `checkpoints/q_learning.npz` (float32 base, float16 residual, uint16 visit counts, JSON meta).

## Deep Q-Network (`reinforcement_learning/dqn/`)
* **Action-in architecture** for the variable action space: each candidate edge is a row `[8 state features | 18 action features]`; an MLP `26 → 128 → 128 → 64 → 1` outputs `Q(s,a)` per row. Up to 8 rows; padding rows are masked to `−∞` for both acting and the Bellman max.
* Action features: length, congested travel time, congestion, free-flow speed, progress ratio, next-node distance to goal, distance reduction, next node visited, is goal, next out-degree, dead end, lanes, road-class one-hot.
* Experience replay (40k), batch 128, Adam 5e-4, Huber loss, gradient clip 10.
* Double-DQN target `y = r + γ (1−done) Q_target(s', argmax_a' Q_online(s',a'))`, target network hard-copied every 500 gradient steps.
* Curriculum: trip length grows from ~4 km to 22 km over the first 40 % of episodes; preference sampled per episode (preference weights are inputs).
* Checkpoint of the best rolling-success model every 50 episodes: `dqn.pt` (PyTorch) + `dqn.npz` (exported weights). The web service runs the identical forward pass in NumPy (`DQN_INFERENCE_BACKEND=numpy`, test-verified parity) so it does not need torch at runtime.

## Training commands
```bash
python scripts/train_q_learning.py --episodes 3000 --alpha 0.3 --gamma 1.0 --epsilon 1.0 --seed 42
python scripts/train_dqn.py --episodes 1500 --lr 5e-4 --gamma 0.995 --batch-size 128 --buffer-size 40000 --seed 42
python scripts/evaluate_models.py --pairs 15 --scenarios low,moderate,heavy,peak --events none,incident_on_route
```
Training history is written to `checkpoints/*_history.json` and shown on the RL Training page.

## Evaluation protocol
For each trip all algorithms get: the same snapped source/destination, the same graph, an identical clone of the traffic model (same clock, noise and incidents) and the same optional event (an accident that appears on Dijkstra's departure-time corridor after departure, or a surge). Every route is replayed through `RouteEnv` to measure distance, time, congestion exposure, reward and fuel cost. Dijkstra is optimal for static costs; when nothing changes during the trip it is expected to match or beat the learned policies, and the comparison page reports those cases.

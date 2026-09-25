# Конфигурация

`EngineConfig` описывает архитектуру и инференс, `TrainConfig` — обучение. Пресеты создаются
через `EngineConfig::tiny(vocab, prompt_len, answer_len)` и `EngineConfig::small(…)`
(`EngineConfig::preset(name, …)` по имени). `CogModel::new` вызывает `validate()`.

## `EngineConfig`

| Поле | `tiny` | Смысл |
|---|---:|---|
| `vocab_size` | аргумент | размер словаря, ≥ 2 |
| `max_prompt_len` | аргумент | длина промпта, под которую обучена таблица позиций |
| `weight_dtype` | `BF16` | точность хранения упакованных весов: `F32`, `F16` или `BF16`; аккумуляция всегда f32 |
| `seed` | 7 | инициализация весов, данные, EMA-копия |

Производные величины: `answer_len() = flow.seq_len`, `plan_len() = horizon + 1`,
`max_positions() = max_prompt_len + seq_len`.

### `ttt: TTTConfig`

| Поле | `tiny` | `small` | Смысл |
|---|---:|---:|---|
| `d_model` | 64 | 128 | ширина эмбеддингов на входе TTT |
| `d_fast` | 32 | 64 | сторона `W_fast`; память контекста = `4·d_fast²` байт |
| `learning_rate` | 1.0 | 1.0 | η, верхняя граница внутреннего шага |
| `adaptive_lr` | true | true | η_t = η·σ(w·x̃_t + b) вместо константы |
| `readout_probes` | 8 | 8 | число проб `r` для чтения `W_fast` |
| `d_ctx` | 96 | 192 | ширина `S_prompt` |

### `jepa: JepaConfig`

| Поле | `tiny` | `small` | Смысл |
|---|---:|---:|---|
| `d_state` | 32 | 64 | `d_s`, ширина латентного состояния |
| `d_action` | 8 | 16 | `d_a`, ширина действия (узкое место: ограничивает, сколько информации несёт один шаг) |
| `d_hidden` | 128 | 256 | скрытый слой всех MLP JEPA |
| `horizon` | 4 | 4 | `H`, число шагов мысли; должно делить длину ответа |
| `ema_decay` | 0.99 | 0.99 | τ целевого энкодера, в `[0, 1)` |
| `vicreg.inv_weight` | 1.0 | | вес invariance |
| `vicreg.var_weight` | 0.5 | | вес variance |
| `vicreg.cov_weight` | 0.04 | | вес covariance |
| `vicreg.gamma` | 1.0 | | целевое стандартное отклонение измерения |
| `vicreg.eps` | 1e-4 | | ε под корнем в variance |
| `goal_weight` | 1.0 | | вес `‖ĝ − s̄_H‖²` |
| `policy_weight` | 1.0 | | вес behaviour cloning политики |

### `planner: PlannerConfig`

| Поле | `tiny` | `small` | Смысл |
|---|---:|---:|---|
| `kind` | `Mppi` | | `Mppi` или `MppiThenGradient` (доводка latent GD) |
| `policy_prior` | true | | тёплый старт MPPI из роллаута политики |
| `num_samples` | 128 | 256 | `M`, траекторий на итерацию |
| `iterations` | 8 | 8 | итераций MPPI; 0 — только роллаут политики |
| `temperature` | 0.1 | | λ в `softmax(−E/λ)` |
| `normalize_costs` | true | | λ' = λ·(mean E − min E) — инвариантно к масштабу энергии |
| `noise_std` | 0.6 | | σ возмущений на первой итерации (действия лежат в `[−1, 1]`) |
| `noise_decay` | 0.75 | | множитель σ на каждую итерацию |
| `action_cost` | 0.01 | | λ_a, штраф `‖A‖²` в энергии |
| `gd_steps` | 20 | | шагов Adam в latent GD |
| `gd_lr` | 0.05 | | learning rate latent GD |

Стоимость планирования пропорциональна `num_samples × iterations × horizon` роллаутам одного шага
world model. Качество растёт с каждым множителем. При хорошей политике достаточно и
`iterations = 4`.

### `flow: FlowConfig`

| Поле | `tiny` | `small` | Смысл |
|---|---:|---:|---|
| `d_token` | 32 | 64 | ширина пространства, в котором живёт поток (`X ∈ ℝ^{L×d_token}`) |
| `d_hidden` | 64 | 128 | ширина трансформера, должна делиться на `n_heads` |
| `n_heads` | 4 | 4 | головы внимания |
| `n_layers` | 2 | 3 | DiT-блоки |
| `mlp_ratio` | 4 | 4 | расширение MLP внутри блока |
| `d_time` | 64 | 128 | ширина синусоидального эмбеддинга времени, чётная |
| `seq_len` | аргумент | | длина ответа `L` |
| `solver.steps` | 16 | | `K`, шагов ODE |
| `solver.solver` | `Heun` | | `Euler` (1 NFE/шаг), `Midpoint` или `Heun` (2 NFE/шаг) |
| `solver.sigma_min` | 1e-4 | | σ_min OT-пути (используется при обучении) |

Стоимость декодинга равна `NFE × стоимость DiT на L позициях`. Уменьшение `solver.steps` —
самый прямой рычаг латентности. Его можно менять на инференсе без переобучения
(`set_solver`, `--ode-steps`), ценой точности.

## `TrainConfig`

`TrainConfig::quick(task)` задаёт значения по умолчанию:

| Поле | По умолчанию | Смысл |
|---|---:|---|
| `task` | аргумент | `Sort`, `Reverse`, `Copy` |
| `batch_size` | 64 | |
| `steps` | 1500 | |
| `lr` / `min_lr` | 2e-3 / 1e-4 | пик и минимум косинусного расписания |
| `warmup` | 100 | шагов линейного разогрева |
| `weight_decay` | 0.01 | AdamW |
| `grad_clip` | 1.0 | клиппинг глобальной нормы; 0 — выключить |
| `ce_weight` | 0.2 | вес CE по одношаговой оценке `x̂_1` |
| `head_weight` | 0.2 | вес CE головки на зашумлённых чистых эмбеддингах |
| `plan_noise` | 0.05 | σ гауссова шума, добавляемого к плану при обучении декодера |
| `compute_dtype` | `None` | `None` = f32 на CPU, bf16 на CUDA; можно `Some(F16)` и т.п. |
| `log_every` | 100 | |
| `eval_every` | 500 | 0 — без промежуточной оценки |
| `eval_samples` | 128 | размер оценочной выборки |

## Ограничения, которые проверяет `validate()`

* `flow.d_hidden % flow.n_heads == 0`;
* `flow.d_time` чётное;
* `flow.seq_len % jepa.horizon == 0`;
* `flow.solver.steps > 0`, `planner.num_samples > 0`;
* `jepa.ema_decay ∈ [0, 1)`;
* `vocab_size ≥ 2`.

Задачи из `data.rs` дополнительно требуют `max_prompt_len == seq_len`. Это проверяет `TaskSampler::new`.

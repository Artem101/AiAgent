# cog_engine — гибридный неавторегрессионный когнитивный движок (TTT + JEPA + CFM) на Rust

```
tokens ─► TTT-Encoder ─► S_prompt ─► E_θ ─► s_0 ─► JEPA-планировщик (π-prior + MPPI в ℝ^{d_s}) ─► S_plan ─► CFM-декодер (K шагов ODE) ─► X_1 ─► argmax ─► tokens
          W_fast: O(d²)                       ĝ = s_0 + G(s_0)    мир: P_φ(s, a)                   DiT + cross-attn к плану
```

* **Нет next-token prediction, нет KV-cache.** Контекст любой длины сжимается в матрицу быстрых весов
  `W_fast ∈ ℝ^{d×d}` (4 KiB в пресете `tiny`), рассуждение идёт в непрерывном латентном пространстве,
  а все `L` выходных токенов материализуются параллельно за `K` шагов интегрирования ODE.
* **Два пути исполнения над одними параметрами.** *Graph path* — это candle + autograd, на нём идёт
  обучение, он работает на любом устройстве. *Kernel path* — упакованные bf16-веса, f32-аккумуляторы,
  AVX2/FMA/F16C, rayon и арена. Горячие циклы kernel path **не делают ни одной аллокации в куче**,
  это проверяет `tests/zero_alloc.rs` через считающий глобальный аллокатор.
* Всё детерминировано: собственный seeded RNG, у каждого MPPI-сэмпла свой поток `(seed, iter, m)`,
  поэтому результат бит-в-бит совпадает при любом числе потоков.

## Документация

Подробная документация лежит в [`docs/`](docs/README.md):

| Документ | О чём |
|---|---|
| [architecture.md](docs/architecture.md) | модули, поток данных, два пути исполнения, память, карта исходников |
| [math.md](docs/math.md) | все формулы: TTT, JEPA и VICReg, энергия и MPPI, CFM и векторное поле, функция потерь |
| [training.md](docs/training.md) | обучение, как читать логи и оценку, точность, чекпойнты, задачи |
| [api.md](docs/api.md) | использование как библиотеки: движок, стадии, компоненты, свои векторные поля |
| [cli.md](docs/cli.md) | команды `demo/train/infer/bench/serve`, протокол TCP-сервера |
| [configuration.md](docs/configuration.md) | все поля конфигурации и пресеты |
| [runtime.md](docs/runtime.md) | арена, «0 аллокаций», точность, SIMD, параллелизм, детерминизм, скорость |
| [development.md](docs/development.md) | тесты, инварианты, рецепты расширения, ограничения |

Примеры: `cargo run --release --example quickstart` (обучение → инференс) и
`cargo run --release --example staged` (API по стадиям).

## Результаты (`cargo run --release -- demo`, 4 ядра CPU, ~4 мин обучения)

Задача `sort`: словарь 10, промпт 8 токенов → 8 токенов ответа, пресет `tiny` (≈307K параметров),
1500 шагов AdamW, batch 64. Оценка на 128 отложенных примерах.

| Режим (1500 шагов) | Токены | Последовательность целиком | Энергия плана (тёплый старт) |
|---|---:|---:|---:|
| Оракульный план (верхняя граница) | 97.9% | 84.4% | — |
| **Движок: π + MPPI (по умолчанию)** | **96.8%** | **80.5%** | 0.0120 (0.0232) |
| абляция: π only | 95.2% | 71.9% | 0.0232 |
| абляция: MPPI (zero init) | 83.8% | 37.5% | 0.0128 (0.0366) |

Примеры (вывод движка на отложенных промптах):

```
✓ prompt [5 0 8 8 2 1 6 7] → target [0 1 2 5 6 7 8 8] | engine [0 1 2 5 6 7 8 8]
✓ prompt [5 1 5 5 4 2 0 8] → target [0 1 2 4 5 5 5 8] | engine [0 1 2 4 5 5 5 8]
✗ prompt [0 0 7 0 2 7 6 4] → target [0 0 0 2 4 6 7 7] | engine [0 0 2 2 4 6 7 7]
```

* **Оракульный план.** Декодер получает teacher-forced траекторию, построенную с доступом к ответу.
  Это верхняя граница для планировщика.
* **π + MPPI** — полный движок по умолчанию: MPPI стартует с роллаута обученной политики.
* **π only** — абляция: только роллаут политики, 0 итераций MPPI.
* **MPPI (zero init)** — абляция: MPPI стартует с нулевых действий, как в «чистом» MPPI.

Поиск траектории (MPPI) вдвое снижает энергию относительно тёплого старта и заметно повышает
точность последовательности. Без обученного prior MPPI в 32-мерном пространстве действий
справляется значительно хуже.

## Соответствие спецификации

| Раздел спецификации | Реализация |
|---|---|
| TTT-Encoder, `W_fast`, rank-1 online GD | `ttt/fast_weights.rs` — `FastWeightsState::step_update`: фьюзнутое ядро `W ← W − η(Wk − v)⊗k`, один проход по строкам, 0 аллокаций. Также батчевая дифференцируемая версия для обучения |
| Адаптивный η | `ttt/layer.rs` — `η_t = η·σ(w_η·x_t + b_η)`; `‖k‖ = 1`, поэтому шаг устойчив (выпуклая комбинация вдоль `k`) |
| `z_t = W_t q_t`, `S_0 = MLP(W^{(N)})` | `ttt/mod.rs` — `S_prompt = MLP(LN([vec(W_fast·P); mean_t z_t]))` с обучаемыми пробами `P`. Потоковый API `absorb/finish` принимает контекст неограниченной длины |
| `E_θ`, `Ē_θ` (EMA) | `jepa/mod.rs` — `Jepa::ema`: `θ̄ ← τθ̄ + (1−τ)θ` после каждого шага оптимизатора |
| World model `P_φ(s, a)` | `jepa/world_model.rs` — `ŝ' = s + MLP([s; a])`: graph-версия и упакованная для роллаутов |
| VICReg | `jepa/vicreg.rs` — invariance / variance / covariance (off-diagonal через `ΣC² − Σdiag²`) |
| MPPI | `jepa/planner.rs` — `JEPAPlanner::plan_into`: M параллельных роллаутов (rayon), `w = softmax(−(E−E_min)/λ)`, нормированная температура, затухание шума, удержание лучшего сэмпла; 0 аллокаций |
| Latent GD | `jepa/planner.rs` — `GradientPlanner::refine`: Adam по `a = tanh(u)` через дифференцируемую world model (режим `--planner mppi+gd`) |
| CFM, OT-путь, σ_min | `flow/mod.rs` — `ot_path`: `X_t = (1−(1−σ)t)X_0 + tX_1`, `u_t = X_1 − (1−σ)X_0` |
| `v_θ(X_t, t, S*)`: cross-attn + AdaLN | `flow/vector_field.rs` — DiT-блоки: adaLN-Zero (self-attn, MLP) + cross-attn к плану. На kernel path K/V плана кэшируются один раз на траекторию |
| Euler / Midpoint / Heun | `flow/ode_solver.rs` — `FlowMatchingSampler::integrate`: буферы из арены, обновления in-place, 0 аллокаций |
| Unembed + argmax | `flow/mod.rs` — `PackedHead::decode_into`: все `L` позиций за один проход |
| Arena / Zero-alloc | `arena.rs` (буферы `Box<[f32]>` фиксированного размера и тензоры с собственным storage) + `kernels/inplace.rs` (candle `InplaceOp{1,2,3}`: ядра пишут прямо в storage тензора) |
| NewType-состояния | `types.rs` — `PromptState`, `LatentState`, `LatentPlan`, `FlowState` с проверкой формы при создании |
| unsafe только для SIMD | `kernels/simd.rs` — единственный файл с `unsafe`: AVX2/FMA/F16C-интринсики за безопасными обёртками с runtime-детекцией |
| rayon / tokio | rayon: MPPI-роллауты и крупные ядра. tokio: `cog_engine serve` (TCP, инференс в `spawn_blocking`) |
| Пайплайн | `pipeline.rs` — `CognitiveEngine::generate_into`: Tokens → TTT → JEPA → CFM → Tokens |

### Сверх спецификации (и зачем)

* **Модель латентных действий** `a_t = tanh(A_ψ(ŝ_t, s̄_{t+1}))`. У спецификации нет источника
  действий `a_t` для обучения `P_φ`. Эта модель выводит «мыслительный шаг» из пары соседних целевых
  состояний и используется только при обучении.
* **Откуда берутся целевые состояния.** `s̄_h = Ē(S_h)`, где `S_h` — readout TTT после промпта **и
  первых `h` чанков ответа**. Траектория рассуждения — это путь в латентном пространстве от «вопроса»
  к «вопросу + ответу».
* **Goal head** `ĝ = s_0 + G(s_0) ≈ s̄_H` — это `energy_target` планировщика при инференсе. Энергия:
  `E = ‖s_H − ĝ‖²/d_s + λ_a‖A‖²/(H·d_a)`.
* **Policy prior** `π(s, ĝ)`. Клонирует действия модели латентных действий, MPPI стартует с её
  роллаута (амортизированный System 2, как в TD-MPC). Абляция в таблице выше показывает, что это
  главный источник качества планировщика.
* **Вспомогательные потери декодера.** CE по одношаговой оценке `x̂_1 = X_t + (1−t)v_θ` и CE головы на
  зашумлённых чистых эмбеддингах. Цели `X_1` — замороженная случайная таблица эмбеддингов (обучаемые
  цели в CFM склонны к коллапсу).

### Отклонения от спецификации (честно)

* `VectorFieldEstimator` принимает `&mut self` и пишет результат в `out: &mut Tensor`
  (`estimate_velocity_into`), а также имеет `prepare(plan)`. Иначе оценщик не может переиспользовать
  предвыделенный workspace без interior mutability. Аллоцирующий `estimate_velocity` оставлен как
  метод по умолчанию. Соответственно, `FlowMatchingSampler` держит `&mut dyn VectorFieldEstimator`.
* Поле `LatentPlan::trajectory` приватное, доступ через `trajectory()`. Иначе проверку формы
  можно обойти.
* `JEPAPlanner` держит упакованную residual-MLP `PackedWorldModel` вместо `Linear`. Метод
  `plan(initial_state, energy_target, device)` с сигнатурой из спецификации есть (аллоцирующий).
  Горячий путь — `plan_into`.
* **Точность при обучении.** Мастер-веса и моменты AdamW хранятся в f32. Compute dtype
  настраивается (`--compute`). На CPU по умолчанию f32: в candle 0.11 нет CPU-matmul для bf16,
  а f16 работает, но в 2.4 раза медленнее. На CUDA по умолчанию bf16. Норма, softmax, VICReg и CE
  всегда считаются в f32. Инференс: веса в bf16 (или f16/f32), аккумуляторы и шаги интегратора в f32.
* **CUDA.** Фича `cuda` включает candle-CUDA для graph path (обучение, `--device cuda`). Ядра
  с нулевыми аллокациями — только host/CPU (MPPI по спецификации и так выполняется на host).
  На GPU in-place-хелперы откатываются на graph-операции. **В этой среде нет GPU, и сборка с
  `--features cuda` не проверялась.**
* Реализован TTT-**Linear**; вариант TTT-MLP не реализован.
* **Строгий 0 аллокаций при включённом параллелизме.** Если вызывать движок из потока вне пула
  rayon, rayon кладёт задачу в глобальную очередь-инжектор, и та изредка аллоцирует новый блок
  (амортизированно). Чтобы гарантия была строгой и при rayon, вызывайте из пула (`pool.install`),
  как в тесте. Последовательные ядра (`kernels::set_parallel(false)`) не аллоцируют никогда.

## Производительность kernel path (пресет `tiny`, 4 ядра)

| Стадия | Латентность | Что внутри |
|---|---:|---|
| TTT encode | 39 µs | 8 токенов × (эмбеддинг, 3 проекции, фьюзнутый rank-1 шаг) |
| JEPA plan | 3.8 ms | π-роллаут + MPPI 128 сэмплов × 8 итераций × H=4 (rayon) |
| CFM decode | 5.0 ms | 16 шагов Heun = 32 оценки DiT, все 8 позиций параллельно |
| **Итого** | **8.8 ms** | **113 запросов/с**, веса 589 KiB (bf16), арена 229 KiB, `W_fast` = 4 KiB |

Что сделано для скорости:
* блочное ядро `dot4`: одна загрузка `x` питает 4 FMA, что дало 4.4 → 10 GMAC/s на bf16-весах;
* порог распараллеливания 1M MAC: на меньших ядрах пробуждение пула rayon стоит дороже работы.
  Вместе с `dot4` это сократило одно вычисление векторного поля с 855 до 207 µs;
* rational-minimax `fast_tanh` без вызова libm (ошибка < 5e-7): libm `tanhf` съедал 39% времени MPPI;
* кэш cross-attention K/V плана на всю траекторию ODE.

## Использование

```bash
git clone https://github.com/Artem101/AiAgent.git && cd AiAgent
cargo run --release -- demo                         # обучить (~4 мин), оценить, замерить
cargo run --release -- train --task sort --steps 1500 --out model.safetensors
cargo run --release -- infer --ckpt model.safetensors --prompt "3 1 4 1 5 9 2 6" [--planner mppi+gd] [--solver heun --ode-steps 16]
cargo run --release -- bench --ckpt model.safetensors
cargo run --release -- serve --ckpt model.safetensors --addr 127.0.0.1:7878   # одна строка токенов → одна строка ответа
```

Задачи: `sort`, `reverse`, `copy`. Пресеты: `tiny`, `small`. Точность: `--compute auto|f32|f16|bf16`.
Устройство обучения: `--device cpu|cuda`.

Как библиотека:

```rust
let model = CogModel::new(EngineConfig::tiny(10, 8, 8), &Device::Cpu)?;
let mut trainer = Trainer::new(model, TrainConfig::quick(Task::Sort))?;
trainer.run(|l| println!("{l}"))?;
let mut engine = CognitiveEngine::from_model(&trainer.model)?;   // pack + preallocate
let mut out = [0u32; 8];
let gen = engine.generate_into(&[3, 1, 4, 1, 5, 9, 2, 6], /*seed*/ 0, &mut out)?; // 0 allocations
```

## Тесты

`cargo test` — 26 тестов:
* **паритет graph ↔ kernel**: TTT-энкодер, векторное поле DiT (f32 и bf16), роллаут world model,
  фьюзнутый rank-1 шаг против эталонного кода из спецификации, `dot/dot4` SIMD против portable-версии;
* **математика**: порядок сходимости Euler/Midpoint/Heun, концы OT-пути, VICReg штрафует коллапс
  и корреляцию, EMA, `fast_tanh`;
* **MPPI**: снижает энергию, бит-в-бит детерминирован при параллельном и последовательном
  исполнении, latent GD улучшает решение;
* **`tests/zero_alloc.rs`**: 0 аллокаций в `step_update` (10 000 шагов), `FlowMatchingSampler`,
  `plan_into` и полном `generate_into` — последовательно и под rayon;
* **`tests/pipeline.rs`**: обучение снижает loss, чекпойнт восстанавливается бит-в-бит,
  генерация детерминирована, память контекста не растёт на промпте из 500 токенов.

## Структура

```
AiAgent/                         # корень репозитория = крейт cog_engine
├── Cargo.toml
├── src/
│   ├── lib.rs, main.rs          # библиотека и CLI (demo/train/infer/bench/serve)
│   ├── config.rs                # TTT/JEPA/Planner/Flow/Train конфиги, пресеты
│   ├── types.rs                 # NewType: PromptState, LatentState, LatentPlan, FlowState
│   ├── arena.rs                 # предвыделенные буферы и тензоры
│   ├── kernels/{mod,simd,inplace,rng}.rs   # host-ядра, SIMD, in-place candle ops, RNG
│   ├── nn.rs                    # ParamStore, Lin/Mlp, LN, MHA (graph path)
│   ├── ttt/{mod,fast_weights,layer}.rs
│   ├── jepa/{mod,world_model,vicreg,planner}.rs
│   ├── flow/{mod,vector_field,ode_solver}.rs
│   ├── model.rs                 # совместная функция потерь, save/load
│   ├── train.rs                 # AdamW, warmup+cosine, clip, EMA, оценка с абляциями
│   ├── data.rs                  # синтетические задачи
│   └── pipeline.rs              # CognitiveEngine: Tokens → TTT → JEPA → CFM → Tokens
└── tests/{zero_alloc,pipeline}.rs
```

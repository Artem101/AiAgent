# Использование как библиотеки

Все функции возвращают `candle_core::Result<T>`. Главные типы реэкспортированы из корня крейта:

```rust
use cog_engine::{CogModel, CognitiveEngine, EngineConfig, TrainConfig};
use cog_engine::{PromptState, LatentState, LatentPlan, FlowState};
```

Запускаемые примеры: `examples/quickstart.rs` (обучение → инференс) и `examples/staged.rs`
(API по стадиям). Все фрагменты ниже взяты из них или повторяют их.

## Уровни API

| Уровень | Типы | Когда нужен |
|---|---|---|
| Высокий | `CogModel`, `Trainer`, `CognitiveEngine::generate` | обучить и отвечать на запросы |
| Стадии | `CognitiveEngine::{encode, think, decode}` | посмотреть на план, подменить стадию |
| Компоненты | `FastWeightsState`, `JEPAPlanner`, `FlowMatchingSampler`, `VectorFieldEstimator` | исследования, свои модули |
| Ядра | `kernels::*`, `kernels::inplace::*`, `Arena` | писать свои zero-alloc компоненты |

## Модель и обучение

```rust
use candle_core::Device;
use cog_engine::{data::Task, train::Trainer, CogModel, EngineConfig, TrainConfig};

let cfg = EngineConfig::tiny(/*vocab*/ 10, /*prompt len*/ 8, /*answer len*/ 8);
let model = CogModel::new(cfg, &Device::Cpu)?;          // проверяет cfg.validate()
let mut tc = TrainConfig::quick(Task::Sort);
tc.steps = 1500;
let mut trainer = Trainer::new(model, tc)?;
let last_eval = trainer.run(|line| println!("{line}"))?; // Option<EvalReport>
trainer.model.save("model.safetensors")?;
```

Загрузка: создайте модель с **той же** конфигурацией, затем `model.load(path)?`.

Полезные методы `CogModel`:

| Метод | Что делает |
|---|---|
| `loss(&batch, &mut rng, &tc) -> (Tensor, LossReport)` | совместная функция потерь (дифференцируемая) и отчёт по слагаемым |
| `teacher_plan(&batch, dtype)` | оракульная траектория `[B, H+1, d_s]` |
| `decode_graph(&plan, &mut rng)` | батчевый декодинг плана на graph path → `Vec<Vec<u32>>` |
| `ema_update()` | шаг EMA целевого энкодера |
| `trainable_vars()`, `num_params()` | параметры для оптимизатора |
| `save(path)` / `load(path)` | чекпойнт (см. [training.md](training.md#чекпойнты)) |

## Инференс: `CognitiveEngine`

```rust
let mut engine = CognitiveEngine::from_model(&model)?;   // упаковка весов + все буферы

// Без аллокаций: выход в буфер вызывающего, out.len() == answer_len
let mut out = [0u32; 8];
let gen = engine.generate_into(&[3, 1, 4, 1, 5, 9, 2, 6], /*seed*/ 0, &mut out)?;
println!("{out:?}, энергия плана {:.4}, {:?}", gen.plan.energy, gen.timings.total());

// Удобная версия: возвращает Vec (одна аллокация на результат)
let (tokens, gen) = engine.generate(&prompt, 0)?;
```

`Generation` содержит `plan: PlanStats` (энергия, энергия тёплого старта, ошибка терминального
состояния, ESS), `timings: StageTimings` (encode / plan / decode) и `refined` (улучшил ли план
latent GD).

Один и тот же `(prompt, seed)` всегда даёт один и тот же ответ. Разные `seed` дают разный шум
MPPI и разный `X_0`.

### Настройка на лету

| Метод | Эффект |
|---|---|
| `set_planner(PlannerKind::MppiThenGradient)` | включить доводку плана градиентным спуском (аллоцирует; в `tiny` ≈ +30 ms на запрос) |
| `set_mppi_iterations(n)` | `0` — план = роллаут политики, без поиска |
| `set_policy_prior(false)` | MPPI стартует с нулевых действий |
| `set_solver(SolverKind::Euler, 8)` | решатель и число шагов ODE (NFE = steps × evals_per_step) |
| `memory()` | `MemoryReport`: веса, арена, размер `W_fast` |
| `last_plan()`, `last_flow_state()` | план и `X_1` последнего запроса (копии, для отладки) |

Методы движка принимают `&mut self`: у него один набор буферов на один запрос. Для параллельного
обслуживания создайте по движку на поток (`from_model` упаковывает собственную копию весов) или
сериализуйте доступ через `Mutex`, как это делает `cog_engine serve`.

### По стадиям

```rust
let s_prompt: PromptState = engine.encode(&prompt)?;             // стадия 1: TTT
let (plan, stats, refined) = engine.think(&s_prompt, seed)?;     // стадия 2: ĝ + π + MPPI
let mut out = [0u32; 8];
engine.decode(&plan, seed, &mut out)?;                            // стадия 3: ODE + argmax
```

`encode` принимает промпт любой длины, память контекста при этом не растёт. Возвращаемые
`PromptState` и `LatentPlan` ссылаются на буферы движка: следующий вызов той же стадии их
перезапишет. Если результат нужно сохранить, сделайте копию (`.tensor().copy()`).

## Компоненты

### Быстрые веса TTT

```rust
use cog_engine::ttt::FastWeightsState;

let mut fast = FastWeightsState::new(/*d*/ 4, /*η*/ 0.5, &Device::Cpu)?;
let k = Tensor::new(&[0.5f32, 0.5, 0.5, 0.5], &dev)?;      // ‖k‖ = 1
let v = Tensor::new(&[1f32, -1.0, 0.0, 2.0], &dev)?;
for _ in 0..10 { fast.step_update(&k, &v)?; }              // W ← W − η(Wk − v)⊗k, без аллокаций
let z = fast.forward(&k)?;                                  // ≈ v
```

Кроме этого есть `step_update_eta` (свой η на шаг), `forward_into` (в готовый буфер),
`step_update_host` и `forward_host` (на срезах `&[f32]`), `reset`, `reconstruction_loss`.
Для обучения есть батчевые дифференцируемые `ttt::fast_weights::{batched_step, batched_apply}`.

На уровне энкодера `PackedTttEncoder` даёт потоковый интерфейс: `reset(ws)`, затем
`absorb(token, segment, ws)` на каждый токен и `finish(ws) -> PromptState`.

### Планировщик

```rust
use cog_engine::jepa::JEPAPlanner;

let planner = JEPAPlanner::new(model.jepa.world.pack(DType::BF16)?, cfg.jepa.horizon, &cfg.planner)
    .with_policy(model.jepa.policy.pack(DType::BF16)?);     // необязательно
let mut arena = Arena::new(&Device::Cpu);
let mut ws = planner.workspace(&mut arena)?;                // все буферы MPPI
let stats = planner.plan_into(&s0, &goal, &mut ws, /*seed*/ 42)?;  // без аллокаций
let plan_tensor = &ws.plan;                                 // [H+1, d_s]
let actions = ws.nominal_actions();                         // [H·d_a]
```

`plan(initial_state, energy_target, device) -> LatentPlan` — аллоцирующая версия с сигнатурой
из спецификации. Энергия фиксирована (`JEPAPlanner::energy`): терминальное расстояние до цели
плюс штраф на действия. Чтобы задать свою, измените этот метод (см. [development.md](development.md)).

Latent GD отдельно: `GradientPlanner { steps, lr, action_cost }.refine(&world_model_graph, &s0, &goal, &init_actions)`.

### ODE-сэмплер и собственное векторное поле

Любой тип, реализующий `VectorFieldEstimator`, можно интегрировать:

```rust
use cog_engine::flow::{FlowMatchingSampler, ODESolverConfig, SamplerBuffers, SolverKind, VectorFieldEstimator};
use cog_engine::kernels::inplace::{host_read, host_write};
use cog_engine::kernels::rng::Rng;

/// v(x, t) = c − x
struct Relax { c: f32 }

impl VectorFieldEstimator for Relax {
    fn estimate_velocity_into(&mut self, x: &Tensor, _t: f32, _plan: &LatentPlan, out: &mut Tensor) -> Result<()> {
        let c = self.c;
        host_read(x, |xs| host_write(out, |o| {
            for (o, &x) in o.iter_mut().zip(xs) { *o = c - x; }
            Ok(())
        }))?
    }
}

let mut field = Relax { c: 2.0 };
let mut bufs = SamplerBuffers::new(&mut arena, /*L*/ 2, /*d*/ 3)?;
let mut sampler = FlowMatchingSampler::new(
    &mut field,
    ODESolverConfig { steps: 8, sigma_min: 0.0, solver: SolverKind::Heun },
);
sampler.sample_into(&plan, &mut bufs, &mut Rng::new(0))?;  // X_0 ~ N(0, I) → X_1 в bufs.x
```

Контракт трейта:
* `prepare(&mut self, plan)` вызывается один раз перед интегрированием. Здесь кэшируется всё, что
  не зависит от `t`. По умолчанию ничего не делает.
* `estimate_velocity_into` должна записать скорость в `out` той же формы, что `x`. Реализация на
  CPU не должна аллоцировать, иначе сэмплер потеряет гарантию «0 аллокаций».
* `estimate_velocity` — аллоцирующая обёртка по умолчанию.

Готовые реализации: `PackedVectorField` (ядра, свой workspace) и `GraphVectorField` (candle,
любое устройство, аллоцирует).

`sample(shape, plan, device, seed) -> FlowState` — аллоцирующая версия, сама создаёт буферы.

## Состояния (NewType)

| Тип | Проверяемая форма | Конструктор |
|---|---|---|
| `PromptState` | `[d_ctx]` | `PromptState::new(tensor, d_ctx)` |
| `LatentState` | `[d_s]` | `LatentState::new(tensor, d_s)` |
| `LatentPlan` | `[H+1, d_s]` | `LatentPlan::new(tensor, plan_len, d_s)`; `initial_state()`, `terminal_state()` |
| `FlowState` | `[L, d_token]` | `FlowState::new(tensor, L, d_token)` |

Поля приватные, тензор доступен через `tensor()` / `trajectory()` или `into_inner()`. Тензор
неверной формы в конструкторе даёт `Err`.

## Конфигурация в коде

```rust
let mut cfg = EngineConfig::tiny(16, 12, 12);
cfg.planner.num_samples = 256;
cfg.flow.solver = ODESolverConfig { steps: 8, sigma_min: 1e-4, solver: SolverKind::Midpoint };
cfg.weight_dtype = DType::F16;
cfg.validate()?;
```

Все поля описаны в [configuration.md](configuration.md).

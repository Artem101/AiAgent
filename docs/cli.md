# Командная строка

```bash
cargo build --release
./target/release/cog_engine <команда> [--флаг значение]…
```

Флаги пишутся в форме `--имя значение`. Без команды выполняется `demo`.

## Команды

### `demo`

Обучает модель, печатает оценки и примеры, затем замеряет латентность и память. Чекпойнт
сохраняется, только если передан `--out`.

```bash
cog_engine demo [--task sort] [--steps 1500] [флаги обучения]
```

### `train`

```bash
cog_engine train --out model.safetensors [--task sort|reverse|copy|browser] [--steps 1500] [--preset tiny|small]
                 [--vocab 10] [--len 8] [--batch 64] [--lr 2e-3] [--seed 7] [--eval-every 500]
                 [--device cpu|cuda] [--compute auto|f32|f16|bf16]
```

| Флаг | По умолчанию | Смысл |
|---|---|---|
| `--out` | `model.safetensors` | путь чекпойнта; рядом пишется `<out>.cfg`. Пустая строка — не сохранять |
| `--task` | `sort` | задача (см. [training.md](training.md#задачи)) |
| `--steps` | 1500 | шагов оптимизатора |
| `--preset` | `tiny` | размеры модели (см. [configuration.md](configuration.md)) |
| `--vocab` | 10 | размер словаря |
| `--len` | 8 | длина промпта и ответа (`N = L`); должна делиться на горизонт (4) |
| `--batch` | 64 | размер батча |
| `--lr` | 2e-3 | пиковый learning rate |
| `--seed` | 7 | инициализация весов и поток данных |
| `--eval-every` | 500 | период оценки; 0 — без промежуточной оценки |
| `--device` | `cpu` | `cuda` требует сборки с `--features cuda` |
| `--compute` | `auto` | dtype прямого/обратного прохода; `auto` = f32 на CPU, bf16 на CUDA |
| `--conv` | 1 (`browser`: 4) | `ttt.conv_width` |
| `--readout-last` | 0 (`browser`: 3) | `ttt.readout_last` |
| `--probe` | 0 (`browser`: 1.0) | `jepa.probe_weight` |

Флаг без значения (`--headed`) можно ставить перед другим флагом.

### `infer`

```bash
cog_engine infer --ckpt model.safetensors --prompt "3 1 4 1 5 9 2 6" [--seed 0]
                 [--planner mppi|mppi+gd] [--solver heun|midpoint|euler] [--ode-steps 16]
```

Пример вывода (чекпойнт `tiny` после 1500 шагов):

```
prompt : 3 1 4 1 5 9 2 6
output : 1 1 2 3 4 5 6 9
target : 1 1 2 3 4 5 6 9   (sort)
plan   : energy 0.0061 (warm start 0.0482), ESS 4.7
latency: encode 149.13µs | plan 3.684257ms | decode 4.455594ms | total 8.288981ms
```

`target` — правильный ответ для задачи из `.cfg`, для сравнения. `ESS` — эффективное число
сэмплов на последней итерации MPPI: маленькое значение означает, что веса сосредоточены на
нескольких траекториях. `--planner mppi+gd` добавляет доводку градиентным спуском, строка `plan`
тогда заканчивается `refined by latent GD`, если это помогло. На том же запросе энергия
снижается с 0.0061 до 0.0060, а plan занимает ~33 ms вместо ~4 ms.

Промпт можно разделять пробелами или запятыми. Длина должна совпадать с `len` из `.cfg`.

### `bench`

```bash
cog_engine bench [--ckpt model.safetensors] [--iters 200] [--planner …] [--solver …] [--ode-steps …]
```

Без `--ckpt` замеряет необученную модель пресета (`--preset`, `--vocab`, `--len`). Для скорости и
памяти этого достаточно.

```
memory : weights 588.9 KiB (BF16) | arena 228.5 KiB in 58 buffers | context state W_fast 4096 B (independent of N)
latency: encode 33.2 µs | plan 3725.1 µs (MPPI 128×8×H=4) | decode 4871.0 µs (16 Heun steps, 32 NFE, L=8 in parallel) | 116 queries/s
```

### `serve`

```bash
cog_engine serve --ckpt model.safetensors [--addr 127.0.0.1:7878] [--planner …] [--solver …]
```

TCP-сервер на tokio с построчным протоколом.

Реальная сессия с тем же чекпойнтом:

| Запрос (одна строка) | Ответ (одна строка) |
|---|---|
| `3 1 4 1 5 9 2 6` | `1 1 3 3 4 5 6 9 \| 10.93032ms` |
| `1 2 3` (неверная длина) | `error: expected 8 tokens, got 3` |
| `0 0 0 0 1 1 1 99` (вне словаря) | `error: token id 99 out of range (vocab 10)` |
| `9,8,7,6,5,4,3,2` | `2 4 4 5 6 7 8 9 \| 9.552167ms` |

Оба ответа здесь неверны (правильные: `1 1 2 3 4 5 6 9` и `2 3 4 5 6 7 8 9`). Модель `tiny`
после 1500 шагов верна примерно в 80% ответов. Первый промпт с seed 0 в `infer` выше решён
правильно, а здесь seed равен 1.

* Каждое соединение обслуживается своей задачей tokio. Инференс выполняется в `spawn_blocking`,
  чтобы не блокировать реактор.
* Движок один, за `Mutex`: запросы выполняются по очереди, по ~10 ms каждый в `tiny`.
* Seed запроса — его порядковый номер в соединении (1, 2, …). Один и тот же промпт в разных
  строках может дать разные ответы, если модель не уверена.
* Ошибка всегда занимает одну строку: многострочные backtrace candle обрезаются.

Проверка из shell:

```bash
printf '3 1 4 1 5 9 2 6\n' | nc 127.0.0.1 7878
```

### `agent`

Задать вопрос агенту-браузеру (чекпойнт обучен с `--task browser`, см. [browser.md](browser.md)).

```bash
cog_engine agent --ckpt agent.safetensors --question "Сколько стоит лампа?" [--world 42]
                 [--browser chrome|sim] [--policy model|expert] [--max-steps 10]
                 [--headed] [--chrome /path/to/chrome] [--site-addr 127.0.0.1:0]
```

| Флаг | По умолчанию | Смысл |
|---|---|---|
| `--question` | `what is the price of the lamp?` | вопрос на русском или английском: товар + атрибут (`price`, `color`, `brand`, `rating`) |
| `--world` | 42 | номер мира песочницы (от него зависят все факты) |
| `--browser` | `chrome` | `chrome` — настоящий Chromium, `sim` — симулятор |
| `--policy` | `model` | `expert` — сценарий-учитель, `--ckpt` не нужен |
| `--max-steps` | 10 | лимит действий |
| `--headed` | нет | показать окно браузера (нужен дисплей) |
| `--chrome` | поиск | путь к браузеру; иначе `COG_CHROME`, Playwright, `PATH` |

Для каждого шага печатаются URL, что модель видит (токены наблюдения), что она делает и
сколько думала. В конце — ответ и сверка с фактом мира.

### `agent-eval`

```bash
cog_engine agent-eval --ckpt agent.safetensors [--episodes 100] [--browser sim|chrome]
                      [--policy model|expert] [--steps 1000] [--seed 1] [--max-steps 10]
```

Прогоняет случайные задачи (мир + вопрос) и печатает успешность, число неверных ответов и
эпизодов без ответа, среднее число шагов, долю ошибочных действий и время. `--steps N` добавляет
точность отдельных шагов по типам действий на `N` случайных состояниях.

### `site`

```bash
cog_engine site [--addr 127.0.0.1:8080]
```

Поднимает песочницу для обычного браузера: `http://127.0.0.1:8080/w/42/`.

## Файл `<ckpt>.cfg`

```
preset=tiny
vocab=10
len=8
task=sort
seed=7
conv=1
readout_last=0
probe=0
```

Этих полей достаточно, чтобы воссоздать `EngineConfig` через `EngineConfig::preset`. Поля
`conv`, `readout_last` и `probe` (`ttt.conv_width`, `ttt.readout_last`, `jepa.probe_weight`)
необязательны: в чекпойнтах, записанных до их появления, они принимают значения 1, 0 и 0. Если
архитектура менялась вручную в коде, а не через пресет, CLI её не восстановит. В этом случае
загружайте модель из кода с той же конфигурацией.

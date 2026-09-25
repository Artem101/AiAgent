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
cog_engine train --out model.safetensors [--task sort|reverse|copy|browser|text] [--steps 1500] [--preset tiny|small]
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
| `--readout-last` | 0 (`browser`, `text`: 4) | `ttt.readout_last` |
| `--pools` | 0 (`browser`, `text`: 4) | `ttt.readout_pools` |
| `--probe` | 0 (`browser`, `text`: 1.0) | `jepa.probe_weight` |
| `--horizon` | 4 (`browser`, `text`: 8) | `jepa.horizon` |
| `--copy` | 0 (`browser`, `text`: 32) | `jepa.copy_dim`, копирование из контекста; `0` — выключить |
| `--probe-states` | 0 (`browser`, `text`: 3) | мыслей в потере пробы за шаг (`s_0`, `s_H` и случайные промежуточные); `0` — все |
| `--probe-goal` | 0 | `1` — обучать пробу ещё и на цели `ĝ` |
| `--corpus` | — | текст для `--task text` или для примеси к `browser` (`scripts/fetch_ru_corpus.sh`) |
| `--valid` | — | отложенный текст: после обучения печатается точность продолжения против униграммной и биграммной базовых линий |
| `--text-mix` | 0.25 | доля текста в батчах `browser` |
| `--init` | — | начать с весов чекпойнта той же архитектуры (оптимизатор — с нуля) |
| `--start-step` | — | вместе с `--init <out>.step<N>`: продолжить прерванный прогон с шага `N` (расписание LR продолжается, поток данных пересеивается) |
| `--save-every` | 0 | сохранять `<out>.step<N>` каждые `N` шагов (для кривых сходимости) |
| `--log-every` | 100 | период строки лога |

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

### Флаги инференса (`infer`, `bench`, `serve`, `agent`, `agent-eval`, `complete`)

Без флагов используется конфигурация чекпойнта.

| Флаг | Смысл |
|---|---|
| `--planner mppi\|mppi+gd` | MPPI или MPPI + latent GD |
| `--tree <B>` | ширина луча латентного дерева гипотез; `0` — без дерева |
| `--iters <n>` | итераций MPPI; `0` — план из дерева или роллаута политики |
| `--solver`, `--ode-steps` | решатель ODE декодера |
| `--decoder flow\|probe\|probe-start\|probe-vote` | декодер действия (см. `decoder` в [configuration.md](configuration.md)) |

### `tokenizer`

```bash
cog_engine tokenizer --corpus data/ru/train.txt [--vocab 1024] [--browser-texts 20000] [--out models/tokenizer_ru.bpe]
```

Обучает байтовый BPE на корпусе и на текстах браузерной задачи (страницы, формулировки,
тексты действий и результаты калькулятора), каждый текст — с ведущим пробелом. Файл
`models/tokenizer_ru.bpe` компилируется в бинарник (`text::ru()`): после переобучения токенизатора
пересоберите проект и обучите модели заново.

### `complete`

```bash
cog_engine complete --ckpt model.safetensors --text "Москва — столица"
```

Продолжение текста: следующие 16 BPE-токенов (все сразу, неавторегрессионно) для чекпойнта
`text` или `browser` с корпусом.

### `agent`

Задать вопрос агенту-браузеру (чекпойнт обучен с `--task browser`, см. [browser.md](browser.md)).
Вопрос — свободный текст на русском: модель читает его как есть. Арифметику агент считает
калькулятором на Python ([calculator.md](calculator.md)).

```bash
cog_engine agent --ckpt agent.safetensors --question "Сколько будет 5+5?" [--world 42]
                 [--browser chrome|sim] [--policy model|expert] [--calc python|rust] [--max-steps 12]
                 [--headed] [--chrome /path/to/chrome] [--site-addr 127.0.0.1:0]
```

| Флаг | По умолчанию | Смысл |
|---|---|---|
| `--question` | `Сколько стоит лампа?` | вопрос на русском; если он совпадает с одним из шаблонов (с точностью до регистра, «ё» и знаков) или это арифметический пример, ответ проверяется |
| `--world` | 42 | номер мира песочницы (от него зависят все факты) |
| `--browser` | `chrome` | `chrome` — настоящий Chromium, `sim` — симулятор |
| `--policy` | `model` | `expert` — сценарий-учитель, `--ckpt` не нужен |
| `--calc` | `python` | исполнитель `CALC`: изолированный `python3` или его точное зеркало на Rust |
| `--max-steps` | 12 | лимит действий |
| `--headed` | нет | показать окно браузера (нужен дисплей) |
| `--allow-internet` | нет | разрешить браузеру внешние адреса (по умолчанию он видит только `localhost`/`127.0.0.1`) |
| `--chrome` | поиск | путь к браузеру; иначе `COG_CHROME`, Playwright, `PATH` |

Для каждого шага печатаются URL, что модель видит (токены наблюдения), как она рассуждала
(дерево гипотез по глубинам, выжившие гипотезы и цепочка мыслей, декодированные пробой), что
она делает и сколько думала, а для `CALC` — что вернул калькулятор (`tool : python: 5+5 = 10`).
В конце — ответ и сверка с фактом мира.

### `agent-eval`

```bash
cog_engine agent-eval --ckpt agent.safetensors [--episodes 100] [--browser sim|chrome]
                      [--policy model|expert] [--calc rust|python] [--split train|heldout|both] [--steps 1000]
                      [--seed 1] [--max-steps 12]
```

Прогоняет случайные задачи (мир + вопрос) и печатает успешность по семействам (поиск,
сравнение, фильтр, арифметика, суммы цен, реплики), число неверных ответов и эпизодов без
ответа, среднее число шагов, долю ошибочных действий и время. Калькулятор по умолчанию —
зеркало на Rust (быстро и детерминированно; результаты те же, что у Python). `--split`
выбирает обучающие или отложенные формулировки. `--steps N` добавляет точность отдельных шагов по типам действий на `N` случайных состояниях.

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
horizon=4
pools=0
copy=0
```

Этих полей достаточно, чтобы воссоздать `EngineConfig` через `EngineConfig::preset`. Поля
`conv`, `readout_last`, `probe`, `horizon`, `pools` и `copy` (`ttt.conv_width`, `ttt.readout_last`,
`jepa.probe_weight`, `jepa.horizon`, `ttt.readout_pools`, `jepa.copy_dim`) необязательны: в
чекпойнтах, записанных до их появления, они принимают значения 1, 0, 0, 4, 0 и 0. Для задач `browser` и `text` настройки
планировщика (дерево, latent GD) берутся из `browser::engine_config`. Если
архитектура менялась вручную в коде, а не через пресет, CLI её не восстановит. В этом случае
загружайте модель из кода с той же конфигурацией.

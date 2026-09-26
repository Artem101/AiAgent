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
| `--decoder flow\|probe\|probe-start\|probe-vote\|probe-consensus\|probe-first` | декодер действия (см. `decoder` в [configuration.md](configuration.md)); у агента по умолчанию `probe-first` |

### `tokenizer`

```bash
cog_engine tokenizer --corpus data/ru20k/tokenizer_corpus.txt --vocab 8192 [--browser-texts 30000] [--out models/tokenizer_ru.bpe]
```

Обучает байтовый BPE на корпусе и на текстах браузерной задачи (страницы, формулировки,
тексты действий и результаты калькулятора), каждый текст — с ведущим пробелом. Поставляемый
токенизатор (8192 токена) обучен на `data/ru20k/tokenizer_corpus.txt` — выборке «ru20k» и UD
([unified.md](unified.md)); по умолчанию `--corpus` — `data/ru/train.txt`, `--vocab` — 1024. Файл
`models/tokenizer_ru.bpe` компилируется в бинарник (`text::ru()`): после переобучения токенизатора
пересоберите проект и обучите модели заново.

### `complete`

```bash
cog_engine complete --ckpt model.safetensors --text "Москва — столица"
```

Продолжение текста: следующие 16 BPE-токенов (все сразу, неавторегрессионно) для чекпойнта
`text` или `browser` с корпусом.

### `train-unified`

Обучает единую модель — агента, который и разговаривает, и ищет, и считает ([unified.md](unified.md)).

```bash
cog_engine train-unified [--preset base|tiny] [--steps 20000] [--batch 32] [--lr 1e-3] [--min-lr 5e-5] [--warmup 300]
                         [--out models/agent.safetensors] [--data data/ru20k|builtin] [--ud data/ru|none]
                         [--save-every 500] [--eval-every N] [--log-every 50] [--seed 7]
                         [--init ckpt [--start-step N]] [--d-model 256] [--layers 4] [--heads 4] [--copy 32] [--teacher-plan 0.3]
```

Данные — сборка «ru20k» (`scripts/fetch_ru20k.sh`, `scripts/build_ru20k.py`) и UD
(`scripts/fetch_ru_corpus.sh`); `--data builtin` — несколько встроенных примеров для проверки
без скачивания. Батч делится между ядрами (до 4 потоков). Лог каждые `--log-every` шагов —
потери и точность самого вероятного токена по источникам:

```
step   2000 | loss 3.412 | nll 2.804 ptr 0.301 jepa 0.307 goal 0.003 | browser 0.08 (97%) dialogue 3.93 (37%) grammar 1.94 (63%) text 5.61 (22%) | |g| 1.02 | lr 9.62e-4 | 0.47 step/s
```

Каждые `--save-every` шагов пишется чекпойнт и `<ckpt>.cfg` (`kind=unified`), каждые
`--eval-every` — потери на отложенных данных и несколько ответов.

### `chat`

Разговор с единой моделью. Каждая реплика — эпизод в браузере: модель может искать,
считать или сразу ответить; предыдущие реплики входят в наблюдение.

```bash
cog_engine chat [--ckpt models/agent.safetensors] [--say "Привет!|Сколько стоит лампа?|Спасибо!"] [--think]
                [--browser sim|chrome] [--calc python|rust] [--temperature 0.7] [--top-k 40] [--search 1] [--world 42] [--max-steps 12]
```

| Флаг | По умолчанию | Смысл |
|---|---|---|
| `--say` | нет | реплики через `\|` вместо чтения из stdin |
| `--think` | нет | печатать рассуждение: дерево гипотез и что сказала бы каждая выжившая гипотеза |
| `--temperature`, `--top-k` | 0.7, 40 | сэмплирование текста реплики (глагол и роль — всегда жадно); `0` — жадно |
| `--search` | 1 | `0` — без латентного поиска (жадная цепочка мыслей) |
| `--browser` | `sim` | `chrome` — настоящий Chromium |

### `unified-eval`

```bash
cog_engine unified-eval [--ckpt models/agent.safetensors] [--n 256] [--grammar 300] [--dialogs 8] [--episodes 200]
                        [--browser sim|chrome] [--search 1] [--temperature 0]
```

Печатает:

- потери на отложенных данных по источникам (`--n` примеров на источник);
- точность ответов на грамматические вопросы (`--grammar` вопросов об отложенных и об обучающих
  леммах, по видам);
- ответы на фиксированный набор реплик;
- ответы на отложенные фрагменты диалогов рядом с настоящими;
- успешность агента на `--episodes` задачах с обучающими и с отложенными формулировками.

### `agent`

Задать вопрос агенту. Команда принимает и единую модель (по умолчанию —
`models/agent.safetensors`), и чекпойнт агента-браузера, обученный с `--task browser` (см.
[browser.md](browser.md)).
Вопрос — свободный текст на русском: модель читает его как есть. Арифметику агент считает
калькулятором на Python ([calculator.md](calculator.md)).

```bash
cog_engine agent --ckpt agent.safetensors --question "Сколько будет 5+5?" [--world 42]
                 [--browser chrome|sim] [--policy model|expert] [--calc python|rust] [--max-steps 12]
                 [--headed] [--chrome /path/to/chrome] [--site-addr 127.0.0.1:0]
```

| Флаг | По умолчанию | Смысл |
|---|---|---|
| `--ckpt` | `models/agent.safetensors` | чекпойнт агента (по умолчанию — поставляемая единая модель) |
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

У единой модели (`train-unified`) свой формат:

```
kind=unified
preset=base
seed=7
d_model=256
layers=4
heads=4
mlp=4
copy=32
teacher_plan=0.3
plan_noise=0.05
```

Остальное — TTT, JEPA, планировщик, размеры наблюдения и действия — берётся из пресета
(`UnifiedConfig::preset`).

Этих полей достаточно, чтобы воссоздать `EngineConfig` через `EngineConfig::preset`. Поля
`conv`, `readout_last`, `probe`, `horizon`, `pools` и `copy` (`ttt.conv_width`, `ttt.readout_last`,
`jepa.probe_weight`, `jepa.horizon`, `ttt.readout_pools`, `jepa.copy_dim`) необязательны: в
чекпойнтах, записанных до их появления, они принимают значения 1, 0, 0, 4, 0 и 0. Для задач `browser` и `text` настройки
планировщика (дерево, latent GD) берутся из `browser::engine_config`. Если
архитектура менялась вручную в коде, а не через пресет, CLI её не восстановит. В этом случае
загружайте модель из кода с той же конфигурацией.

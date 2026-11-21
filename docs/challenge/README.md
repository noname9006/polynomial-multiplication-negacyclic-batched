# Условия челленджа FHERMA (выжимка, снято с сайта 21.11.2026)

Первоисточники (читать их, если выжимка расходится):
- Челлендж: https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025
  - правила: `?tab=rules`, лидерборд: `?tab=leaderboard`, обсуждения: `?tab=discussions`
- Спецификация (решает рейтинг): https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic-batched
- Исходная спецификация: https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic

Файлы: `SPEC.md` — что считать; `RULES.md` — правила и замер; `LEADERBOARD.md` — соперники.
Закрытие: **29.11.2026 00:00**. Приз $5000 за самую быструю корректную реализацию.

## Обзор челленджа (вкладка Overview)
- Название: «Build the Fastest GPU-Accelerated Polynomial Multiplication with NVIDIA cuPQC», тип OPTIMIZATION, партнёр NVIDIA. cuPQC не обязателен — алгоритм свободный.
- Рейтинг решает **только batched-точка** B=128, N=65536, W=1747 по `negacyclic-batched@1.0.0` (с 11.11.2026). Реализации без batched-результата ставятся ниже, по точке #1.
- Точки челленджа:

| точка | спецификация | B | N | W | L | сидов | роль |
|---|---|---|---|---|---|---|---|
| batched | negacyclic-batched@1.0.0 | 128 | 65536 | 1747 | 55 | 3 | **решает место** |
| #1 | negacyclic@1.0.0 | — | 65536 | 1747 | 55 | 20 | показ; место без batched |
| #2 | negacyclic@1.0.0 | — | 32768 | 868 | 28 | 20 | показ |

- W — потолок безопасности 128 бит для данного N (HE Security Guidelines, табл. 5.2).
- Машина замера: `fherma-gpu-rtx6000` — Xeon 6776P, 24 ядра, 214 ГБ, NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SM, 97887 MiB).
- Один участник — одна строка лидерборда (лучшая из его реализаций). Можно подавать много версий. Код можно не публиковать.
- Как участвовать: реализация спецификации batched → объявить покрытие точки → Submit; FHERMA сама прогоняет и измеряет.

## Интерфейс харнесса (наш `solve.cu`)
- `fherma_init(point)` — не замеряется, получает только N, W, L, q.
- `fherma_h2d` (опц.) — только копия a, b на GPU, ≤ 0.45 с.
- `fherma_run` — **замеряется**; возвращается после синхронизации.
- `fherma_d2h` (опц.) — только копия c на хост, ≤ 0.25 с.
- `main.cpp`, `fherma.h`, `CMakeLists.txt` не менять — их подменяет платформа.

## История изменений условий
- 28.09.2026 — добавлена точка N=65536, W=1747, место решает она.
- 11.11.2026 — добавлена batched-точка, место решает она.
- 13.11.2026 — разрешены `fherma_h2d`/`fherma_d2h` вне замера; ошибочные batched-прогоны по single-спецификации сняты.
- 15.11.2026 — лимиты h2d 0.45 с / d2h 0.25 с.

## Обсуждения (Discussions)
- Batched-точка решает рейтинг; был ошибочный результат 894 µs по небатчевой спецификации, его убрали.
- h2d/d2h вне тайминга — разрешено (у нас сделано). Технических подсказок от организаторов нет.

## Наш статус
- Реализация `evgeniy-kepa/exact-rns-ntt-batched-negacyclic-multiplication`, UNDER REVIEW, в челлендж не подана (подачу делает только владелец).
- Лучший результат 30.9 мс (master `31b7eb8`). Цель — 10 мс.

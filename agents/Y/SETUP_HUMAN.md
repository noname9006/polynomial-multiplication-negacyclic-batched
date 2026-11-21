# Как поднять агента Y (инструкция для человека)

Агент Y работает на отдельной машине, но в том же GitHub-репозитории и в том же аккаунте FHERMA.
Второй аккаунт FHERMA заводить нельзя: правила челленджа запрещают несколько аккаунтов ради обхода ограничений, это грозит дисквалификацией.

## 1. Что нужно на машине
- git, Python 3.10+, pipx;
- агент (opencode или Claude Code);
- GPU не нужен: код проверяется прогонами на FHERMA;
- для локальной проверки синтаксиса — g++ (C++17) и sh: `sh tools/nvsyntax/check.sh`.

## 2. Доступ к GitHub (fine-grained токен)
1. GitHub (аккаунт владельца репозитория) → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token.
2. Срок — 30 дней. Repository access: Only select repositories → `polynomial-multiplication-negacyclic-batched`.
3. Permissions → Repository permissions → Contents: Read and write. Больше ничего.
4. На машине агента:
   ```
   git config --global credential.helper manager
   git clone https://github.com/noname9006/polynomial-multiplication-negacyclic-batched.git
   ```
   При первом `git push` логин — владелец репозитория, пароль — токен. Git Credential Manager его запомнит.
5. Рекомендуется защитить `master`: Settings → Branches → правило для `master`, запись только владельцу. В master вливает только основной агент.

## 3. Доступ к FHERMA (токен CLI)
1. На сайте FHERMA: Command line → Issue a token (показывается один раз, действует 90 дней).
2. На машине агента:
   ```
   pipx install fherma
   ```
3. Положите токен в окружение **сами**, не через чат с агентом. В PowerShell (ввод скрыт, в историю не попадает):
   ```
   $t = Read-Host "FHERMA token" -AsSecureString; [Environment]::SetEnvironmentVariable("FHERMA_TOKEN", [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($t)), "User")
   ```
   Либо выполните `fherma auth login` и вставьте токен в запрос CLI.
4. Перезапустите терминал и агента, чтобы он увидел переменную.
5. Проверьте сами: `fherma --help` и любая команда, читающая аккаунт.

Токен не вписывать в файлы репозитория, задания, `.env`, аргументы команд и чат. После челленджа отозвать оба токена.

## 4. Запуск агента
Откройте агента в папке клона и дайте ему первым сообщением:

> Прочитай `agents/Y/README_AGENT_Y.md` и работай по нему. Текущее задание — последний файл `agents/Y/TASK_*.md`.

## 5. Связь с основным агентом
- Задания основной агент кладёт в `agents/Y/TASK_*.md` в ветке `master`.
- Y работает в ветках `agent-y/<тема>` и пишет отчёты в `agents/Y/REPORT.md` в своей ветке.
- Чтобы основной агент узнал о сдаче, сообщите ему: «Y сдал, ветка agent-y/<тема>».

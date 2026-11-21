# Как поднять агента Y (инструкция для человека)

Агент Y работает на отдельной машине, но в том же GitHub-репозитории и в том же аккаунте FHERMA.
Второй аккаунт FHERMA заводить нельзя: правила челленджа запрещают несколько аккаунтов ради обхода ограничений, это грозит дисквалификацией.

## 1. Что нужно на машине
- git, Python 3.10+, pipx;
- агент (opencode или Claude Code);
- GPU не нужен: код проверяется прогонами на FHERMA;
- для локальной проверки синтаксиса — g++ (C++17) и sh: `sh tools/nvsyntax/check.sh`.

## 2. Доступ к GitHub (deploy key — ключ только для этого репозитория)
Deploy key — SSH-ключ, привязанный к одному репозиторию, а не к аккаунту. Токен аккаунта агенту не нужен.

Доступ к GitHub есть только у владельца репозитория. Поэтому шаги делятся: **[человек]** — на машине агента, **[владелец]** — в GitHub.

1. **[человек]** Создайте ключ (без пароля, чтобы агент мог пушить сам):
   ```
   ssh-keygen -t ed25519 -C "agent-y" -f ~/.ssh/y_deploy -N ""
   ```
2. **[человек]** Отправьте владельцу **только публичную часть** — содержимое `~/.ssh/y_deploy.pub` (одна строка, начинается с `ssh-ed25519`). Её можно пересылать в любом мессенджере. Файл `~/.ssh/y_deploy` без `.pub` — секретный, его никому не отправлять.
3. **[владелец]** GitHub → репозиторий `polynomial-multiplication-negacyclic-batched` → Settings → Deploy keys → Add deploy key: название `agent-y`, вставить присланную строку, отметить **Allow write access** → Add key. Сообщить человеку, что ключ добавлен.
4. **[человек]** Добавьте в `~/.ssh/config`:
   ```
   Host github-y
       HostName github.com
       User git
       IdentityFile ~/.ssh/y_deploy
       IdentitiesOnly yes
   ```
5. **[человек]** Клонируйте и проверьте:
   ```
   git clone git@github-y:noname9006/polynomial-multiplication-negacyclic-batched.git
   ssh -T git@github-y
   ```
   Вторая команда должна ответить приветствием с именем репозитория.
6. **[владелец]** Защитите `master`: Settings → Branches → Add rule → `master` → Restrict who can push → только владелец. Deploy key с записью может пушить в любую ветку, а в master вливает только основной агент.
7. **[владелец]** Отключить доступ — удалить ключ в Settings → Deploy keys.

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

Токен не вписывать в файлы репозитория, задания, `.env`, аргументы команд и чат. После челленджа отозвать токен FHERMA и удалить deploy key.

## 4. Запуск агента
Откройте агента в папке клона и дайте ему первым сообщением:

> Прочитай `agents/Y/README_AGENT_Y.md` и работай по нему. Текущее задание — последний файл `agents/Y/TASK_*.md`.

## 5. Связь с основным агентом
- Задания основной агент кладёт в `agents/Y/TASK_*.md` в ветке `master`.
- Y работает в ветках `agent-y/<тема>` и пишет отчёты в `agents/Y/REPORT.md` в своей ветке.
- Чтобы основной агент узнал о сдаче, сообщите ему: «Y сдал, ветка agent-y/<тема>».

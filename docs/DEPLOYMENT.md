# Орналастыру (продакшн)

Бұл нұсқаулық SafeNet KZ-ты нақты ұйымда іске қосуға арналған: қол жеткізуді басқару, аудит, HTTPS,
PostgreSQL және API интеграциясы.

## 1. Режимдер

| режим | қалай қосылады | не болады |
|---|---|---|
| `development` (әдепкі) | `make app`, `make demo`, `make api`, тесттер | Django dev-сервері, `DEBUG` қосулы, HTTP |
| `production` | `SAFENET_ENV=production` (`.env`), `make serve` | `DEBUG` өшірулі, міндетті құпия кілт, HTTPS cookie, HSTS, waitress/gunicorn, логтар файлда |

Продакшнда міндетті параметрлер (`DJANGO_SECRET_KEY`, `DJANGO_ALLOWED_HOSTS`) жоқ болса немесе әлсіз болса,
қосымша **іске қосылмайды**. Бұл әдейі жасалған қорғаныс.

## 2. Талаптар

- Python 3.12+.
- PostgreSQL 14+. SQLite тек бір серверлі шағын орнатуда жарайды.
- Tesseract 5 (rus + kaz + eng) — скриншоттарды оқу үшін. Ол болмаса, скриншоттар мәтінсіз талданады, ал
  нәтижеге ескерту қосылады.
- TLS сертификаты бар reverse proxy: nginx (Linux) немесе IIS / nginx (Windows).

## 3. Орнату

### 3.1. Код пен тәуелділіктер

```bash
git clone https://github.com/nureketech/django-SafeNet.git
```

```bash
cd django-SafeNet
```

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements-prod.txt
```

```bash
.venv/bin/pip install -e .
```

Windows-та `.venv\Scripts\pip` қолданылады. `requirements-prod.txt` құрамында PostgreSQL драйвері (psycopg)
мен gunicorn бар (gunicorn тек Linux-та жұмыс істейді).

### 3.2. Модельдер

Модельдер мен граф `models/` бумасына жасалады, бұл буманы репозиторий сақтамайды.

```bash
make data train eval
```

Бұл команда шамамен 1–2 минут алады. Ағымдағы модельдер синтетикалық деректерде оқытылған. Пилотта олар
аналитиктердің шешімдері бойынша қайта оқытылады ([ROADMAP.md](ROADMAP.md)).

OCR үшін:

```bash
make tessdata
```

### 3.3. Баптаулар (`.env`)

Үлгіні көшіріп, мәндерді толтырыңыз:

```bash
cp .env.example .env
```

`.env` файлын ешқашан репозиторийге қоспаңыз (ол `.gitignore`-да тұр).

| айнымалы | міндетті | мағынасы |
|---|:-:|---|
| `SAFENET_ENV` | ✅ | `production` |
| `DJANGO_SECRET_KEY` | ✅ | ≥ 50 кездейсоқ таңба: `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `DJANGO_ALLOWED_HOSTS` | ✅ | сервистің домені (үтір арқылы) |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | proxy артында | толық origin, мысалы `https://safenet.example.kz` |
| `SAFENET_HTTPS` |  | `1` (әдепкі): secure cookie, HSTS, https-ке бағыттау. `0` — тек TLS-сіз оқшау тест стенді үшін |
| `SAFENET_BEHIND_PROXY` |  | `1`: TLS-ті nginx/IIS аяқтайды (`X-Forwarded-Proto`/`-For` сенімді) |
| `SAFENET_DB_ENGINE` |  | `postgresql` немесе `sqlite` |
| `SAFENET_DB_NAME`, `_USER`, `_PASSWORD`, `_HOST`, `_PORT` |  | PostgreSQL қосылымы |
| `SAFENET_SESSION_HOURS` |  | сессия ұзақтығы (әдепкі 8 сағ.; браузер жабылғанда да аяқталады) |
| `SAFENET_LOGIN_MAX_FAILURES`, `SAFENET_LOGIN_LOCK_MINUTES` |  | қате кіру шегі және уақытша бұғат (5 рет / 15 мин) |
| `SAFENET_API_RATE` |  | бір токенге сұрау лимиті (әдепкі `600/m`) |
| `SAFENET_LOG_DIR` |  | `safenet.log` пен `audit.log` бумасы |
| `SAFENET_FEEDBACK_PATH` |  | аналитиктер шешімдерінің CSV көшірмесі |
| `SAFENET_DEMO_MODE` |  | `1` — «Тексеру» бетінде мысал мәтіндер (тек демонстрацияға) |
| `SAFENET_FETCH_ENABLED` |  | `1` (әдепкі) — t.me сілтемесінен пост пен арнаны оқу; серверден `https://t.me`-ге шығу керек. Жабық желіде `0` |
| `SAFENET_FETCH_TIMEOUT` |  | t.me-ден жауап күту уақыты, секунд (әдепкі 10) |

Баптауларды тексеру:

```bash
python manage.py check --deploy
```

### 3.4. Дерекқор және бірінші әкімші

```bash
python manage.py migrate
```

```bash
python manage.py createsuperuser
```

`migrate` екі рөл-топты өзі жасайды:

| рөл (топ) | рұқсаттар |
|---|---|
| `analyst` | кейстерді көру, контентті тексеру, нәтижені кезекке сақтау, шешім қою |
| `observer` | тек көру: кезек, кейстер, байланыстар, метрикалар, шешімдер тарихы |

Әкімші `/admin/` бетінде пайдаланушыларды жасап, оларға рөл береді (**Groups**). Рөлі жоқ пайдаланушы
жүйеге кіре алады, бірақ беттерді көре алмайды (403).

### 3.5. Синтетикалық кейстерді тазалау

Платформа синтетикалық деректерді өзі жүктемейді: `make app` кезекті толтырмайды, ал `load_cases` пен
`load_demo` командалары продакшнда (`SAFENET_ENV=production`) іске қосылмайды. Байланыстар графы тек
тексерілген және API арқылы сақталған кейстерден құрылады. Егер әзірлеу кезінде кезекке синтетикалық
(`dataset`) немесе демо (`demo`) кейстер жүктелген болса, іске қоспас бұрын оларды алып тастаңыз:

```bash
python manage.py purge_cases --origin dataset demo --yes
```

`--yes`-сіз команда тек санын көрсетеді.

## 4. Іске қосу

### 4.1. Windows Server (waitress)

```bash
make.bat serve PORT=8000
```

Бұл команда мына қадамдарды орындайды:

1. `SAFENET_ENV=production` орнатады;
2. `migrate` пен `collectstatic` жасайды;
3. `check --deploy` тексерісін өткізеді;
4. waitress-ті `0.0.0.0:8000` адресінде іске қосады (8 ағын, `THREADS=` арқылы өзгертіледі).

Алдына IIS (ARR + URL Rewrite) немесе nginx қойып, TLS-ті сонда аяқтаңыз. Қызмет ретінде іске қосу үшін
NSSM немесе Task Scheduler қолдануға болады.

### 4.2. Linux (gunicorn + nginx + systemd)

```bash
python manage.py collectstatic --noinput
```

`/etc/systemd/system/safenet.service`:

```ini
[Unit]
Description=SafeNet KZ
After=network.target postgresql.service

[Service]
User=safenet
WorkingDirectory=/opt/safenet
EnvironmentFile=/opt/safenet/.env
ExecStart=/opt/safenet/.venv/bin/gunicorn app.wsgi:application --bind 127.0.0.1:8000 --workers 4 --timeout 120
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

nginx (үзінді):

```nginx
server {
    listen 443 ssl http2;
    server_name safenet.example.kz;
    ssl_certificate     /etc/ssl/safenet.crt;
    ssl_certificate_key /etc/ssl/safenet.key;
    client_max_body_size 12m;            # скриншоттар ≤ 10 МБ

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

`.env`-те `SAFENET_BEHIND_PROXY=1` болуы керек. Статикалық файлдарды қосымша өзі береді (WhiteNoise), nginx-те
бөлек баптау қажет емес.

## 5. API интеграциясы

Әр жүйеге (коллектор, платформа, басқа ведомство) жеке токен беріледі:

```bash
python manage.py create_api_token telegram-collector --user admin
```

- Токен **бір рет** көрсетіледі. Дерекқорда тек оның SHA-256 хэші сақталады.
- Токенді `/admin/` → **API tokens** бөлімінде кері қайтарып алуға болады (**Revoke**).
- Соңғы қолданылған уақыты да сол жерде көрінеді.

Сұрау мысалы:

```bash
curl -X POST https://safenet.example.kz/analyze -H "Authorization: Bearer sn_..." -H "Content-Type: application/json" --data-binary @item.json
```

`item.json` мысалы:

```json
{"id": "tg-100234", "text": "...", "source_type": "post", "platform": "telegram", "account_id": "acc_...", "save": true}
```

- `"save": true` нәтижені аналитиктер кезегіне қосады (дереккөзі `api`).
- Бірдей `id` қайта жіберілсе, кейс жаңартылады, ал аналитик шешімдері сақталып қалады.
- `POST /batch` бір сұрауда 500 жазбаға дейін қабылдайды.
- `POST /analyze/image` скриншотты (≤ 10 МБ) қабылдайды. Сурет дискіге сақталмайды, тек танылған мәтін
  сақталады.
- Лимиттен асса, жауап 429 болады.
- `GET /health` мониторинг үшін: токенсіз тек `status` пен `version` қайтарады.
- Swagger (`/docs`) тек жүйеге кірген пайдаланушыға ашылады.

## 6. Логтар және аудит

| файл | мазмұны |
|---|---|
| `$SAFENET_LOG_DIR/safenet.log` | қосымша хабарлары, ескертулер мен қателер (10 × 10 МБ айналым) |
| `$SAFENET_LOG_DIR/audit.log` | әрекеттер журналы (50 × 10 МБ айналым) |
| `$SAFENET_FEEDBACK_PATH` | аналитиктер шешімдері (CSV, `user` бағанымен) |

`audit.log`-қа жазылатын әрекеттер:

- кіру, шығу, сәтсіз кіру және бұғат;
- кейсті ашу, контентті тексеру, шешім қою;
- API шақырулары (токен атауымен).

Әр жол форматы: `action=… user=… ip=…`. Шешімдер дерекқорда да сақталады: пайдаланушы, уақыт және
түсініктеме.

## 7. Сақтық көшірме және жаңарту

**Сақтық көшірме.** Күн сайын мыналардың көшірмесін алыңыз:

- дерекқор (`pg_dump`);
- `SAFENET_FEEDBACK_PATH`;
- `.env`;
- `models/` (немесе оны қайта жасайтын коммит).

**Жаңарту:**

```bash
git pull
```

```bash
pip install -r requirements-prod.txt
```

```bash
python manage.py migrate
```

Содан соң қызметті қайта іске қосыңыз. Модельдер өзгерсе, `make train eval` іске қосылып, `reports/metrics.md`
бұрынғы нұсқамен салыстырылады.

## 8. Іске қосар алдындағы тексеріс

- [ ] `SAFENET_ENV=production`, кездейсоқ `DJANGO_SECRET_KEY`, нақты `DJANGO_ALLOWED_HOSTS`
- [ ] HTTPS бар, `python manage.py check --deploy` ескертусіз өтеді
- [ ] PostgreSQL, сақтық көшірме бапталған
- [ ] Әкімші мен аналитиктер жасалған, рөлдер берілген, ортақ аккаунт жоқ
- [ ] `purge_cases --origin dataset demo --yes` орындалған
- [ ] `SAFENET_DEMO_MODE=0`
- [ ] Интеграцияларға жеке токендер берілген, қолданылмайтындары кері алынған
- [ ] `audit.log` пен `safenet.log` жазылып жатыр, диск көлемі бақыланады
- [ ] Деректерді сақтау мерзімі мен қол жеткізу тәртібі ұйымның ішкі ережелерімен бекітілген

# Profile Deduplication System

MVP-сервис для поиска, оценки и кластеризации дублей клиентских профилей на маркетплейсе скидок.

Система принимает batch-файлы с профилями пользователей, сохраняет их в объектное хранилище, запускает асинхронную обработку, строит пары-кандидаты через PySpark Blocking, рассчитывает признаки, применяет LightGBM-модель и формирует рекомендации по объединению профилей.

Ключевая особенность системы — инкрементальная обработка: новый batch сравнивается не только внутри себя, но и с профилями, которые уже были ранее загружены в систему.

---

## 1. Цель проекта

Цель проекта — разработать production-like MVP системы Entity Resolution для поиска дублей клиентских профилей.

Система должна:

* принимать batch-файлы в формате CSV или Parquet;
* сохранять исходные файлы в S3-compatible storage;
* хранить metadata, статусы обработки, предсказания и кластеры в PostgreSQL;
* асинхронно запускать ML-пайплайн через очередь;
* строить candidate pairs без полного сравнения всех профилей со всеми;
* рассчитывать признаки для пар профилей;
* применять ML-модель для оценки вероятности дубля;
* формировать бизнес-рекомендации: `auto_merge`, `manual_review`, `no_duplicate`;
* объединять связанные профили в кластеры;
* показывать результаты через Streamlit UI.

---

## 2. Архитектура системы

Общая схема обработки:

```text
Streamlit UI
→ Backend API
→ MinIO + PostgreSQL
→ Redis Queue
→ Pipeline Worker
→ PySpark Blocking
→ pandas Feature Engineering
→ LightGBM Inference
→ Business Decisions
→ Clustering
→ PostgreSQL
→ Backend API
→ Streamlit UI
```

### Компоненты

| Компонент                  | Назначение                                                                                      |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| Streamlit UI               | Пользовательский интерфейс для загрузки batch-файлов, запуска обработки и просмотра результатов |
| Backend API                | FastAPI-сервис для загрузки файлов, управления batch-ами и выдачи результатов                   |
| MinIO                      | S3-compatible объектное хранилище для CSV/Parquet batch-файлов                                  |
| PostgreSQL                 | Хранение metadata, статусов, профилей, predictions и clusters                                   |
| Redis                      | Очередь задач для асинхронной обработки batch-файлов                                            |
| Pipeline Worker            | Сервис, который слушает Redis Queue и запускает ML-пайплайн                                     |
| PySpark Blocking           | Генерация пар-кандидатов через blocking keys                                                    |
| pandas Feature Engineering | Расчёт признаков для каждой пары профилей                                                       |
| LightGBM                   | ML-модель для расчёта `match_score`                                                             |
| Clustering                 | Построение connected components поверх pairwise predictions                                     |

---

## 3. Инкрементальный сценарий обработки

Система использует master table `profiles`, поэтому при обработке нового batch-файла она учитывает историю ранее загруженных профилей.

### Как работает incremental matching

1. Пользователь загружает первый batch.
2. Backend API сохраняет файл в MinIO и создаёт запись в PostgreSQL.
3. Worker обрабатывает batch и сохраняет уникальные профили в таблицу `profiles`.
4. Пользователь загружает второй batch.
5. Worker сохраняет новые профили и загружает historical profiles из PostgreSQL.
6. PySpark Blocking строит пары двух типов:

   * `new ↔ new` — пары внутри нового batch;
   * `new ↔ historical` — пары между новым batch и ранее загруженными профилями.
7. Пары `historical ↔ historical` не пересчитываются, потому что они уже были обработаны ранее.
8. Для найденных пар рассчитываются признаки, LightGBM выдаёт `match_score`, а business rules формируют рекомендацию.
9. Пары с рекомендацией `auto_merge` объединяются в кластеры.

Такой подход позволяет системе находить совпадения между профилями, загруженными в разные моменты времени.

---

## 4. ML-пайплайн

### 4.1. PySpark Blocking

Полное сравнение всех профилей со всеми имеет квадратичную сложность и плохо масштабируется. Поэтому перед ML-инференсом используется blocking.

Blocking строит candidate pairs только для профилей, у которых совпал хотя бы один blocking key.

Используемые blocking keys:

* `phone`;
* `first_name + email_domain`;
* `last_name + email_domain`;
* `first_name + birthday`;
* `first_name + sex`.

PySpark используется для масштабируемого построения пар-кандидатов через self-join внутри блоков.

### 4.2. Feature Engineering

Для каждой пары профилей рассчитываются признаки:

| Признак                | Смысл                                          |
| ---------------------- | ---------------------------------------------- |
| `time_diff_hours`      | Разница во времени создания событий            |
| `event_count_diff`     | Разница в количестве событий                   |
| `fs_jaccard`           | Jaccard similarity по `fs_features`            |
| `fs_count_diff`        | Разница количества `fs_features`               |
| `same_geoid`           | Совпадает ли географический признак            |
| `same_first_name`      | Совпадает ли имя                               |
| `same_phone`           | Совпадает ли телефон                           |
| `same_sex`             | Совпадает ли пол                               |
| `same_email_domain`    | Совпадает ли домен email                       |
| `same_np_device`       | Совпадает ли device из non-processing features |
| `local_hour_mean_diff` | Разница среднего локального часа активности    |

`entity_id` не используется как признак модели. Он нужен только для offline evaluation.

### 4.3. LightGBM Inference

Модель LightGBM получает таблицу признаков и возвращает вероятность того, что два профиля относятся к одному пользователю.

Результат модели:

```text
profile1
profile2
match_score
is_duplicate
```

### 4.4. Business Decisions

После ML-инференса `match_score` переводится в бизнес-рекомендацию:

| Условие                      | Рекомендация    | Смысл                                  |
| ---------------------------- | --------------- | -------------------------------------- |
| `match_score >= 0.90`        | `auto_merge`    | Профили можно автоматически объединить |
| `0.65 <= match_score < 0.90` | `manual_review` | Нужно ручное рассмотрение              |
| `match_score < 0.65`         | `no_duplicate`  | Профили не считаются дублями           |

### 4.5. Clustering

Pairwise predictions дают связи между парами профилей. Чтобы получить группы профилей одного пользователя, система строит граф:

```text
profile_id = вершина графа
пара auto_merge = ребро графа
```

Затем connected components превращаются в кластеры.

Пример:

```text
A похож на B
B похож на C
→ cluster = [A, B, C]
```

---

## 5. Структура PostgreSQL

Схема PostgreSQL создаётся через SQL-init скрипт:

```text
postgres/init/001_create_batches.sql
```

Приложения Backend API и Pipeline Worker не выполняют DDL-операции. Они работают только с уже созданными таблицами.

### Основные таблицы

| Таблица       | Назначение                                                   |
| ------------- | ------------------------------------------------------------ |
| `batches`     | Metadata batch-файлов и статусы обработки                    |
| `profiles`    | Master table всех уникальных профилей, загруженных в систему |
| `predictions` | Pairwise predictions для обработанных batch-файлов           |
| `clusters`    | Кластеры профилей, построенные по auto_merge-связям          |

### Таблица `batches`

Хранит информацию о загруженных batch-файлах:

* `batch_id`;
* `filename`;
* `object_key`;
* `s3_uri`;
* `status`;
* `uploaded_at`;
* `updated_at`.

### Таблица `profiles`

Master table профилей:

* `profile_id`;
* `first_seen_batch_id`;
* `last_seen_batch_id`;
* `first_seen_at`;
* `last_seen_at`;
* `source_filename`;
* основные поля профиля;
* feature-поля.

Эта таблица нужна для инкрементальной дедупликации.

### Таблица `predictions`

Хранит результат ML-инференса для пар профилей:

* `profile1`;
* `profile2`;
* `match_score`;
* `is_duplicate`;
* `recommendation`.

### Таблица `clusters`

Хранит результаты кластеризации:

* `cluster_id`;
* `batch_id`;
* `profile_id`;
* `cluster_size`;
* `recommendation_mode`.

---

## 6. API

Backend API реализован на FastAPI.

### Основные endpoints

| Method | Endpoint                          | Назначение                                 |
| ------ | --------------------------------- | ------------------------------------------ |
| `GET`  | `/health`                         | Проверка состояния API                     |
| `POST` | `/upload`                         | Загрузка CSV/Parquet batch-файла           |
| `GET`  | `/batches`                        | Получить список batch-файлов               |
| `GET`  | `/batches/{batch_id}`             | Получить metadata batch-а                  |
| `POST` | `/batches/{batch_id}/process`     | Поставить batch в Redis Queue на обработку |
| `GET`  | `/batches/{batch_id}/predictions` | Получить pairwise predictions              |
| `GET`  | `/batches/{batch_id}/clusters`    | Получить кластеры профилей                 |

---

## 7. Запуск проекта

### 7.1. Требования

Для запуска нужны:

* Docker;
* Docker Compose;
* Python 3.11 для локальных вспомогательных скриптов.

### 7.2. Переменные окружения

Проект использует `.env` файл.

Пример:

```env
POSTGRES_USER=root
POSTGRES_PASSWORD=root
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=flocktory

MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ENDPOINT=minio:9000
MINIO_BUCKET=flocktory-batches

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_QUEUE_NAME=batch_processing_queue

BACKEND_API_URL=http://backend-api:8000
```

### 7.3. Полный запуск

Из корня проекта:

```bash
docker compose up -d postgres minio minio-init redis backend-api pipeline-worker streamlit
```

Проверить сервисы:

```bash
docker compose ps
```

Проверить API:

```bash
curl http://localhost:8000/health
```

Открыть Streamlit UI:

```text
http://localhost:8501
```

---

## 8. Demo-сценарий

Для демонстрации инкрементального matching используется сценарий с двумя batch-файлами.

### 8.1. Сгенерировать demo-batches

Идея demo:

* в первом batch находится один профиль из известного кластера;
* во втором batch находятся другие профили из того же кластера;
* система должна найти совпадение между новым batch и historical profiles.

### 8.2. Чистый запуск demo

Если нужны чистые данные:

```bash
docker compose down -v
docker compose up -d postgres minio minio-init redis backend-api pipeline-worker streamlit
```

### 8.3. Загрузить первый batch

Через Streamlit UI или через curl:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@./incremental_known_cluster_batch_1.parquet"
```

Запустить обработку:

```bash
curl -X POST http://localhost:8000/batches/<BATCH_1_ID>/process
```

Ожидаемый результат в логах worker-а:

```text
Profiles saved to master table: ...
Historical profiles loaded: 0
```

### 8.4. Загрузить второй batch

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@./incremental_known_cluster_batch_2.parquet"
```

Запустить обработку:

```bash
curl -X POST http://localhost:8000/batches/<BATCH_2_ID>/process
```

Ожидаемый результат в логах worker-а:

```text
Historical profiles loaded: больше 0
Candidate pairs generated by Spark before filtering: больше 0
Candidate pairs after incremental filtering: больше 0
Features generated: больше 0
Predictions saved to PostgreSQL: больше 0
Cluster records saved to PostgreSQL: больше 0
```

### 8.5. Посмотреть результаты

Через API:

```bash
curl http://localhost:8000/batches/<BATCH_2_ID>/predictions
curl http://localhost:8000/batches/<BATCH_2_ID>/clusters
```

Или через Streamlit:

```text
http://localhost:8501
```

Во вкладке **Кластеры** можно увидеть профили, объединённые в один кластер, а также batch первого и последнего появления каждого профиля.

---

## 9. Логи и отладка

Логи worker-а:

```bash
docker compose logs pipeline-worker --tail=300 -f
```

Логи backend-а:

```bash
docker compose logs backend-api --tail=200 -f
```

Проверить таблицы PostgreSQL:

```bash
docker exec -it postgres psql -U root -d flocktory -c "\dt"
```

Проверить количество профилей:

```bash
docker exec -it postgres psql -U root -d flocktory -c "SELECT COUNT(*) FROM profiles;"
```

Посмотреть профили:

```bash
docker exec -it postgres psql -U root -d flocktory -c "SELECT profile_id, first_seen_batch_id, last_seen_batch_id, source_filename FROM profiles LIMIT 10;"
```

---

## 10. Структура проекта

Структура репозитория:

```text
.
├── backend-api/
│   └── app/
│       ├── main.py
│       ├── db.py
│       ├── redis_queue.py
│       └── storage.py
│
├── pipeline-worker/
│   └── app/
│       ├── worker.py
│       ├── run_batch.py
│       ├── db.py
│       ├── storage.py
│       ├── spark_blocking.py
│       ├── features.py
│       ├── clustering.py
│       ├── decisions/
│       │   └── recommendations.py
│       └── models/
│           ├── predict.py
│           └── artifacts/
│               └── matching_model_v1.joblib
│
├── postgres/
│   └── init/
│       └── 001_create_batches.sql
│
├── streamlit/
│   └── app.py
│
├── docker-compose.yml
├── .env
└── README.md
```

---

## 11. Ограничения MVP

MVP имеет ряд ограничений:

1. Используется batch-processing, а не real-time streaming.
2. Кластеры строятся на основе `auto_merge` pairwise predictions.
3. Полная production-логика с разрешением конфликтов при слиянии старых кластеров пока не реализована.
4. Feature engineering использует упрощённое агрегирование профиля.
5. Нет отдельного сервиса мониторинга качества модели и data drift.
6. Нет полноценной системы версионирования модели и признаков.
7. Нет авторизации и разграничения доступа в UI/API.

---

## 12. Возможные улучшения

Следующие шаги развития системы:

* добавить глобальную таблицу `global_clusters`;
* реализовать полноценное merge/unmerge управление кластерами;
* добавить ручной review для пар `manual_review`;
* внедрить мониторинг качества модели;
* добавить data drift monitoring;
* добавить версионирование модели и feature schema;
* вынести ML inference в отдельный сервис;
* добавить Airflow/Prefect для orchestration;
* реализовать near-real-time обработку через Kafka;
* добавить авторизацию в Streamlit и Backend API;
* добавить CI/CD pipeline.

---

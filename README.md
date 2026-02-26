## fun-ai-station-api (Backend)

Python backend service for Fun AI Station.

### Tech stack
- **FastAPI**: HTTP API framework
- **SQLAlchemy 2.x**: ORM
- **Alembic**: DB migrations
- **MySQL**: primary database (local dev)

### Local setup

1) Create venv and install deps

```bash
cd fun-ai-station-api
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Configure env

```bash
copy local.env.example local.env
```

也可以使用推荐路径（更适合服务器部署）：

```bash
copy local.env.example configs/fun-ai-station-api.env
```

3) Run migrations (creates tables)

```bash
alembic upgrade head
```

4) Start dev server

```bash
uvicorn src.main:app --reload --port 8001
```

### Scheduler worker (方案 2：独立进程)

启动一个单独的 worker 进程，用于轮询数据库并触发用户定时任务：

```bash
cd fun-ai-station-api
python -m src.scheduler_worker --poll 5 --batch 10
```

单次执行（调试用）：

```bash
python -m src.scheduler_worker --once
```

### API docs
- Swagger UI: `http://localhost:8001/docs`
- OpenAPI JSON: `http://localhost:8001/openapi.json`

### Endpoints
- `GET /health`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `GET /agents`
- `GET /agents/{agent_id}`
- `POST /chat/sessions`
- `GET /chat/sessions/{session_id}`
- `POST /chat/sessions/{session_id}/messages`
- `GET /chat/sessions/{session_id}/messages`
- `GET /scheduled-tasks`
- `POST /scheduled-tasks`
- `PUT /scheduled-tasks/{task_id}`
- `DELETE /scheduled-tasks/{task_id}`
- `GET /scheduled-tasks/{task_id}/runs`


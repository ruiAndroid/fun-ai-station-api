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

3) Run migrations (creates tables)

```bash
alembic upgrade head
```

4) Start dev server

```bash
# Windows PowerShell (current session):
$env:ENV_FILE="local.env"
uvicorn src.main:app --reload --port 8001
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


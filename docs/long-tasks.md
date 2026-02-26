### 长任务（Long Tasks）模块说明

长任务用于执行可能超过 HTTP 请求超时的操作。当前 MVP 仅支持一种任务类型：

- `orchestrator_execute`：调用 orchestrator 的 `/dispatch/execute`，并保存最终 `output`/`result`。

---

## 1) 组件与职责

- **数据模型**
  - `src/models/long_task.py`
  - Alembic：`alembic/versions/0007_long_tasks.py`
- **HTTP API（创建/查询/取消）**
  - `src/api/routes/long_tasks.py`（挂载在 `src/api/router.py`）
- **Long Task Worker（独立进程轮询 DB）**
  - `src/long_task_worker.py`
  - 通过 `src/core/orchestrator_client.py:dispatch_execute(..., timeout_seconds=...)` 执行编排
- **前端页面（简易管理台）**
  - `src/app/(site)/long-tasks/page.tsx`
  - `src/app/(site)/long-tasks/long-tasks-client.tsx`
  - API SDK：`src/lib/long-tasks.ts`
- **部署（systemd）**
  - `deploy/systemd/fun-ai-station-long-scheduler.service`
  - `deploy/scripts/update-api.sh`（更新 API 时若检测到 unit 已安装，会顺带重启 long scheduler）

---

## 2) API

- `GET /long-tasks`：列出当前用户任务（按 `updated_at` 倒序，最多 50）
- `POST /long-tasks/orchestrator-execute`：创建任务入队
- `GET /long-tasks/{task_id}`：查询任务详情
- `POST /long-tasks/{task_id}/cancel`：请求取消（best-effort）

---

## 3) Worker

启动 long task worker：

```bash
cd fun-ai-station-api
python -m src.long_task_worker --poll 2 --batch 10 --lease 600
```

配置项：

- `LONG_TASK_EXECUTE_TIMEOUT_SECONDS`：单次 orchestrator 执行超时（秒），默认 600

---

## 4) systemd

模板：`deploy/systemd/fun-ai-station-long-scheduler.service`

- `WorkingDirectory=/opt/fun-ai-station-api`
- `ExecStart=/opt/fun-ai-station-api/.venv/bin/python -m src.long_task_worker ...`
- 日志：
  - stdout：`/data/funai/logs/fun-ai-station-long-scheduler/out.log`
  - stderr：`/data/funai/logs/fun-ai-station-long-scheduler/err.log`


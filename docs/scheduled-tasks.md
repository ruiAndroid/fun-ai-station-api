### 定时任务（Scheduled Tasks）模块说明

本文档记录 fun-ai-station 的“用户级定时任务”能力：如何创建任务、如何计算下次执行时间、worker 如何抢占/执行任务，以及部署与排障要点。

---

## 1) 组件与职责

- **数据模型**
  - `src/models/scheduled_task.py`
  - Alembic：`alembic/versions/0006_scheduled_tasks.py`
- **HTTP API（CRUD + Runs）**
  - `src/api/routes/scheduled_tasks.py`（挂载在 `src/api/router.py`）
- **Scheduler Worker（独立进程轮询 DB）**
  - `src/scheduler_worker.py`
  - 通过 `src/core/orchestrator_client.py:dispatch_execute(...)` 触发编排/执行
- **前端页面（简单管理台）**
  - `src/app/(site)/scheduled-tasks/page.tsx`
  - `src/app/(site)/scheduled-tasks/scheduled-tasks-client.tsx`
  - API SDK：`src/lib/scheduled-tasks.ts`
- **部署（systemd）**
  - `deploy/systemd/fun-ai-station-scheduler.service`
  - `deploy/scripts/update-api.sh`（更新 API 时顺带重启 scheduler）

---

## 2) 数据表与字段语义

### 2.1 scheduled_tasks

- `user_id`：多租户隔离；API 层只允许读写自己的任务
- `enabled`：是否启用
- `schedule_type`：
  - `cron`：cron 表达式（默认）
  - `interval`：间隔秒数（`schedule_expr=int`，<=0 会回退到 60）
  - `once`：一次性任务（由 `next_run_at` 决定执行时刻；执行成功后会自动禁用）
- `schedule_expr`：随 `schedule_type` 变化
- `timezone`：cron 的解释时区（用于把“本地时区的下一次触发点”转换为 UTC）
- `payload`：执行入参（自由 JSON），worker 会规范化为 orchestrator 的 `dispatch_execute(...)` 参数
- `next_run_at` / `last_run_at`：UTC naive datetime（数据库里不带 tzinfo）
- `locked_by` / `locked_until`：简单租约（lease）字段，用于多 worker 并发时避免重复执行

### 2.2 scheduled_task_runs

每次执行都会写一条 run 记录（用于前端“运行记录”展示）：

- `status`: `running | success | failed`
- `trace_id`: worker 生成，用于把本次执行串起来排查
- `error` / `result`: 失败原因与返回结果（JSON）

---

## 3) API 设计（/scheduled-tasks）

文件：`src/api/routes/scheduled_tasks.py`

- `GET /scheduled-tasks`：列出当前用户任务（按 `updated_at` 倒序）
- `POST /scheduled-tasks`：创建任务
  - 若未传 `next_run_at` 且 `schedule_type != once`，后端会基于 `schedule_type/schedule_expr/timezone` 计算 `next_run_at`
- `PUT /scheduled-tasks/{task_id}`：更新任务
  - 更新时会清理 `locked_by/locked_until`（避免“锁卡住”的错觉）
- `DELETE /scheduled-tasks/{task_id}`：删除任务
- `GET /scheduled-tasks/{task_id}/runs`：列出最近 50 条运行记录

示例（interval，每 60s 执行一次）：

```bash
curl -X POST "$API_BASE/scheduled-tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "demo interval",
    "enabled": true,
    "schedule_type": "interval",
    "schedule_expr": "60",
    "timezone": "UTC",
    "payload": {
      "text": "每天 9 点提醒我打卡",
      "context": {"channel":"scheduled_task"}
    }
  }'
```

示例（once，一次性在指定时间执行；注意 `next_run_at` 需要是 ISO datetime）：

```bash
curl -X POST "$API_BASE/scheduled-tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "demo once",
    "schedule_type": "once",
    "timezone": "UTC",
    "next_run_at": "2026-02-27T01:30:00",
    "payload": {"text":"hello","mode":"hybrid"}
  }'
```

---

## 4) Worker 执行逻辑（DB poll + lease + dispatch）

文件：`src/scheduler_worker.py`

### 4.1 抢占（claim）

- 周期性 tick（默认 `--poll 5` 秒）：
  - 查询 `enabled = true` 且 `next_run_at <= now` 且（未锁或锁已过期）
  - `SELECT ... FOR UPDATE SKIP LOCKED` 抢占一批（`--batch`）
  - 写入 `locked_by/locked_until`（租约时长 `--lease`，默认 120 秒）

### 4.2 执行（run）

- 为每个 task 写入一条 `scheduled_task_runs(status=running, trace_id=...)`
- 解析 `payload`（`parse_payload`）映射到 orchestrator：
  - `text`：`payload.text`（兼容 `payload.input`）
  - `context`：dict（worker 会 best-effort 注入 `user_id` / `scheduled_task_id`）
  - `default_agent`：缺省走 `SCHEDULER_DEFAULT_AGENT`；否则 `OPENCLAW_DEFAULT_AGENT`；再缺省为 `attendance`
  - `mode`：缺省走 `SCHEDULER_ROUTER_MODE`；否则 `ROUTER_MODE`；再缺省为 `hybrid`
  - `forced_agent/items`：当前不支持在 payload 中覆盖（避免用户绕过编排层）
- 调用 `dispatch_execute(...)`：
  - 成功：run 标记 `success`，task 计算下一次执行时间
  - 失败：run 标记 `failed`，task 做一个基础退避（当前实现：+60s，避免热循环）

### 4.3 next_run_at 计算规则

- `interval`：`after_utc + seconds`，其中：
  - `seconds=int(schedule_expr)`
  - 若 `seconds <= 0` 则回退为 `60`
  - 若 `seconds < SCHEDULED_TASK_INTERVAL_MIN_SECONDS` 则抬到该最小值
- `cron`：用 `croniter` 基于 `timezone` 计算“下一次本地时间”，再转换回 UTC naive
- `once`：
  - `ok=true`：自动 `enabled=false` 且 `next_run_at=null`
  - `ok=false`：会写入一个“下次重试时间”（当前逻辑同样走 +60s backoff）

### 4.4 安全限制（建议保留）

- `SCHEDULED_TASK_INTERVAL_MIN_SECONDS`：interval 最小秒数（默认 10），低于该值会被 API 拒绝/worker 抬高
- `SCHEDULED_TASKS_MAX_ENABLED_PER_USER`：每个用户最多同时启用的定时任务数量（默认 20），超过会被 API 拒绝

---

## 5) 部署与运维

### 5.1 systemd

模板：`deploy/systemd/fun-ai-station-scheduler.service`

- `WorkingDirectory=/opt/fun-ai-station-api`
- `ExecStart=/opt/fun-ai-station-api/.venv/bin/python -m src.scheduler_worker --poll 5 --batch 10 --lease 120`
- 依赖：`After=network.target mysqld.service fun-agent-service.service`

日志（当前 unit 采用 append 到文件）：

- stdout：`/data/funai/logs/fun-ai-station-scheduler/out.log`
- stderr：`/data/funai/logs/fun-ai-station-scheduler/err.log`

### 5.2 update-api.sh

`deploy/scripts/update-api.sh` 会在更新/重启 `fun-ai-station-api` 后：

- 若检测到 `fun-ai-station-scheduler.service` 已安装，则创建日志目录并重启 scheduler
- 若未安装，则打印 WARN 跳过

---

## 6) 排障建议

- 任务不触发：
  - 查 DB：`next_run_at` 是否为过去时间、`enabled` 是否为 true
  - 查锁：`locked_until` 是否被某个 worker 长期占用（可通过 `PUT` 更新任务触发清锁）
  - 看 scheduler 日志：是否能连 DB、是否能调用 orchestrator
- cron 时间不对：
  - 确认 `timezone` 是否正确（cron 是按该时区解释后再转换为 UTC）
  - `schedule_expr` 为空会回退到 `* * * * *`（每分钟）

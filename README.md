# AI API

一个基于 FastAPI 的 OpenAI 兼容代理服务。

这个项目最早来源于一个轻量级代理实现，当前版本已经围绕实际使用场景做了较多调整，重点解决了多 Key 轮询时的限频、重试与兼容性问题，适合作为自托管的统一 API 入口。

## 项目定位

适用于以下场景：

- 需要把多个上游 API Key 封装成一个统一入口
- 需要通过自定义 Bearer Token 控制下游客户端访问
- 需要减少 429、临时性 5xx 等错误对调用方的影响
- 需要把不同兼容后端统一暴露为 OpenAI 风格接口

## 当前特性

- 多 API Key 轮询
- 全局限频控制，避免请求过于密集
- 单 Key 限频控制，降低同一个 Key 被连续打爆的概率
- 自动重试与换 Key
- 支持非流式和流式聊天补全
- 支持 `embeddings` 透传
- 支持模型列表查询
- Bearer Token 鉴权
- 默认开启 CORS
- 可选 Gemini 兼容处理

## 工作方式

服务对外暴露 OpenAI 风格接口，对内将请求转发到你配置的上游 `BASE_URL`。

请求进入后大致会经历以下流程：

1. 校验客户端传入的 Bearer Token
2. 按轮询顺序选择一个上游 API Key
3. 执行全局限频与单 Key 限频
4. 将请求转发给上游兼容接口
5. 如果遇到可恢复错误，则自动重试或切换到下一个 Key

默认会对以下状态码进行重试：

- `429`、`401`、`403`：认为当前 Key 不可用，尝试切换下一个 Key
- `408`、`500`、`502`、`503`、`504`：认为属于临时故障，继续尝试

## 兼容后端

默认模式是 `generic`，适用于标准的 OpenAI 兼容接口。

当后端为 Gemini 兼容接口时，可以：

- 将环境变量 `BACKEND_MODE` 设置为 `gemini`
- 或在请求里使用以 `gemini` 开头的模型名，让服务自动识别

Gemini 模式下会额外做两件事：

- 清洗历史消息中的工具调用记录，避免把不兼容的函数调用结构继续回传给 Gemini
- 当请求包含 `tools` 时，自动设置 `parallel_tool_calls=false`

## 环境要求

- Python 3.10+
- 可选：Docker

## 安装与启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

项目默认读取根目录下的 `.env` 文件。
你可以直接复制 [`.env.example`](c:\Users\Jackson\gitWorks\ai-api\.env.example) 为 `.env` 后再修改。

示例：

```env
API_KEYS=["sk-key-1","sk-key-2"]
ALLOWED_TOKENS=["client-token-1","client-token-2"]
BASE_URL="https://api.openai.com/v1"
BACKEND_MODE="generic"
GLOBAL_THROTTLE_INTERVAL=1.0
KEY_THROTTLE_INTERVAL=6.0
```

配置项说明：

- `API_KEYS`：上游 API Key 列表，必填
- `ALLOWED_TOKENS`：允许客户端访问本服务的 Bearer Token 列表，必填
- `BASE_URL`：上游兼容接口地址，必填
- `BACKEND_MODE`：后端模式，可选值为 `generic` 或 `gemini`
- `GLOBAL_THROTTLE_INTERVAL`：任意两次请求发送之间的最小全局间隔，单位秒
- `KEY_THROTTLE_INTERVAL`：同一个 Key 两次使用之间的最小间隔，单位秒
- `ENV_FILE`：可选，用于指定另一个环境文件路径

### 3. 本地运行

```bash
python main.py
```

或：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

服务默认监听 `8000` 端口。

### 4. 快速验证

服务启动后，可以先调用健康检查：

```bash
curl http://127.0.0.1:8000/health
```

再测试一次聊天补全：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer client-token-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ],
    "stream": false
  }'
```

## Docker 部署

### 构建镜像

```bash
docker build -t ai-api .
```

### 启动容器

```bash
docker run -d \
  --name ai-api \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  ai-api
```

如果你不使用 `.env` 文件，也可以直接通过 `-e` 传入环境变量。

当前依赖在 `requirements.txt` 中使用了版本区间约束，目的是降低上游大版本升级带来的兼容性风险。

## API 接口

### 健康检查

无需鉴权：

```http
GET /health
```

### 获取模型列表

```http
GET /v1/models
Authorization: Bearer client-token-1
```

### 聊天补全

```http
POST /v1/chat/completions
Authorization: Bearer client-token-1
Content-Type: application/json

{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": "你好"
    }
  ],
  "stream": false
}
```

流式请求同样可用，只需把 `stream` 设为 `true`。

### 向量接口

```http
POST /v1/embeddings
Authorization: Bearer client-token-1
Content-Type: application/json

{
  "model": "text-embedding-3-small",
  "input": "hello"
}
```

## 返回行为说明

- 对外接口尽量保持 OpenAI 兼容风格
- 非流式请求会返回上游 JSON 结果
- 流式请求会以 `text/event-stream` 方式转发
- 当所有 Key 都失败时，服务会返回聚合后的错误信息

## 项目结构

- `main.py`：主应用入口、路由、转发与重试逻辑
- `app/config.py`：环境变量与配置加载
- `Dockerfile`：容器构建配置
- `requirements.txt`：Python 依赖

## 使用建议

- `ALLOWED_TOKENS` 不要直接复用上游真实 API Key
- 生产环境建议通过反向代理或网关进一步限制来源
- 如果上游限速严格，优先根据实际配额调大两个 throttle 参数
- 默认 CORS 为全开放，如有公网暴露需求，建议按实际来源收紧

## 致谢

项目起点来自一个开源代理实现，但当前版本已经针对多 Key 调度、限频控制、异常恢复和 Gemini 兼容性做了较多重构与补充。

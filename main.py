from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
import json
import time
from copy import deepcopy
from itertools import cycle
import asyncio
import httpx
import uvicorn
from app import config

# ===== 日志配置 =====
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===== FastAPI 应用初始化 =====
app = FastAPI()

# ===== 跨域配置 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== API Key 配置 =====
API_KEYS = config.settings.API_KEYS
key_cycle = cycle(API_KEYS)
key_lock = asyncio.Lock()
GLOBAL_THROTTLE_INTERVAL = config.settings.GLOBAL_THROTTLE_INTERVAL
KEY_THROTTLE_INTERVAL = config.settings.KEY_THROTTLE_INTERVAL
RPM_COOLDOWN_SECONDS = config.settings.RPM_COOLDOWN_SECONDS
BASE_URL = getattr(config.settings, "BASE_URL", None)
BACKEND_MODE = getattr(config.settings, "BACKEND_MODE", None)

if not BASE_URL:
    raise RuntimeError("BASE_URL is required")

# ===== 错误分组 =====
RETRY_WITH_NEXT_KEY = {429, 401, 403}   # 该 Key 有问题，换一个可能好用
RETRY_TRANSIENT = {500, 502, 503, 504, 408}  # 临时故障，重试可能好用

# ===== 节流状态（基于发送时间） =====
global_last_send_time = 0       # 全局上次发送时间
key_last_send_time = {}         # 每个 Key 的上次发送时间
throttle_lock = asyncio.Lock()  # 节流锁，保证判断+等待+记录的原子性

# ===== RPM 冷却状态 =====
key_rpm_cooldown_until: dict[str, float] = {}  # {api_key: cooldown_expire_timestamp}


def classify_429(response_text: str) -> str:
    """
    解析 429 响应体，区分限流类型：
      'tpm'     - Token 用量超限（换 key 无意义，应立即失败）
      'rpm'     - 请求频率超限（换 key 可能有效）
      'unknown' - 无法判断，保守地继续重试
    """
    text_lower = response_text.lower()
    # TPM / Quota 关键词优先判断
    if any(k in text_lower for k in ["token", "quota", "tokens per", "tpm"]):
        return "tpm"
    # RPM 关键词
    if any(k in text_lower for k in [
        "rate limit", "rpm", "requests per minute",
        "rate_limit_exceeded", "too many requests",
    ]):
        return "rpm"
    return "unknown"


def mark_rpm_cooldown(api_key: str):
    """将 key 标记为 RPM 冷却状态，冷却期间尽量避免再次选用"""
    expire = time.monotonic() + RPM_COOLDOWN_SECONDS
    key_rpm_cooldown_until[api_key] = expire
    logger.info(
        f"Key **********{api_key[-6:]} marked RPM cooldown for {RPM_COOLDOWN_SECONDS}s"
    )

# ===== 授权验证 =====
async def verify_authorization(authorization: str = Header(None)):
    if not authorization:
        logger.error("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        logger.error("Invalid Authorization header format")
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = authorization.replace("Bearer ", "")
    if token not in config.settings.ALLOWED_TOKENS:
        logger.error("Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# ===== 统一节流（基于发送时间，锁内原子完成） =====
async def enforce_throttle(api_key: str):
    """在请求发送前执行节流：检查间隔、等待、记录发送时间，全部在锁内原子完成"""
    global global_last_send_time

    async with throttle_lock:
        now = asyncio.get_event_loop().time()

        # 1. 全局间隔：距离上次任意请求发送的间隔
        global_wait = GLOBAL_THROTTLE_INTERVAL - (now - global_last_send_time)
        if global_wait > 0:
            logger.info(f"Global throttle: waiting {global_wait:.2f}s before request with key **********{api_key[-6:]}")
            await asyncio.sleep(global_wait)
            now = asyncio.get_event_loop().time()

        # 2. 单 Key 间隔：距离该 Key 上次使用的间隔
        if api_key in key_last_send_time:
            key_wait = KEY_THROTTLE_INTERVAL - (now - key_last_send_time[api_key])
            if key_wait > 0:
                logger.info(f"Key throttle: waiting {key_wait:.2f}s for key **********{api_key[-6:]}")
                await asyncio.sleep(key_wait)
                now = asyncio.get_event_loop().time()

        # 3. 记录本次发送时间（在锁内，发送前记录）
        global_last_send_time = now
        key_last_send_time[api_key] = now

# ===== 错误分类判断 =====
def should_retry(status_code: int) -> bool:
    """判断该状态码是否应该重试"""
    return status_code in RETRY_WITH_NEXT_KEY or status_code in RETRY_TRANSIENT


def detect_backend_mode(body: dict) -> str:
    """Resolve backend mode from explicit config first, then request model."""
    if BACKEND_MODE in {"generic", "gemini"}:
        return BACKEND_MODE

    model = str(body.get("model") or "").strip().lower()
    if model.startswith("gemini"):
        return "gemini"

    return "generic"


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                chunks.append(json.dumps(item, ensure_ascii=False))
                continue

            if item.get("type") == "text" and item.get("text"):
                chunks.append(item["text"])
            else:
                chunks.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(chunk for chunk in chunks if chunk)
    return json.dumps(content, ensure_ascii=False)


def sanitize_gemini_messages(messages: list) -> list:
    """
    Rewrite tool-call history into plain text context.

    Gemini requires provider-specific metadata such as thought_signature for
    functionCall history. OpenAI-compatible agents do not send it back, so we
    drop prior function-call records while preserving any assistant prose and
    the tool outputs that followed.
    """
    sanitized = []
    tool_call_name_by_id = {}

    for message in messages:
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            content_text = _content_to_text(message.get("content"))

            for tool_call in message.get("tool_calls", []):
                function_data = tool_call.get("function") or {}
                tool_name = function_data.get("name") or "tool"
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    tool_call_name_by_id[tool_call_id] = tool_name

            # Do not echo raw tool invocation payloads back into the message
            # history. Keep only any natural-language assistant text.
            if content_text:
                sanitized.append({"role": "assistant", "content": content_text})
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            tool_name = message.get("name") or tool_call_name_by_id.get(tool_call_id) or "tool"
            tool_output = _content_to_text(message.get("content"))
            sanitized.append(
                {
                    "role": "user",
                    "content": f"Tool result from {tool_name}:\n{tool_output}",
                }
            )
            continue

        sanitized.append(message)

    return sanitized


def prepare_chat_body(body: dict) -> dict:
    if detect_backend_mode(body) != "gemini":
        return body

    prepared = deepcopy(body)

    if isinstance(prepared.get("messages"), list):
        original_messages = prepared["messages"]
        prepared["messages"] = sanitize_gemini_messages(original_messages)
        if prepared["messages"] != original_messages:
            logger.info("Sanitized tool-call history for Gemini-compatible backend")

    if prepared.get("tools"):
        # Gemini tool execution is more reliable when constrained to one call at a time.
        prepared["parallel_tool_calls"] = False

    return prepared

# ===== 非流式请求的重试处理 =====
async def call_with_retry(path: str, body: dict = None, method: str = "POST"):
    """非流式请求：透传 body 到上游 API，自动轮询 Key 并重试"""
    tried_keys = set()
    last_exception = None

    while len(tried_keys) < len(API_KEYS):
        async with key_lock:
            api_key = next(key_cycle)

        if api_key in tried_keys:
            continue
        tried_keys.add(api_key)
        logger.info(f"Using API key (attempt {len(tried_keys)}/{len(API_KEYS)}): **********{api_key[-6:]}")

        try:
            # 节流控制（基于发送时间）
            await enforce_throttle(api_key)

            url = f"{BASE_URL}{path}"
            async with httpx.AsyncClient(timeout=120.0) as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                else:
                    response = await client.post(url, json=body, headers=headers)

            if response.status_code == 200:
                logger.info(f"Request successful with key **********{api_key[-6:]}")
                return response.json()

            # 判断是否可重试
            if should_retry(response.status_code):
                # --- 差异化处理 429 ---
                if response.status_code == 429:
                    kind = classify_429(response.text)
                    retry_after = response.headers.get("retry-after", str(RPM_COOLDOWN_SECONDS))
                    if kind == "tpm":
                        logger.warning(
                            f"TPM quota exceeded with key **********{api_key[-6:]}. "
                            f"Retry-After: {retry_after}s. Stopping key rotation."
                        )
                        raise HTTPException(
                            status_code=429,
                            detail=f"Token quota exceeded. Please retry after {retry_after}s.",
                            headers={"Retry-After": retry_after},
                        )
                    else:
                        # RPM 或 unknown：标记冷却，继续换 key
                        mark_rpm_cooldown(api_key)
                        logger.warning(
                            f"RPM rate limited (type={kind}) with key **********{api_key[-6:]} "
                            f"({len(tried_keys)}/{len(API_KEYS)} keys tried)."
                        )
                else:
                    error_type = "key-issue" if response.status_code in RETRY_WITH_NEXT_KEY else "transient"
                    logger.warning(
                        f"HTTP {response.status_code} ({error_type}) with key **********{api_key[-6:]} "
                        f"({len(tried_keys)}/{len(API_KEYS)} keys tried): {response.text[:200]}"
                    )
                last_exception = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
                continue  # enforce_throttle 会自动控制间隔

            # 不可重试的错误，直接返回
            logger.error(
                f"Non-recoverable HTTP {response.status_code} with key **********{api_key[-6:]}: "
                f"{response.text[:200]}"
            )
            raise HTTPException(status_code=response.status_code, detail=response.text)

        except httpx.HTTPStatusError:
            continue

        except HTTPException:
            raise

        except Exception as e:
            logger.error(f"Unexpected error with key **********{api_key[-6:]}: {str(e)}")
            last_exception = e
            continue

    # 所有 key 都失败了
    logger.error(f"All {len(API_KEYS)} API keys exhausted. Tried: {len(tried_keys)}")

    if isinstance(last_exception, httpx.HTTPStatusError):
        status = last_exception.response.status_code
        if status == 429:
            raise HTTPException(status_code=429, detail=f"All {len(API_KEYS)} API keys are rate limited. Please try again later.")
        elif status in {503, 502, 500, 504}:
            raise HTTPException(status_code=status, detail="Service temporarily unavailable. Try again later.")
        elif status in {401, 403}:
            raise HTTPException(status_code=status, detail=f"All {len(API_KEYS)} API keys failed authentication/authorization.")

    raise HTTPException(status_code=500, detail="All provider attempts failed.")

# ===== 流式请求的重试处理 =====
async def call_streaming_with_retry(path: str, body: dict):
    """流式请求：透传 body 到上游 API，返回 httpx 流式响应和使用的 API key"""
    tried_keys = set()

    while len(tried_keys) < len(API_KEYS):
        async with key_lock:
            api_key = next(key_cycle)

        if api_key in tried_keys:
            continue
        tried_keys.add(api_key)
        logger.info(f"Using API key (stream, attempt {len(tried_keys)}/{len(API_KEYS)}): **********{api_key[-6:]}")

        try:
            # 节流控制（基于发送时间）
            await enforce_throttle(api_key)

            url = f"{BASE_URL}{path}"
            client = httpx.AsyncClient(timeout=120.0)
            request = client.build_request(
                "POST",
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response = await client.send(request, stream=True)

            if response.status_code == 200:
                logger.info(f"Streaming request successful with key **********{api_key[-6:]}")
                return response, client, api_key

            # 非 200，读取响应体后关闭
            error_body = await response.aread()
            await response.aclose()
            await client.aclose()

            if should_retry(response.status_code):
                # --- 差异化处理 429 ---
                if response.status_code == 429:
                    kind = classify_429(error_body.decode())
                    retry_after = response.headers.get("retry-after", str(RPM_COOLDOWN_SECONDS))
                    if kind == "tpm":
                        logger.warning(
                            f"[Stream] TPM quota exceeded with key **********{api_key[-6:]}. "
                            f"Retry-After: {retry_after}s. Stopping key rotation."
                        )
                        raise HTTPException(
                            status_code=429,
                            detail=f"Token quota exceeded. Please retry after {retry_after}s.",
                            headers={"Retry-After": retry_after},
                        )
                    else:
                        # RPM 或 unknown：标记冷却，继续换 key
                        mark_rpm_cooldown(api_key)
                        logger.warning(
                            f"[Stream] RPM rate limited (type={kind}) with key **********{api_key[-6:]} "
                            f"({len(tried_keys)}/{len(API_KEYS)} keys tried)."
                        )
                else:
                    error_type = "key-issue" if response.status_code in RETRY_WITH_NEXT_KEY else "transient"
                    logger.warning(
                        f"HTTP {response.status_code} ({error_type}) in streaming with key **********{api_key[-6:]} "
                        f"({len(tried_keys)}/{len(API_KEYS)} keys tried): {error_body.decode()[:200]}"
                    )
                continue  # enforce_throttle 会自动控制间隔
            else:
                logger.error(
                    f"Non-recoverable HTTP {response.status_code} in streaming with key **********{api_key[-6:]}"
                )
                raise HTTPException(status_code=response.status_code, detail=error_body.decode())

        except HTTPException:
            raise

        except Exception as e:
            logger.error(f"Unexpected streaming error with key **********{api_key[-6:]}: {str(e)}")
            continue

    # 所有 key 都失败
    logger.error(f"All {len(API_KEYS)} API keys exhausted for streaming. Tried: {len(tried_keys)}")
    raise HTTPException(status_code=429, detail=f"All {len(API_KEYS)} API keys failed for streaming request. Please try again later.")

# ===== 模型列表 =====
@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    await verify_authorization(authorization)
    return await call_with_retry("/models", method="GET")

# ===== Chat Completions（透传模式） =====
@app.post("/v1/chat/completions")
async def chat_completion(request: Request, authorization: str = Header(None)):
    await verify_authorization(authorization)
    body = await request.json()
    body = prepare_chat_body(body)
    is_stream = body.get("stream", False)

    if is_stream:
        try:
            response, client, used_api_key = await call_streaming_with_retry("/chat/completions", body)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Streaming setup error: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected streaming error.")

        async def generate():
            try:
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                yield f"data: {json.dumps({'error': f'Streaming interrupted: {str(e)}'})}\n\n"
            finally:
                yield "data: [DONE]\n\n"
                logger.info(f"Streaming completely finished with key **********{used_api_key[-6:]}")
                await response.aclose()
                await client.aclose()

        return StreamingResponse(content=generate(), media_type="text/event-stream")

    else:
        return await call_with_retry("/chat/completions", body)

# ===== Embeddings（透传模式） =====
@app.post("/v1/embeddings")
async def embedding(request: Request, authorization: str = Header(None)):
    await verify_authorization(authorization)
    body = await request.json()
    return await call_with_retry("/embeddings", body)

# ===== 健康检查（无需鉴权） =====
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

# ===== 启动入口 =====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import logging
import json
import asyncio
import httpx
import openai
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

# ===== API Key 管理 =====
THROTTLE_INTERVAL = config.settings.THROTTLE_INTERVAL


class KeyManager:
    """管理 API Key 的轮转、节流和重试"""

    def __init__(self, keys: List[str]):
        self.keys = keys
        self.index = 0
        self.lock = asyncio.Lock()
        self.last_request_end_time: dict[str, float] = {}
        self.global_last_request_end_time: float = 0

    async def get_keys_for_retry(self) -> List[str]:
        """返回从当前位置开始的完整 key 列表（用于 retry 遍历所有 key）"""
        async with self.lock:
            start = self.index % len(self.keys)
            self.index += len(self.keys)  # 推进，避免下一个请求取到相同起点
            return self.keys[start:] + self.keys[:start]

    async def enforce_throttle(self, api_key: str, is_streaming: bool = False):
        """在请求开始前检查间隔"""
        current_time = asyncio.get_event_loop().time()

        # 检查全局间隔（基于上次请求结束时间）
        time_since_last_global = current_time - self.global_last_request_end_time
        if time_since_last_global < THROTTLE_INTERVAL:
            wait_time = THROTTLE_INTERVAL - time_since_last_global
            request_type = "streaming" if is_streaming else "regular"
            logger.info(
                f"Global throttle: waiting {wait_time:.2f}s before {request_type} request with key **********{api_key[-6:]}"
            )
            await asyncio.sleep(wait_time)
            current_time = asyncio.get_event_loop().time()  # 刷新时间

        # 检查单个 key 的间隔（基于该key上次请求结束时间）
        if api_key in self.last_request_end_time:
            time_since_last_key = current_time - self.last_request_end_time[api_key]
            if time_since_last_key < THROTTLE_INTERVAL:
                wait_time = THROTTLE_INTERVAL - time_since_last_key
                request_type = "streaming" if is_streaming else "regular"
                logger.info(
                    f"Key throttle: waiting {wait_time:.2f}s for {request_type} request with key **********{api_key[-6:]}"
                )
                await asyncio.sleep(wait_time)

    async def record_request_end(self, api_key: str, is_streaming: bool = False):
        """在请求结束后记录时间"""
        current_time = asyncio.get_event_loop().time()
        self.last_request_end_time[api_key] = current_time
        self.global_last_request_end_time = current_time

        request_type = "streaming" if is_streaming else "regular"
        logger.debug(
            f"Recorded {request_type} request end time for key **********{api_key[-6:]}"
        )


key_manager = KeyManager(config.settings.API_KEYS)


# ===== 请求体模型 =====
class ChatRequest(BaseModel):
    messages: List[dict]
    model: str = "llama-3.2-90b-text-preview"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8000
    stream: Optional[bool] = False
    tools: Optional[List[dict]] = []
    tool_choice: Optional[str] = "auto"


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "text-embedding-004"
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = 1536
    user: Optional[str] = None
    response_format: Optional[str] = "float"


# ===== 授权验证 =====
async def verify_authorization(authorization: str = Header(None)):
    if not authorization:
        logger.error("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        logger.error("Invalid Authorization header format")
        raise HTTPException(
            status_code=401, detail="Invalid Authorization header format"
        )
    token = authorization.replace("Bearer ", "")
    if token not in config.settings.ALLOWED_TOKENS:
        logger.error("Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


# ===== Retry 封装（非流式） =====
async def call_with_retry(func, *args, **kwargs):
    keys = await key_manager.get_keys_for_retry()
    last_exception = None
    rate_limited_count = 0
    total_keys = len(keys)

    for attempt, api_key in enumerate(keys):
        logger.info(
            f"Using API key (attempt {attempt + 1}/{total_keys}): **********{api_key[-6:]}"
        )

        try:
            # 请求前间隔控制
            await key_manager.enforce_throttle(api_key, is_streaming=False)

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=config.settings.BASE_URL,
                max_retries=0,
            )
            result = await func(client, *args, **kwargs)
            logger.info(f"Request successful with key **********{api_key[-6:]}")

            # 普通请求结束后记录时间
            await key_manager.record_request_end(api_key, is_streaming=False)
            return result

        except openai.RateLimitError as e:
            rate_limited_count += 1
            logger.warning(
                f"Rate limit error with key **********{api_key[-6:]} ({rate_limited_count}/{total_keys} keys exhausted): {str(e)}"
            )
            last_exception = e

            if attempt < total_keys - 1:
                logger.info(
                    f"Waiting {THROTTLE_INTERVAL}s before trying next API key..."
                )
                await asyncio.sleep(THROTTLE_INTERVAL)
            continue

        except openai.APIStatusError as e:
            if e.status_code in [429, 503]:
                rate_limited_count += 1
                logger.warning(
                    f"API status {e.status_code} with key **********{api_key[-6:]} ({rate_limited_count}/{total_keys} keys exhausted): {str(e)}"
                )
                last_exception = e

                if attempt < total_keys - 1:
                    logger.info(
                        f"Waiting {THROTTLE_INTERVAL}s before trying next API key..."
                    )
                    await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            else:
                logger.error(
                    f"Non-recoverable API error with key **********{api_key[-6:]}: {e.status_code} - {str(e)}"
                )
                raise HTTPException(status_code=e.status_code, detail=str(e))

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in [429, 503]:
                rate_limited_count += 1
                logger.warning(
                    f"HTTP {status} error with key **********{api_key[-6:]} ({rate_limited_count}/{total_keys} keys exhausted): {e.response.text}"
                )
                last_exception = e

                if attempt < total_keys - 1:
                    logger.info(
                        f"Waiting {THROTTLE_INTERVAL}s before trying next API key..."
                    )
                    await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            else:
                logger.error(
                    f"Non-recoverable HTTP error with key **********{api_key[-6:]}: {status} - {e.response.text}"
                )
                raise HTTPException(status_code=status, detail=e.response.text)

        except Exception as e:
            logger.error(
                f"Unexpected error with key **********{api_key[-6:]}: {str(e)}"
            )
            last_exception = e

            if attempt < total_keys - 1:
                logger.info(
                    f"Waiting {THROTTLE_INTERVAL}s before trying next API key..."
                )
                await asyncio.sleep(THROTTLE_INTERVAL)
            continue

    # 所有 key 都失败了
    logger.error(
        f"All {total_keys} API keys exhausted. Rate limited: {rate_limited_count}"
    )

    if isinstance(last_exception, openai.RateLimitError):
        raise HTTPException(
            status_code=429,
            detail=f"All {total_keys} API keys are rate limited. Please try again later.",
        )
    elif isinstance(last_exception, openai.APIStatusError):
        if last_exception.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail=f"All {total_keys} API keys are rate limited. Please try again later.",
            )
        elif last_exception.status_code == 503:
            raise HTTPException(
                status_code=503, detail="Model is overloaded. Try again later."
            )
    elif isinstance(last_exception, httpx.HTTPStatusError):
        status = last_exception.response.status_code
        if status == 429:
            raise HTTPException(
                status_code=429,
                detail=f"All {total_keys} API keys are rate limited. Please try again later.",
            )
        elif status == 503:
            raise HTTPException(
                status_code=503, detail="Model is overloaded. Try again later."
            )

    raise HTTPException(status_code=500, detail="All provider attempts failed.")


# ===== 流式请求的重试处理 =====
async def call_streaming_with_retry(func, *args, **kwargs):
    keys = await key_manager.get_keys_for_retry()
    rate_limited_count = 0
    total_keys = len(keys)

    for attempt, api_key in enumerate(keys):
        logger.info(
            f"Using API key (stream, attempt {attempt + 1}/{total_keys}): **********{api_key[-6:]}"
        )

        try:
            # 请求前间隔控制
            await key_manager.enforce_throttle(api_key, is_streaming=True)

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=config.settings.BASE_URL,
                max_retries=0,
            )
            result = await func(client, *args, **kwargs)
            logger.info(
                f"Streaming request successful with key **********{api_key[-6:]}"
            )

            return result, api_key

        except openai.RateLimitError as e:
            rate_limited_count += 1
            logger.warning(
                f"Rate limit error in streaming with key **********{api_key[-6:]} ({rate_limited_count}/{total_keys} keys exhausted): {str(e)}"
            )

            if attempt < total_keys - 1:
                logger.info(
                    f"Waiting {THROTTLE_INTERVAL}s before trying next streaming API key..."
                )
                await asyncio.sleep(THROTTLE_INTERVAL)
            continue

        except openai.APIStatusError as e:
            if e.status_code in [429, 503]:
                rate_limited_count += 1
                logger.warning(
                    f"API status {e.status_code} in streaming with key **********{api_key[-6:]} ({rate_limited_count}/{total_keys} keys exhausted): {str(e)}"
                )

                if attempt < total_keys - 1:
                    logger.info(
                        f"Waiting {THROTTLE_INTERVAL}s before trying next streaming API key..."
                    )
                    await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            else:
                logger.error(
                    f"Non-recoverable API error in streaming with key **********{api_key[-6:]}: {e.status_code} - {str(e)}"
                )
                raise HTTPException(status_code=e.status_code, detail=str(e))

        except httpx.HTTPStatusError as e:
            if e.response.status_code in [429, 503]:
                rate_limited_count += 1
                logger.warning(
                    f"HTTP {e.response.status_code} error in streaming with key **********{api_key[-6:]} ({rate_limited_count}/{total_keys} keys exhausted)"
                )

                if attempt < total_keys - 1:
                    logger.info(
                        f"Waiting {THROTTLE_INTERVAL}s before trying next streaming API key..."
                    )
                    await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            else:
                logger.error(
                    f"Non-recoverable HTTP error in streaming with key **********{api_key[-6:]}: {e.response.status_code}"
                )
                raise HTTPException(
                    status_code=e.response.status_code, detail=e.response.text
                )

        except Exception as e:
            logger.error(
                f"Unexpected streaming error with key **********{api_key[-6:]}: {str(e)}"
            )

            if attempt < total_keys - 1:
                logger.info(
                    f"Waiting {THROTTLE_INTERVAL}s before trying next streaming API key..."
                )
                await asyncio.sleep(THROTTLE_INTERVAL)
            continue

    # 所有 key 都失败
    logger.error(
        f"All {total_keys} API keys exhausted for streaming. Rate limited: {rate_limited_count}"
    )
    raise HTTPException(
        status_code=429,
        detail=f"All {total_keys} API keys failed for streaming request. Please try again later.",
    )


# ===== 模型列表 =====
@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    await verify_authorization(authorization)

    async def list_func(client):
        return await client.models.list()

    return await call_with_retry(list_func)


# ===== Chat Completions =====
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, authorization: str = Header(None)):
    await verify_authorization(authorization)

    # 构建 tools 参数（仅在有 tools 时传递）
    tool_kwargs = {}
    if request.tools:
        tool_kwargs["tools"] = request.tools
        tool_kwargs["tool_choice"] = request.tool_choice

    if request.stream:

        async def stream_func(client):
            return await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                **tool_kwargs,
            )

        try:
            response, used_api_key = await call_streaming_with_retry(stream_func)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Streaming setup error: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected streaming error.")

        async def generate():
            try:
                async for chunk in response:
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                error_payload = json.dumps(
                    {
                        "error": {
                            "message": f"Streaming interrupted: {str(e)}",
                            "type": "server_error",
                        }
                    }
                )
                yield f"data: {error_payload}\n\n"
            finally:
                # 流式请求完全结束后记录时间
                yield "data: [DONE]\n\n"
                logger.info(
                    f"Streaming completely finished with key **********{used_api_key[-6:]}"
                )
                await key_manager.record_request_end(used_api_key, is_streaming=True)

        return StreamingResponse(content=generate(), media_type="text/event-stream")

    else:

        async def chat_func(client):
            return await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
                **tool_kwargs,
            )

        return await call_with_retry(chat_func)


# ===== Embeddings =====
@app.post("/v1/embeddings")
async def embedding(request: EmbeddingRequest, authorization: str = Header(None)):
    await verify_authorization(authorization)

    async def emb_func(client):
        return await client.embeddings.create(
            input=request.input, model=request.model
        )

    return await call_with_retry(emb_func)


# ===== 健康检查 =====
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


# ===== 启动入口 =====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

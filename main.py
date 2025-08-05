from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import logging
from itertools import cycle
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

# ===== API Key 配置 =====
API_KEYS = config.settings.API_KEYS
key_cycle = cycle(API_KEYS)
key_lock = asyncio.Lock()
THROTTLE_INTERVAL = config.settings.THROTTLE_INTERVAL
proxy_queue = asyncio.Queue()

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
    user: Optional[str]
    response_format: Optional[str] = "float"

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

# ===== 节流处理器 =====
@app.on_event("startup")
async def start_throttle_worker():
    asyncio.create_task(throttle_worker())

async def throttle_worker():
    while True:
        request_func, params, future = await proxy_queue.get()
        try:
            result = await request_func(*params)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

async def proxy_with_throttle(request_func, *params):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await proxy_queue.put((request_func, params, future))
    return await future

# ===== Retry 封装 =====
async def call_with_retry(func, *args, **kwargs):
    tried_keys = set()
    last_exception = None
    for _ in range(len(API_KEYS)):
        async with key_lock:
            api_key = next(key_cycle)
            logger.info(f"Using API key: **********{api_key[-6:]}")
        if api_key in tried_keys:
            continue
        tried_keys.add(api_key)
        try:
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=config.settings.BASE_URL,
                max_retries=0,
            )
            result = await func(client, *args, **kwargs)
            await asyncio.sleep(THROTTLE_INTERVAL)
            return result
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.warning(f"HTTP {status} error: {e.response.text}")
            last_exception = e
            if status in [429, 503]:
                await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            raise HTTPException(status_code=status, detail=e.response.text)
        except Exception as e:
            logger.error(f"Unexpected provider error: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected provider error.")

    # 所有 key 都失败
    if isinstance(last_exception, httpx.HTTPStatusError):
        status = last_exception.response.status_code
        if status == 429:
            raise HTTPException(status_code=429, detail="All API keys are rate limited.")
        if status == 503:
            raise HTTPException(status_code=503, detail="Model is overloaded. Try again later.")
    raise HTTPException(status_code=500, detail="All provider attempts failed.")

# ===== 模型列表 =====
@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    await verify_authorization(authorization)
    async def list_func(client):
        return await client.models.list()
    return await proxy_with_throttle(call_with_retry, list_func)

# ===== Chat Completions =====
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, authorization: str = Header(None)):
    await verify_authorization(authorization)
    if request.stream:
        async with key_lock:
            api_key = next(key_cycle)
        logger.info(f"Using API key (stream): **********{api_key[-6:]}")
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=config.settings.BASE_URL,
            max_retries=0
        )
        try:
            response = await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                raise HTTPException(status_code=503, detail="Model is overloaded.")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected streaming error.")

        async def generate():
            try:
                async for chunk in response:
                    yield f"data: {chunk.model_dump_json()}\n\n"
            finally:
                # 流结束标识
                yield "data: [DONE]\n\n"

        return StreamingResponse(content=generate(), media_type="text/event-stream")

    else:
        async def chat_func(client):
            return await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )
        return await proxy_with_throttle(call_with_retry, chat_func)

# ===== Embeddings =====
@app.post("/v1/embeddings")
async def embedding(request: EmbeddingRequest, authorization: str = Header(None)):
    await verify_authorization(authorization)
    async def emb_func(client):
        return await client.embeddings.create(
            input=request.input,
            model=request.model
        )
    return await proxy_with_throttle(call_with_retry, emb_func)

# ===== 健康检查 =====
@app.get("/health")
async def health_check(authorization: str = Header(None)):
    await verify_authorization(authorization)
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

# ===== 启动入口 =====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

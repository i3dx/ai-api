from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
from typing import List, Optional
import logging
from itertools import cycle
import asyncio
import uvicorn
from app import config
from openai import RateLimitError
import inspect

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API密钥配置
API_KEYS = config.settings.API_KEYS
key_cycle = cycle(API_KEYS)
key_lock = asyncio.Lock()
THROTTLE_INTERVAL = config.settings.THROTTLE_INTERVAL
proxy_queue = asyncio.Queue()


class ChatRequest(BaseModel):
    messages: List[dict]
    model: str = "llama-3.2-90b-text-preview"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8000
    stream: Optional[bool] = False
    tools: Optional[List[dict]] = []
    tool_choice: Optional[str] = "auto"


class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = "text-embedding-004"
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = 1536
    user: Optional[str]
    response_format: Optional[str] = "float"


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


@app.on_event("startup")
async def start_throttle_worker():
    asyncio.create_task(throttle_worker())


async def throttle_worker():
    while True:
        item = await proxy_queue.get()
        request_func, params, future = item
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


async def call_openai_with_retry(func, *args, **kwargs):
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
                max_retries=0
            )
            result = await func(client, *args, **kwargs)
            await asyncio.sleep(THROTTLE_INTERVAL)
            return result
        except RateLimitError as e:
            logger.warning(f"API key **********{api_key[-6:]} rate limited. Switching to next key.")
            last_exception = e
            await asyncio.sleep(THROTTLE_INTERVAL)
            continue
        except Exception as e:
            logger.error(f"OpenAI call error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    logger.error("All API keys are rate limited.")
    raise HTTPException(status_code=429, detail="All API keys are rate limited")


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    await verify_authorization(authorization)

    async def list_func(client):
        return await client.models.list()

    async def do_request():
        return await call_openai_with_retry(list_func)

    return await proxy_with_throttle(do_request)


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, authorization: str = Header(None)):
    await verify_authorization(authorization)

    if request.stream:
        # 流式请求：直接使用 AsyncOpenAI，不走节流队列
        async with key_lock:
            api_key = next(key_cycle)
        logger.info(f"Using API key (stream): **********{api_key[-6:]}")

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=config.settings.BASE_URL,
            max_retries=0
        )

        response = await client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )

        async def generate():
            async for chunk in response:
                yield f"data: {chunk.model_dump_json()}\n\n"

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

        async def do_request():
            return await call_openai_with_retry(chat_func)

        return await proxy_with_throttle(do_request)


@app.post("/v1/embeddings")
async def embedding(request: EmbeddingRequest, authorization: str = Header(None)):
    await verify_authorization(authorization)

    async def emb_func(client):
        return await client.embeddings.create(input=request.input, model=request.model)

    async def do_request():
        return await call_openai_with_retry(emb_func)

    return await proxy_with_throttle(do_request)


@app.get("/health")
async def health_check(authorization: str = Header(None)):
    await verify_authorization(authorization)
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

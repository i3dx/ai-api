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

# ===== 全局请求间隔控制 =====
last_request_end_time = {}  # 记录每个 API key 的最后请求结束时间
global_last_request_end_time = 0  # 记录全局最后请求结束时间

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

# ===== 统一的请求间隔控制 =====
async def enforce_throttle_before_request(api_key: str, is_streaming: bool = False):
    """在请求开始前检查间隔"""
    global global_last_request_end_time
    current_time = asyncio.get_event_loop().time()
    
    # 检查全局间隔（基于上次请求结束时间）
    time_since_last_global = current_time - global_last_request_end_time
    if time_since_last_global < THROTTLE_INTERVAL:
        wait_time = THROTTLE_INTERVAL - time_since_last_global
        request_type = "streaming" if is_streaming else "regular"
        logger.info(f"Global throttle: waiting {wait_time:.2f}s before {request_type} request with key **********{api_key[-6:]}")
        await asyncio.sleep(wait_time)
    
    # 检查单个 key 的间隔（基于该key上次请求结束时间）
    if api_key in last_request_end_time:
        time_since_last_key = current_time - last_request_end_time[api_key]
        if time_since_last_key < THROTTLE_INTERVAL:
            wait_time = THROTTLE_INTERVAL - time_since_last_key
            request_type = "streaming" if is_streaming else "regular"
            logger.info(f"Key throttle: waiting {wait_time:.2f}s for {request_type} request with key **********{api_key[-6:]}")
            await asyncio.sleep(wait_time)

async def record_request_end(api_key: str, is_streaming: bool = False):
    """在请求结束后记录时间
    对于流式请求：在流开始返回时记录（不等待完全结束）
    对于普通请求：在完整响应后记录
    """
    global global_last_request_end_time
    current_time = asyncio.get_event_loop().time()
    last_request_end_time[api_key] = current_time
    global_last_request_end_time = current_time
    
    request_type = "streaming" if is_streaming else "regular"
    logger.debug(f"Recorded {request_type} request end time for key **********{api_key[-6:]}")

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

# ===== 修复后的 Retry 封装 =====
async def call_with_retry(func, *args, **kwargs):
    tried_keys = set()
    last_exception = None
    rate_limited_count = 0
    
    for attempt in range(len(API_KEYS)):
        async with key_lock:
            api_key = next(key_cycle)
            logger.info(f"Using API key (attempt {attempt + 1}/{len(API_KEYS)}): **********{api_key[-6:]}")
        
        if api_key in tried_keys:
            logger.warning(f"Key **********{api_key[-6:]} already tried, skipping")
            continue
        tried_keys.add(api_key)
        
        try:
            # 请求前间隔控制
            await enforce_throttle_before_request(api_key, is_streaming=False)
            
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=config.settings.BASE_URL,
                max_retries=0,
            )
            result = await func(client, *args, **kwargs)
            logger.info(f"Request successful with key **********{api_key[-6:]}")
            
            # 普通请求结束后记录时间
            await record_request_end(api_key, is_streaming=False)
            return result
            
        except openai.RateLimitError as e:
            rate_limited_count += 1
            logger.warning(f"Rate limit error with key **********{api_key[-6:]} ({rate_limited_count}/{len(API_KEYS)} keys exhausted): {str(e)}")
            last_exception = e
            
            # 如果还有其他 key 可以尝试，等待一段时间再重试
            if attempt < len(API_KEYS) - 1:
                logger.info(f"Waiting {THROTTLE_INTERVAL}s before trying next API key...")
                await asyncio.sleep(THROTTLE_INTERVAL)
            continue
            
        except openai.APIStatusError as e:
            if e.status_code in [429, 503]:
                rate_limited_count += 1
                logger.warning(f"API status {e.status_code} with key **********{api_key[-6:]} ({rate_limited_count}/{len(API_KEYS)} keys exhausted): {str(e)}")
                last_exception = e
                
                # 如果还有其他 key 可以尝试，等待一段时间再重试
                if attempt < len(API_KEYS) - 1:
                    logger.info(f"Waiting {THROTTLE_INTERVAL}s before trying next API key...")
                    await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            else:
                logger.error(f"Non-recoverable API error with key **********{api_key[-6:]}: {e.status_code} - {str(e)}")
                raise HTTPException(status_code=e.status_code, detail=str(e))
                
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in [429, 503]:
                rate_limited_count += 1
                logger.warning(f"HTTP {status} error with key **********{api_key[-6:]} ({rate_limited_count}/{len(API_KEYS)} keys exhausted): {e.response.text}")
                last_exception = e
                
                # 如果还有其他 key 可以尝试，等待一段时间再重试
                if attempt < len(API_KEYS) - 1:
                    logger.info(f"Waiting {THROTTLE_INTERVAL}s before trying next API key...")
                    await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            else:
                logger.error(f"Non-recoverable HTTP error with key **********{api_key[-6:]}: {status} - {e.response.text}")
                raise HTTPException(status_code=status, detail=e.response.text)
                
        except Exception as e:
            logger.error(f"Unexpected error with key **********{api_key[-6:]}: {str(e)}")
            last_exception = e
            
            # 对于未知错误也添加间隔
            if attempt < len(API_KEYS) - 1:
                logger.info(f"Waiting {THROTTLE_INTERVAL}s before trying next API key...")
                await asyncio.sleep(THROTTLE_INTERVAL)
            continue

    # 所有 key 都失败了
    logger.error(f"All {len(API_KEYS)} API keys exhausted. Rate limited: {rate_limited_count}")
    
    if isinstance(last_exception, openai.RateLimitError):
        raise HTTPException(status_code=429, detail=f"All {len(API_KEYS)} API keys are rate limited. Please try again later.")
    elif isinstance(last_exception, openai.APIStatusError):
        if last_exception.status_code == 429:
            raise HTTPException(status_code=429, detail=f"All {len(API_KEYS)} API keys are rate limited. Please try again later.")
        elif last_exception.status_code == 503:
            raise HTTPException(status_code=503, detail="Model is overloaded. Try again later.")
    elif isinstance(last_exception, httpx.HTTPStatusError):
        status = last_exception.response.status_code
        if status == 429:
            raise HTTPException(status_code=429, detail=f"All {len(API_KEYS)} API keys are rate limited. Please try again later.")
        elif status == 503:
            raise HTTPException(status_code=503, detail="Model is overloaded. Try again later.")
            
    raise HTTPException(status_code=500, detail="All provider attempts failed.")

# ===== 修复后的流式请求的重试处理 =====
async def call_streaming_with_retry(func, *args, **kwargs):
    tried_keys = set()
    rate_limited_count = 0
    used_api_key = None
    
    for attempt in range(len(API_KEYS)):
        async with key_lock:
            api_key = next(key_cycle)
            logger.info(f"Using API key (stream, attempt {attempt + 1}/{len(API_KEYS)}): **********{api_key[-6:]}")
            
        if api_key in tried_keys:
            logger.warning(f"Streaming key **********{api_key[-6:]} already tried, skipping")
            continue
        tried_keys.add(api_key)
        
        try:
            # 请求前间隔控制
            await enforce_throttle_before_request(api_key, is_streaming=True)
            
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=config.settings.BASE_URL,
                max_retries=0
            )
            result = await func(client, *args, **kwargs)
            logger.info(f"Streaming request successful with key **********{api_key[-6:]}")
            
            # 记录使用的API key，但不立即记录结束时间
            used_api_key = api_key
            # 将API key 信息附加到响应对象上
            if hasattr(result, '__dict__'):
                result._api_key_used = api_key
            
            return result, used_api_key
            
        except openai.RateLimitError as e:
            rate_limited_count += 1
            logger.warning(f"Rate limit error in streaming with key **********{api_key[-6:]} ({rate_limited_count}/{len(API_KEYS)} keys exhausted): {str(e)}")
            
            # 如果还有其他 key 可以尝试，等待一段时间再重试
            if attempt < len(API_KEYS) - 1:
                logger.info(f"Waiting {THROTTLE_INTERVAL}s before trying next streaming API key...")
                await asyncio.sleep(THROTTLE_INTERVAL)
            continue
            
        except openai.APIStatusError as e:
            if e.status_code in [429, 503]:
                rate_limited_count += 1
                logger.warning(f"API status {e.status_code} in streaming with key **********{api_key[-6:]} ({rate_limited_count}/{len(API_KEYS)} keys exhausted): {str(e)}")
                
                # 如果还有其他 key 可以尝试，等待一段时间再重试
                if attempt < len(API_KEYS) - 1:
                    logger.info(f"Waiting {THROTTLE_INTERVAL}s before trying next streaming API key...")
                    await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            else:
                logger.error(f"Non-recoverable API error in streaming with key **********{api_key[-6:]}: {e.status_code} - {str(e)}")
                raise HTTPException(status_code=e.status_code, detail=str(e))
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [429, 503]:
                rate_limited_count += 1
                logger.warning(f"HTTP {e.response.status_code} error in streaming with key **********{api_key[-6:]} ({rate_limited_count}/{len(API_KEYS)} keys exhausted)")
                
                # 如果还有其他 key 可以尝试，等待一段时间再重试
                if attempt < len(API_KEYS) - 1:
                    logger.info(f"Waiting {THROTTLE_INTERVAL}s before trying next streaming API key...")
                    await asyncio.sleep(THROTTLE_INTERVAL)
                continue
            else:
                logger.error(f"Non-recoverable HTTP error in streaming with key **********{api_key[-6:]}: {e.response.status_code}")
                raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
                
        except Exception as e:
            logger.error(f"Unexpected streaming error with key **********{api_key[-6:]}: {str(e)}")
            
            # 对于未知错误也添加间隔
            if attempt < len(API_KEYS) - 1:
                logger.info(f"Waiting {THROTTLE_INTERVAL}s before trying next streaming API key...")
                await asyncio.sleep(THROTTLE_INTERVAL)
            continue
    
    # 所有 key 都失败
    logger.error(f"All {len(API_KEYS)} API keys exhausted for streaming. Rate limited: {rate_limited_count}")
    raise HTTPException(status_code=429, detail=f"All {len(API_KEYS)} API keys failed for streaming request. Please try again later.")

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
        async def stream_func(client):
            return await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
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
                yield f"data: {{'error': 'Streaming interrupted: {str(e)}'}}\n\n"
            finally:
                # 流式请求完全结束后记录时间
                yield "data: [DONE]\n\n"
                logger.info(f"Streaming completely finished with key **********{used_api_key[-6:]}")
                await record_request_end(used_api_key, is_streaming=True)

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

import asyncio
import functools
import time
import logging

logger = logging.getLogger(__name__)

def retry_async(retries=3, delay=1.0, backoff=2.0, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            _retries, _delay = retries, delay
            while _retries > 0:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"[retry_async] 捕捉到錯誤：{e}，重試 {_retries} 次...")
                    _retries -= 1
                    if _retries == 0:
                        logger.error(f"[retry_async] 已達最大重試次數，失敗拋出錯誤")
                        raise
                    await asyncio.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator

def retry_sync(retries=3, delay=1.0, backoff=2.0, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _retries, _delay = retries, delay
            while _retries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"[retry_sync] 捕捉到錯誤：{e}，重試 {_retries} 次...")
                    _retries -= 1
                    if _retries == 0:
                        logger.error(f"[retry_sync] 已達最大重試次數，失敗拋出錯誤")
                        raise
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator

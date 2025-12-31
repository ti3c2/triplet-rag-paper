import asyncio
import hashlib
import logging
from functools import lru_cache, wraps
from textwrap import fill
from typing import Awaitable, List, Optional

import openai
import tiktoken
import tqdm
from openai.types.chat import ChatCompletion
from openai.types.create_embedding_response import CreateEmbeddingResponse
from transformers import AutoTokenizer

from .settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


def retry_on_exception_async(
    func,
    max_retries: int = settings.openai_max_retries,
    timeout: float = settings.openai_timeout,
):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        for i in range(max_retries):
            try:
                if i > 0:
                    logger.warning(f"Retrying {func.__name__} for the {i+1} time")
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error executing {func.__name__}: {str(e)}")
                await asyncio.sleep(timeout)
        return None

    return wrapper


@retry_on_exception_async
async def create_chat_completion(
    client: openai.AsyncOpenAI,
    parse: bool = False,
    *args,
    **kwargs,
) -> ChatCompletion:
    if messages := kwargs.get("messages", []):
        log_method = logger.info if settings.log_chat_completion_input else logger.debug
        log_method(
            f"Chat completion input with t={kwargs.get('temperature', None)}:\n"
            + "\n".join(
                [f"{message['role']}: {message['content']}" for message in messages]
            )
        )
    func = (
        client.chat.completions.create
        if not parse
        else client.beta.chat.completions.parse
    )
    out = await func(*args, **kwargs)
    if out is None:
        logger.error("Chat completion returned None")
        return None
    if parse:
        logger.debug(
            f"Chat completion output:\n{out.choices[0].message.parsed.model_dump_json(indent=2)}"
        )
    else:
        logger.debug(f"Chat completion output:\n{out.choices[0].message.content}")
    logger.debug(f"Chat completion usage:{out.usage.model_dump_json()}")
    return out


@lru_cache
def _get_tokenizer(emb_model: str) -> AutoTokenizer:
    logger.info("Loading tokenizer for %s", emb_model)
    return AutoTokenizer.from_pretrained(
        emb_model,
        use_fast=True,
    )


def crop_text_to_max_tokens(
    text: str,
    emb_model: str = settings.openai_embedding_model,
    max_tokens: Optional[int] = settings.openai_embedding_max_tokens,
) -> str:
    if "openai" in settings.openai_api_base_embedding:
        tok = tiktoken.encoding_for_model(emb_model)
        max_tokens = max_tokens or 8192
        n_special_tokens = len(tok.special_tokens_set)
        ids = tok.encode(text)
        max_tokens -= n_special_tokens
        if len(ids) > max_tokens:
            logger.info("Cropping text from %d to %d tokens", len(ids), max_tokens)
            ids = ids[:max_tokens]
        return tok.decode(ids)
    else:
        tok = _get_tokenizer(emb_model)
        max_tokens = max_tokens or tok.model_max_length
        # NOTE: Hard coded because tokenizer shows 512 tokens but it's 256
        if emb_model == "sentence-transformers/all-MiniLM-L6-v2":
            max_tokens = 256
        n_special_tokens = tok.num_special_tokens_to_add(pair=False)
        ids = tok.encode(
            text,
            truncation=True,
            add_special_tokens=False,
        )
        max_tokens -= n_special_tokens
        if len(ids) > max_tokens:
            logger.info("Cropping text from %d to %d tokens", len(ids), max_tokens)
            ids = ids[:max_tokens]
        return tok.decode(ids, skip_special_tokens=True)


@retry_on_exception_async
async def create_embeddings(
    client: openai.AsyncOpenAI,
    *args,
    **kwargs,
) -> CreateEmbeddingResponse:
    return await client.embeddings.create(*args, **kwargs)


async def semaphore_task(semaphore, task, pbar: Optional[tqdm.tqdm] = None):
    """Execute a task with a semaphore."""
    async with semaphore:
        result = await task
        if pbar:
            pbar.update(1)
        return result


async def execute_with_semaphore(
    tasks: List[Awaitable],
    max_concurrency: int = settings.openai_chat_completion_max_concurrency,
    show_progress: bool = True,
):
    """Execute tasks with limited concurrency using a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrency)
    logger.info(f"Executing {len(tasks)} tasks with max concurrency {max_concurrency}")
    pbar = (
        tqdm.tqdm(total=len(tasks), desc="Executing tasks") if show_progress else None
    )
    results = await asyncio.gather(
        *[semaphore_task(semaphore, task, pbar) for task in tasks]
    )
    if pbar:
        pbar.close()
    return results


def fill_text(text: str, width: int = 100, indent: str = "") -> str:
    paras = []
    for line in text.split("\n"):
        paras.append(
            fill(
                line,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
            )
        )
    return "\n\n".join(paras)


def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

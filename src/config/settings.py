import datetime as dt
import logging
from pathlib import Path
from typing import List, Literal, Optional, Union

import httpx
import openai
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .prompts import Prompt, PromptSettings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dataset(BaseModel):
    NATURAL_QUESTIONS: str = Field(default="natural-questions", exclude=True)
    SQUAD: str = Field(default="squad", exclude=True)
    MULTI_HOP_RAG: str = Field(default="multihoprag", exclude=True)
    TECH_QA: str = Field(default="tech-qa", exclude=True)
    TREC_COVID: str = Field(default="trec-covid", exclude=True)


class ProjectSettings(
    BaseSettings,
    PromptSettings,  # enable access to prompt_xxx
    Dataset,  # enable access to Dataset.xxx to avoid typos
):
    # Paths
    path_root: Path = Path(__file__).parents[2]
    path_data: Path = path_root / "data"
    path_data_datasets: Path = path_data / "datasets"
    path_data_sql: Path = path_data / "sql" / "store.db"
    path_eval: Path = path_data / "eval"
    path_logs: Path = path_root / "logs"
    path_logs_file: Path = (
        path_logs / f"{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    path_config: Path = path_root / "config"
    path_data_ragas_queries: Path = path_data / "ragas_queries"

    # SQL
    sqlite_database_url: str = f"sqlite+aiosqlite:///{path_data_sql}"
    sql_emb_dtype: str = "float32"
    use_sqlite: bool = True
    sql_pool_size: int = 20  # Base connection pool size
    sql_max_overflow: int = 30  # Additional connections on demand (total: pool_size + max_overflow)
    sql_pool_timeout: int = 60  # Seconds to wait for a connection

    postgres_user: str = "raguser"
    postgres_password: str = "ragpass"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 9700
    postgres_db: str = "ragdb"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_base_embedding: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_proxy_url: Optional[str] = None
    openai_use_proxy: bool = False
    openai_max_retries: int = 3
    openai_timeout: float = 0.5
    openai_chat_completion_max_concurrency: int = 10
    openai_embedding_batch_size: int = 200
    openai_embedding_max_tokens: Optional[int] = None  # Use max available for model

    # QuOTE Settings
    question_generation_mode: str = "q"  # q,qa separated by comma
    qa_separator: str = "\n"
    qa_sequential_generation: str = "single"  # single, sequential, sequential_batch
    question_generation_temperature: float = 0.7
    answer_generation_temperature: float = 0.1
    question_generation_max_completion_tokens: int = 8192
    question_generation_model: str = "gpt-4o-mini"
    question_generation_docs_commited: int = 1
    question_generation_prompt_name: Optional[str] = None

    # RAG Settings
    rag_dataset: str = Dataset().NATURAL_QUESTIONS
    rag_max_rows: int = 10000
    rag_deduplication_factor: int = 1
    rag_retrieval_sources: List[Literal["chunk", "query"]] = ["chunk", "query"]
    rag_retrieval_mode: str = "knn"
    rag_retrieval_ks: List[int] = [1, 5, 20]
    rag_retrieval_metric: str = "cosine"
    rag_retrieval_algorithm: str = "brute"

    rag_generate_answer: bool = True
    rag_add_chunk_to_query: bool = False
    rag_use_chunk_for_qa_context: bool = False
    rag_generation_model: str = "gpt-4o-mini"
    rag_generation_model_api_base: Optional[str] = None
    rag_generation_temperature: float = 0.1
    rag_generation_max_tokens: int = 1000
    rag_batch_size: int = 50
    rag_readonly_eval: bool = True
    # Triplet/augmentation flags
    rag_use_triplet_contexts: bool = False
    rag_max_triplet_contexts: int = -1
    rag_persist_triplets: bool = False
    rag_triplet_json_path: Optional[str] = None
    rag_retrieval_query_origins: List[str] = ["dataset"]  # dataset, quote, eval
    rag_triplet_generate_answer: bool = True
    rag_n_triplet_contexts: int = 5

    rag_write_json: bool = True
    rag_write_txt: bool = True
    rag_write_spreadsheet: bool = True

    # Spreadsheet
    spreadsheet_id: Optional[str] = None
    spreadsheet_gid: Optional[str] = None

    # Ragas Settings
    ragas_enabled: bool = True
    eval_llm: str = "gpt-4o-mini"
    eval_llm_api_base: Optional[str] = None
    eval_embedding_model: str = "text-embedding-3-small"
    eval_embedding_model_api_base: Optional[str] = None
    eval_context_ks: List[int] = rag_retrieval_ks
    eval_max_queries: Optional[int] = None
    eval_max_queries_ragas: Optional[int] = None
    eval_whitelist_origins: List[str] = ["eval"]
    ragas_metrics_init: List[str] | str = Field(
        # Parse to List[str] if string in ragas_metrics attribute
        # Using actual ragas column names to avoid mapping issues
        default=[
            # Generation metrics (using ragas naming)
            "faithfulness",
            "answer_relevancy",
            "nv_accuracy",
            "nv_context_relevance",
            "nv_response_groundedness",
            "summary_score",
            # Classical metrics for generation
            "rouge_score",
            "bleu_score",
            "non_llm_string_similarity",
            "string_present",
            "exact_match",
            "semantic_similarity",
            "factual_correctness",
            # Retrieval metrics
            "context_precision",
            "context_recall",
            "context_entity_recall",
            "noise_sensitivity",
        ],
        alias="ragas_metrics",
        exclude=True,
    )
    ragas_max_workers: int = 16
    ragas_batch_size: Optional[int] = None
    ragas_max_retries: int = 3
    ragas_timeout: float = 30.0

    # Logging
    log_level: int = logging.INFO
    log_chat_completion_input: bool = False
    log_retrieved_items: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def postgres_database_url(self) -> str:
        return f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_database_url_async(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @computed_field
    @property
    def ragas_metrics(self) -> List[str]:
        return (
            self.ragas_metrics_init.split(",")
            if isinstance(self.ragas_metrics_init, str)
            else self.ragas_metrics_init
        )

    @computed_field
    @property
    def question_generation_default_prompt(self) -> Prompt:
        return self.get_prompt(self.question_generation_prompt_name)

    @property
    def rag_retrieval_k(self) -> int:
        return max(self.rag_retrieval_ks)

    @property
    def sql_database_url(self) -> str:
        return (
            self.sqlite_database_url
            if self.use_sqlite
            else self.postgres_database_url_async
        )

    def get_openai_client(
        self,
        base_url: Optional[str] = None,
        embedding: bool = False,
    ) -> openai.AsyncOpenAI:
        base_url = base_url or (
            self.openai_api_base if not embedding else self.openai_api_base_embedding
        )
        return openai.AsyncOpenAI(
            api_key=self.openai_api_key,
            base_url=base_url,
            http_client=(
                httpx.AsyncClient(proxy=self.openai_proxy_url)
                if self.openai_use_proxy
                else None
            ),
        )

    async def ping_model(self, model: str) -> bool:
        for base_url in set(
            [
                self.openai_api_base,
                self.openai_api_base_embedding,
                self.rag_generation_model_api_base,
                self.eval_llm_api_base,
                self.eval_embedding_model_api_base,
            ]
        ):
            client = self.get_openai_client(
                base_url=base_url,
                embedding=(
                    base_url
                    in [
                        self.openai_api_base_embedding,
                        self.eval_embedding_model_api_base,
                    ]
                ),
            )
            try:
                logger.info(f"Pinging model {model} on {base_url}")
                models = await client.models.list()
                if any(model == m.id for m in models.data):
                    logger.info(f"Model {model} is available on {base_url}")
                    return True
                else:
                    logger.error(f"Model {model} is not available on {base_url}")
            except Exception as e:
                logger.error(f"Error pinging model {model} on {base_url}: {e}")
        return False

    def find_file(
        self,
        fname: Union[str, Path],
        base: Optional[Path] = None,
    ) -> Optional[Path]:
        if (fpath := Path(fname)).exists():
            return fpath
        base = base or self.path_data
        files = list(base.rglob(f"*{fname}*", case_sensitive=False))
        if len(files) > 1:
            logger.warning(f"Multiple files found for '{fname}': {[f.relative_to(self.path_root) for f in files]}")  # fmt:skip
        for path in files:
            if path.is_file():
                logger.info(f"Found file for '{fname}': {path.relative_to(self.path_root)}")  # fmt: skip
                return path
        logger.error(f"No file found for '{fname}'")
        return None


settings = ProjectSettings()

# Log both to console and file
settings.path_logs.mkdir(parents=True, exist_ok=True)
handlers = [
    logging.StreamHandler(),
    logging.FileHandler(settings.path_logs_file),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d",
    handlers=handlers,
)

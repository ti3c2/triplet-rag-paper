# Enhancing Small Language Models in RAG via Triplet-Guided Contextual Distillation

## QuickStart

Install dependencies with uv:
```bash
uv sync
```

### Configuration
Create a `.env` file in the project root and set it up according to `.env.example`

## Structure

```
config/
└── google_config.json   # Google API credentials for spreadsheet integration (place here)
data/
├── datasets             # Huggingface datasets
├── ragas_queries        # RAGAS CSV queries (place files here; any subfolders; search by filename)
├── eval                 # Evaluation results
│   ├── 20250804_220605_natural-questions_text-embedding-3-small_gpt-4o-mini_gpt-4o-mini
│   │   ├── 20250804_220605.json        # JSON dump of Evaluation Result
│   │   ├── 20250804_220605.txt         # Formatted Evaluation Result for reading
│   └───└── metrics_20250804_220605.csv # Metrics
├── sql/store.db         # SQLite database folder
src/
├── config/
│   ├── settings.py      # Project configuration and settings
│   ├── prompts.py       # Prompt templates for basic/complex modes
│   ├── spreadsheets.py  # Google Spreadsheet integration functions
│   └── utils.py         # Util functions used in different parts of code
├── etl/
│   └── etl_hf.py        # Data processing and embedding generation
├── rag/
│   ├── evaluate.py      # Pipeline for evaluation
│   ├── generation.py    # LLM response generation
│   ├── models.py        # Pydantic data models
│   ├── pipeline.py      # Main RAG pipeline implementation
│   ├── retrieval.py     # KNN retrieval implementations
└── store_rel/
    ├── entry.py         # Database connection management
    └── schema.py        # SQLAlchemy models and schema
```

## Configuration

The system is configured through environment variables that can be set in a `.env` file in the project root. All configuration options are defined in `src/config/settings.py`.

### Environment Variables

Create a `.env` file in the project root with the following variables:

#### Required Variables
```bash
# OpenAI API Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here
```

#### Path Configuration
```bash
# Paths (optional - defaults provided)
PATH_ROOT=/path/to/project/root
PATH_DATA=/path/to/data/directory
PATH_DATA_SQL=/path/to/sqlite/database.db
PATH_EVAL=/path/to/evaluation/results
PATH_LOGS=/path/to/logs/directory
```

#### Database Configuration
```bash
# SQL Database Settings (SQLite - default)
SQL_DATABASE_URL=sqlite:///data/sql/store.db
SQL_EMB_DTYPE=float32
USE_SQLITE=true  # Set to false to use PostgreSQL

# PostgreSQL Configuration (Optional - only needed if USE_SQLITE=false)
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpass
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=9700
POSTGRES_DB=ragdb
```

#### OpenAI API Settings
```bash
# OpenAI Configuration
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_BASE_EMBEDDING=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_PROXY_URL=http://proxy.example.com:8080  # Optional
OPENAI_USE_PROXY=false
OPENAI_MAX_RETRIES=3
OPENAI_TIMEOUT=0.5
OPENAI_CHAT_COMPLETION_MAX_CONCURRENCY=10
OPENAI_EMBEDDING_BATCH_SIZE=200
OPENAI_EMBEDDING_MAX_TOKENS=8191  # Optional - uses model default if not set
```

#### QuOTE Question Generation Settings
```bash
# Question Generation
PROMPT_MODE=basic  # Options: basic, complex
QUESTION_GENERATION_MODE=q  # Options: q, qa (comma-separated)
QA_SEPARATOR=\n
QUESTION_GENERATION_TEMPERATURE=0.0
QUESTION_GENERATION_MAX_COMPLETION_TOKENS=3000
QUESTION_GENERATION_MODEL=gpt-4o-mini
QUESTION_GENERATION_DOCS_COMMITED=1
```

#### RAG Pipeline Settings
```bash
# RAG Configuration
RAG_DATASET=natural-questions  # Options: natural-questions, squad, multihoprag
RAG_MAX_ROWS=10000
RAG_DEDUPLICATION_FACTOR=1
RAG_RETRIEVAL_SOURCES=["chunk", "query"]  # JSON array
RAG_RETRIEVAL_MODE=knn
RAG_RETRIEVAL_KS=[1, 5, 20]  # JSON array
RAG_RETRIEVAL_METRIC=cosine
RAG_RETRIEVAL_ALGORITHM=brute

# Answer Generation
RAG_GENERATE_ANSWER=true
RAG_GENERATION_MODEL=gpt-4o-mini
RAG_GENERATION_TEMPERATURE=0.1
RAG_GENERATION_MAX_TOKENS=1000
RAG_BATCH_SIZE=50
```

#### Triplet & Two-Stage RAG Settings
```bash
# Triplet-Augmented RAG Configuration
RAG_USE_TRIPLET_CONTEXTS=false           # Enable triplet context injection during RAG
RAG_TRIPLET_APPLY_FORMATTING=triplet     # Options: chunk, qa, triplet
RAG_MAX_TRIPLET_CONTEXTS=-1              # Max triplet contexts to inject (-1 for unlimited)
RAG_N_TRIPLET_CONTEXTS=5                 # Number of contexts to use per triplet
RAG_TRIPLET_GENERATE_ANSWER=true         # Generate answers when using triplets
RAG_PERSIST_TRIPLETS=false               # Persist triplets to DB during evaluation
RAG_TRIPLET_JSON_PATH=                   # Path to triplets JSON file (optional)
RAG_RETRIEVAL_QUERY_ORIGINS=["dataset"]  # Query origins for retrieval: dataset, quote, eval, two-stage
EVAL_WHITELIST_ORIGINS=["eval"]          # Origins allowed for evaluation queries
```

#### Logging Configuration
```bash
# Logging
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
```

#### RAGAS Evaluation Configuration
```bash
# RAGAS LLM-based Evaluation Settings
RAGAS_ENABLED=true  # Enable RAGAS evaluation for trcli eval run
EVAL_LLM=gpt-4o-mini
EVAL_LLM_API_BASE=https://api.openai.com/v1
EVAL_EMBEDDING_MODEL=text-embedding-3-small
EVAL_EMBEDDING_MODEL_API_BASE=https://api.openai.com/v1
# RAGAS_METRICS=[faithfulness,answer_relevancy,context_precision] # Don't set for calculating all metrics
RAGAS_MAX_WORKERS=100      # Concurrent workers for RAGAS evaluation
RAGAS_MAX_RETRIES=8        # Max retries for failed RAGAS evaluations
RAGAS_TIMEOUT=500          # Timeout for RAGAS operations (seconds)
# RAGAS_BATCH_SIZE=10      # Batch size for RAGAS evaluation. Don't set unless you really want it
EVAL_CONTEXT_KS=[1,5,20]   # JSON array of K values for context evaluation
```

#### Google Spreadsheet Configuration
```bash
# Spreadsheet Integration (Optional)
RAG_WRITE_SPREADSHEET=true  # Enable automatic writing to spreadsheet after evaluation
SPREADSHEET_ID=your_google_spreadsheet_id_here
SPREADSHEET_GID=your_sheet_name_or_numeric_gid
```

### Configuration Details

- **Prompt Modes**:
  - `basic`: Simple question generation prompts
  - `complex`: Complex question generation prompts with detailed instructions

- **Retrieval Sources**:
  - `chunk`: Retrieve only using document chunks
  - `query`: Retrieve using only generated questions
  - Both can be used together for hybrid retrieval

- **Question Generation Mode**
  - `q`: embed only queries
  - `qa`: append answer to the query and embed

- **Query Origins** (`RAG_RETRIEVAL_QUERY_ORIGINS`):
  - `dataset`: Original questions from the dataset (default)
  - `quote`: Questions generated by QuOTE pipeline
  - `eval`: Questions marked for evaluation ground-truth
  - `two-stage`: Questions used in two-stage RAG retrieval

- **Triplet Formatting** (`RAG_TRIPLET_APPLY_FORMATTING`):
  - `triplet`: Full triplet format with contexts, reference question, and reference answer
  - `chunk`: Only chunk contexts from triplets (no QA pairs)
  - `qa`: Only question-answer pairs from triplets (no chunk contexts)

- **RAGAS Evaluation**:
  - RAGAS provides LLM-based evaluation metrics for RAG systems
  - Includes generation metrics (faithfulness, answer relevancy), classical metrics (ROUGE, BLEU, semantic similarity), and retrieval metrics (context precision, context recall)
  - Can run automatically after standard evaluation or separately on existing results
  - Results are saved alongside standard evaluation metrics and can be integrated with spreadsheet reporting

### Batching and Concurrency Settings

Key settings to optimize performance and manage API rate limits. Larger = fewer calls but may hit token limits for OpenAI.

- **`OPENAI_EMBEDDING_BATCH_SIZE`** (default: 200): Text items per embedding API request.
- **`OPENAI_CHAT_COMPLETION_MAX_CONCURRENCY`** (default: 10): Number of async calls to the API done simultaneously.
- **`RAG_BATCH_SIZE`** (default: 50): Number of queries processed together during evaluation.

### Supported Datasets

- **Natural Questions**: Question-answering from real Google search queries - [github.com/facebookresearch/KILT](https://github.com/facebookresearch/KILT)
- **SQuAD**: Stanford Question Answering Dataset - [huggingface/squad](https://huggingface.co/datasets/rajpurkar/squad)
- **MultiHopRAG**: Multi-hop reasoning questions - [huggingface/multihoprag](https://huggingface.co/datasets/yixuantt/MultiHopRAG)

### Import RAGAS CSV queries

You can import queries produced by the RAGAS framework (or similar CSV exports) into the database and evaluate them with the rest of the pipeline.

1) Place CSVs under `data/ragas_queries` (you may organize them in any subfolders). The loader will find files by filename anywhere inside this folder.

2) Load queries into the DB (associate with your dataset and tag with a prompt name), then embed them and evaluate.

Examples:

```bash
# Place CSV in data/ragas_queries/squad/squad_ragas_singlehop_unique_pg_medium.csv

# Load by filename (found recursively under data/ragas_queries)
trcli questions load_csv squad_ragas_singlehop_unique_pg_medium.csv \
  --dataset=squad \
  --prompt_name=ragas_squad \
  --llm=ragas

# Load an entire directory of CSVs
trcli questions load_csv data/ragas_queries/squad/ \
  --dataset=squad \
  --prompt_name=ragas_squad \
  --llm=ragas

# Load multiple files at once
trcli questions load_csv [squad_ragas_singlehop_unique_pg_medium.csv,squad_ragas_singlehop_unique_pg_test.csv] \
  --dataset=squad \
  --prompt_name=ragas_squad \
  --llm=ragas

# After loading, embed the imported queries (filter by the llm you used above)
trcli embed queries --dataset=squad --llm=ragas

# Now you can run evaluation (optionally with RAGAS)
trcli eval run --dataset=squad --run_ragas=true
```

Expected CSV columns (minimum):
- `user_input` (required)
- `original_row_idx` (required)

Optional columns used if present:
- `reference`, `response`, `multi_responses`, `reference_contexts`, `retrieved_contexts`, `rubrics`, `persona_name`, `persona_role_description`, `node_id`, `source_chunk_idx`

Notes:
- Files are discovered by filename under `data/ragas_queries`, so you can pass just the filename instead of the full path.
- Include both `original_row_idx` and `source_chunk_idx` when available. The loader uses these to map imported questions to existing chunks and to set stable IDs for deduplication.

Sample CSV rows (the `hf_id` is in `source_chunk_idx` column)

```csv
intra_chunk_qid,original_row_idx,user_input,reference,response,multi_responses,reference_contexts,retrieved_contexts,rubrics,persona_name,persona_role_description,node_id,source_chunk_idx
0,0,How many people were reported killed in the 2008 Sichuan earthquake according to the records?,"According to the records, 69,197 people were reported killed in the 2008 Sichuan earthquake.",,,"The 2008 Sichuan earthquake or the Great Sichuan earthquake, measured at 8.0 Ms and 7.9 Mw, and occurred at 02:28:01 PM China Standard Time at epicenter (06:28:01 UTC) on May 12 in Sichuan province, killed 69,197 people and left 18,222 missing.",,,Seismologist Li,"Studies earthquakes and their impacts, particularly focusing on the 2008 Sichuan earthquake.",e882ecf7-d450-437c-b68b-82fe9845c6e5,3286
1,1,How many people were reported dead as a direct result of the 2008 Sichuan earthquake?,"The 2008 Sichuan earthquake resulted in the death of 69,197 people.",,,"The 2008 Sichuan earthquake or the Great Sichuan earthquake, measured at 8.0 Ms and 7.9 Mw, and occurred at 02:28:01 PM China Standard Time at epicenter (06:28:01 UTC) on May 12 in Sichuan province, killed 69,197 people and left 18,222 missing.",,,Seismologist Li,"Studies earthquakes and their impacts, particularly focusing on the 2008 Sichuan earthquake.",e882ecf7-d450-437c-b68b-82fe9845c6e5,3286
0,3,"How far was the tremor felt during the Wenchuan earthquake, specifically reaching Beijing?","The tremor from the Wenchuan earthquake was felt as far away as Beijing, which is approximately 1,500 km (930 mi) from the epicenter.",,,"It is also known as the Wenchuan earthquake (Chinese: 汶川大地震; pinyin: Wènchuān dà dìzhèn; literally: ""Great Wenchuan earthquake""), after the location of the earthquake's epicenter, Wenchuan County, Sichuan. The epicenter was 80 kilometres (50 mi) west-northwest of Chengdu, the provincial capital, with a focal depth of 19 km (12 mi). The earthquake was also felt in nearby countries and as far away as both Beijing and Shanghai—1,500 km (930 mi) and 1,700 km (1,060 mi) away—where office buildings swayed with the tremor. Strong aftershocks, some exceeding magnitude 6, continued to hit the area even months after the main quake, causing new casualties and damage.",,,Seismologist Li,"Studies earthquakes and their impacts, particularly focusing on the aftermath and aftershocks of major seismic events like the Wenchuan earthquake.",3cb3fde4-7ef2-424d-a658-c2237b713aa8,3294
```

## Google Spreadsheet Integration

The system can automatically write evaluation results to Google Spreadsheets for easy analysis and sharing. This feature requires setting up Google API credentials and configuring your spreadsheet details.

### Google API Setup

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Required APIs**:
   - Navigate to "APIs & Services" > "Library"
   - Enable the following APIs:
     - **Google Sheets API**
     - **Google Drive API**

3. **Create Service Account**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in the service account details
   - Skip role assignment (you can add roles later if needed)
   - Click "Done"

4. **Generate Service Account Key**:
   - Click on your newly created service account
   - Go to the "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose **JSON** format
   - Download the key file

5. **Place Credentials File**:
   ```
   Place the downloaded JSON file as:
   config/google_config.json
   ```

### Spreadsheet Setup

1. **Create or Open Google Spreadsheet**:
   - Create a new Google Spreadsheet or open an existing one
   - Copy the structure from the Example spreadsheet (removed for anonimization)
   - Share the spreadsheet for editing.

For privacy-sensitive evaluations you may share "Editor" permissions only with your service account email (found in the JSON file).

2. **Get Spreadsheet Details**:
    - For syncronizing this project with your spreadsheet, you need to setup Spreadsheet ID and GID. Get these values from the url:
   ```bash
   # From URL: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid={GID}
   #
   # SPREADSHEET_ID: The long string in the URL
   # GID: The sheet identifier (can be numeric ID or exact sheet name)
   ```

3. **Configure Environment Variables**:
   ```bash
   # Add to your .env file:
   RAG_WRITE_SPREADSHEET=true
   SPREADSHEET_ID=
   SPREADSHEET_GID=Sheet1  # Sheet name or numeric GID
   ```

### Data Format

When enabled, evaluation results are automatically written to your spreadsheet in the following format:

| experiment_name | dataset | qa_type | qgen_model | embedding_model | rag_model | deduplication_factor | k | context_accuracy | title_accuracy | mrr | ndcg | queries_ratio |
|----------------|---------|---------|------------|----------------|-----------|-------------------|---|-----------------|----------------|-----|------|---------------|
| 20250824_123456_natural-questions_... | natural-questions | query | Qwen/Qwen2.5-32B | all-MiniLM-L6-v2 | Qwen/Qwen2.5-7B | 1 | 1 | 0.3164 | 0.956 | 0.3452 | 0.3452 | 0.9342 |
| | | | | | | 1 | 5 | 0.5133 | 0.9874 | 0.4248 | 0.2729 | 0.9024 |
| | | | | | | 1 | 20 | 0.6868 | 0.9966 | 0.4427 | 0.1706 | 0.8847 |

## SQLite vs Postgres

By default, the system uses **SQLite** which is stored in `./data/sql/store.db`. **SQLite is sufficient for most use cases and no additional setup is required.**

If you still want to use PostgreSQL, the project includes full PostgreSQL support with Docker setup and migration tools.

#### Postgres Setup Steps

1. **Create PostgreSQL environment variables**

   Add these variables to your `.env` file:
   ```bash
   # PostgreSQL Configuration (Optional)
   POSTGRES_USER=raguser
   POSTGRES_PASSWORD=ragpass
   POSTGRES_HOST=127.0.0.1
   POSTGRES_PORT=9700
   POSTGRES_DB=ragdb
   USE_SQLITE=false  # Set to false to use PostgreSQL instead of SQLite
   ```

2. **Start PostgreSQL with Docker**
   ```bash
   # Start PostgreSQL service
   docker-compose up -d postgres

   # Check if PostgreSQL is running
   docker-compose ps
   ```

3. **Database schema setup**

   The database schema is created automatically when you start the PostgreSQL service. The Docker Compose configuration includes an Alembic service that runs migrations automatically after PostgreSQL is ready.

4. **Optional: Migrate existing SQLite data to PostgreSQL**

   If you already have data in SQLite and want to migrate it:
   ```bash
   # Make sure you have a newly initialized PostgreSQL database
   docker compose down -v # remove the existing volume
   docker-compose up -d postgres

   # Use the CLI to migrate data
   trcli db export_to_postgres

   # Or specify custom database URLs
   trcli db export_to_postgres --source_database_url="sqlite:///custom.db" --target_database_url="postgresql+asyncpg://user:pass@host:port/db"
   ```

### Database Schema
```mermaid
erDiagram
    Doc ||--o{ Chunk : has
    Doc ||--o{ Query : has
    Chunk ||--o{ ChunkVector : has
    Query ||--o{ QueryVector : has
    Query ||--|| Answer : has
    Query ||--o{ RagResponseDB : has
    Query ||--o{ QueryTriplet : has
    Prompt ||--o{ Query : has
    EvaluationSet ||--o{ RagResponseDB : has
    EvaluationSet ||--o{ QueryTriplet : has

    Doc {
        int id PK
        string dataset
        string title
        text text
        string text_hash
        string hf_id
        boolean is_virtual
        json meta
    }
    Chunk {
        int id PK
        int doc_id FK
        text text
        string text_hash
        string hf_id
        json meta
    }
    ChunkVector {
        int id PK
        int parent_id FK
        blob vector
        string emb_model
    }
    QueryVector {
        int id PK
        int parent_id FK
        blob vector
        string emb_model
    }
    Query {
        int id PK
        int doc_id FK
        text text
        string text_hash
        string qa_type
        string llm
        int prompt_id FK
        string hf_id
        json meta
    }
    Prompt {
        int id PK
        string name
        text text
        string text_hash
    }
    Answer {
        int id PK
        int query_id FK
        text text
    }
    EvaluationSet {
        int id PK
        datetime datetime
        string dataset
        json metrics
        json settings
    }
    RagResponseDB {
        int id PK
        int evaluation_set_id FK
        json retrieval_result
        int query_id FK
    }
    QueryTriplet {
        int id PK
        int evaluation_set_id FK
        int eval_query_id FK
        int gold_answer_id FK
        json contexts_by_k
        json rag_answers_by_k
        datetime created_at
        datetime archived_at
    }
```

## CLI Reference

The command-line interface provides tools for running ETL operations, question generation, embedding creation, RAG queries, and evaluation. The CLI tool is named `trcli` and is the main entry point for all operations.

### Installation and Setup

```bash
# Install the package
uv pip install .

# Activate virtual environment
. .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate.ps1  # Windows PowerShell

# Verify installation
trcli --help
```

### Command Structure

All commands follow the pattern: `trcli <command> <subcommand> [options]`

### Default Values and Argument-Free Usage

**Most CLI commands can be run without any arguments** - they will use sensible defaults from your configuration file (`.env`). This makes it easy to get started quickly:

```bash
# These commands work out of the box with defaults or variables from the .env:
trcli etl fill_db                      # Load the dataset from huggingface to SQL
trcli questions generate               # Generate questions
trcli embed all                        # Embed chunks and questions
trcli embed chunks                     # Embed chunks only
trcli embed queries                    # Embed queries only
trcli rag query "Your question here"   # Query and output to the console
trcli eval run                         # Run evaluation based on queries from the dataset
trcli db stats                         # Show breakdown of database tables
```

**Configuration Priority:**
1. Command-line arguments (highest priority)
2. Environment variables from `.env` file
3. Built-in defaults (lowest priority)

This means you can:
- Set your preferred defaults in `.env` and rarely need command arguments
- Override specific settings per command when needed
- Mix and match: use defaults for most parameters, specify only what you want to change

### Getting Help

```bash
# General help
trcli --help

# Command-specific help
trcli eval --help

# Subcommand help
trcli eval run --help
```

### Common Workflows

#### Complete Pipeline Setup
```bash
# 1. Load data
trcli etl fill_db

# 2. Generate questions
trcli questions generate

# 3. Generate embeddings
trcli embed all

# 4. Test queries
trcli rag query "What is machine learning?"

# 5. Run evaluation (with RAGAS)
trcli eval run --run_ragas=true

# Or run RAGAS evaluation separately on existing results
trcli eval run_ragas path/to/evaluation_results.json
```

#### Quick Testing
```bash
# Setup RAG_DATASET un .env
RAG_DATASET=squad

# Load small number of rows for testing
trcli etl fill_db --n_rows=1000

# Generate embeddings
trcli embed all

# Test single query
trcli rag query "Test question" --k=5

# Test small number of queries
trcli eval run --max_queries 10
```


### Triplet-Augmented RAG

This mode stores evaluation triplets (question, retrieved contexts across all k, generated answer-by-k) and can use triple-block formatting during generation.

#### 1) Generate eval-origin queries

```bash
# Use your dataset; origin=eval marks queries for evaluation ground-truth
trcli questions generate --dataset=squad --origin=eval
```

#### 2) Embed eval-origin queries

```bash
# Ensure eval-origin queries have embeddings for evaluation selection
trcli embed queries --dataset=squad --llm="${QUESTION_GENERATION_MODEL}"
```

#### 3) Run evaluation and persist triplets

Set which query origins to evaluate via config (affects selection of evaluation queries):

```bash
# .env
RAG_RETRIEVAL_QUERY_ORIGINS=["eval"]
RAG_PERSIST_TRIPLETS=true
```

Then run:

```bash
trcli eval run --dataset=squad --silent=false
```

This writes standard artifacts under `data/eval/...` and persists triplets into the DB linked to an `EvaluationSet`.

#### 4) Use triplet formatting during RAG

Enable triple-block formatting in responses:

```bash
# .env
RAG_USE_TRIPLET_CONTEXTS=true

# Optionally reset evaluation selection back to dataset-origin queries for inference
RAG_RETRIEVAL_QUERY_ORIGINS=["dataset"]
```

Then query as usual:

```bash
trcli rag query "What is the capital of France?"
```

Notes:
- Triple-block formatting collapses the prompt context as:
  - `Question:` original question
  - `Retrieved context i:` texts for each retrieved item
  - `Answer:` placed at the bottom (generated by LLM during inference)
- Retrieval corpus filtering for query-vectors uses `RAG_RETRIEVAL_QUERY_ORIGINS`.

#### 5) Run triplet evaluation

Run evaluation using the triplet-augmented pipeline:

```bash
# Standard triplet evaluation (uses eval-origin index)
trcli eval run_triplet --dataset=squad

# Or use triplets from a JSON file
trcli eval run_triplet --dataset=squad --triplet_json_path=data/eval/experiment/triplets.json
```

#### 6) Cleanup and retention

Archive triplets for a specific evaluation run:

```bash
trcli db archive_eval_run --evaluation_set_id 123
```

Hard delete an evaluation run (cascades to triplets and rag_responses):

```bash
trcli db delete_eval_run --evaluation_set_id 123
```

Prune old runs by dataset (keep the newest N):

```bash
trcli db prune_triplets --dataset=squad --keep_last=3
```

Or delete runs older than N days (only archived by default):

```bash
trcli db prune_triplets --dataset=squad --older_than_days=30
```


### Two-Stage RAG Pipeline

Two-stage RAG provides an advanced retrieval approach that combines query-based and chunk-based retrieval in sequence:

- **Stage 1 (Query Retrieval)**: Retrieve over synthetic/evaluation query vectors with `origin="two-stage"` to locate associated triplets.
- **Stage 2 (Chunk Retrieval)**: Retrieve over chunk vectors for document context.
- **Generation**: Contexts from both stages are concatenated (triplet contexts first, then chunk contexts) and passed to the LLM for answer generation.

#### 1) Generate two-stage queries

```bash
# Generate queries with origin=two-stage for the query-based retrieval stage
trcli questions generate --dataset=squad --origin=two-stage
```

#### 2) Embed two-stage queries

```bash
# Create embeddings for the two-stage queries
trcli embed queries --dataset=squad --llm="${QUESTION_GENERATION_MODEL}"
```

#### 3) Run two-stage evaluation

```bash
trcli eval run_two_stage_triplet --dataset=squad --k_query=5 --k_chunk=10
```

**Two-Stage Configuration Options:**
- `--k_query`: Number of queries to retrieve in stage 1 (default: max(ks))
- `--k_chunk`: Number of chunks to retrieve in stage 2 (default: max(ks))
- Both stages can be controlled independently for optimal retrieval balance

#### How Two-Stage Differs from Standard Triplet RAG

| Aspect | Triplet RAG | Two-Stage RAG |
|--------|-------------|---------------|
| Stage 1 | Query retrieval with `origin=eval` | Query retrieval with `origin=two-stage` |
| Stage 2 | N/A (triplet contexts only) | Chunk retrieval for additional context |
| Context | Triplet contexts replace retrieval | Triplet + chunk contexts combined |
| Use case | Reference QA injection | Hybrid retrieval augmentation |


### Available Commands Breakdown

#### 1. ETL Commands (`etl`)

Load and process datasets into the database.

##### `etl fill_db`
Load datasets into the database from HuggingFace.

```bash
trcli etl fill_db [DATASETS] [OPTIONS]

# Parameters:
#   datasets: List of datasets to load (optional)
#             Options: squad, natural-questions, multihoprag
#             Default: Uses RAG_DATASET from config
#   --n_rows: Number of rows to process per dataset (default: 10000, 0 for all)
#   --n_commited: Number of rows to commit at once (default: 1000)

# Examples:
trcli etl fill_db ["squad"] --n_rows=1000
trcli etl fill_db ["squad","natural-questions"] --n_rows=5000 --n_commited=500
trcli etl fill_db ["multihoprag"] --n_rows=0  # Load all rows
```

For using without arguments:
**`etl fill_db`**
- `settings.rag_max_rows` → `RAG_MAX_ROWS` (default rows to process)
- `settings.rag_dataset` → `RAG_DATASET` (default dataset when none specified)


#### 2. Question Generation Commands (`questions`)

Generate questions for documents using LLMs.

##### `questions generate`
Generate questions for documents in the database.

```bash
trcli questions generate [OPTIONS]

# Parameters:
#   --dataset: Dataset to generate questions for (default: from config)
#   --max_docs: Maximum number of documents to process (-1 for all)
#   --batch_size: Batch size for processing (default: 10)

# Examples:
trcli questions generate --dataset=squad
trcli questions generate --dataset=natural-questions --max_docs=100
trcli questions generate --batch_size=5 --max_docs=50
```

##### `questions convert_q_to_qa`
Convert existing queries with `qa_type='q'` to `qa_type='qa'` by appending their answers.

This command finds existing queries that have `qa_type='q'` (questions only) and creates new queries with `qa_type='qa'` (question + answer pairs) by appending the answer text to the question text using the configured `QA_SEPARATOR`.

```bash
trcli questions convert_q_to_qa [OPTIONS]

# Parameters:
#   --dataset: Dataset to convert queries for (default: from config)
#   --llm: LLM model to filter queries by (default: from config)

# Examples:
trcli questions convert_q_to_qa --dataset=squad
trcli questions convert_q_to_qa --dataset=natural-questions --llm=gpt-4o-mini
trcli questions convert_q_to_qa  # Uses defaults from config
```

##### `questions load_csv`
Load evaluation queries from CSV files into the database with proper prompt tagging.

CSV files should contain `original_row_idx` (mapping to chunk `hf_id`) and `user_input` (query text). Optional columns include `reference` (ground truth), `response` (model output), persona fields, and context information.

```bash
trcli questions load_csv [CSV_PATHS] [OPTIONS]

# Parameters:
#   csv_paths: Path(s) to CSV file(s) or directory(ies) containing CSV files
#   --dataset: Dataset name to associate queries with (required)
#   --prompt_name: Prompt name for query tagging (default: "ragas_queries")
#   --prompt_text: Prompt description (default: value of prompt_name)
#   --n_rows: Maximum rows to process per file (0 for all)
#   --n_committed: Commit frequency (default: 100)

# Examples:
# Make sure to specify prompt_name
trcli questions load_csv queries.csv --dataset=squad --prompt_name=eval_queries
trcli questions load_csv csv_directory/ --dataset=squad --n_rows=1000 --prompt_name=eval_queries
trcli questions load_csv [file1.csv,file2.csv,dir1/] --dataset=my_dataset --prompt_name=eval_queries
trcli questions load_csv queries.csv --dataset=my_dataset --prompt_name=eval_queries
```

For using without arguments:
**`questions generate`**
- `settings.rag_dataset` → `RAG_DATASET` (default dataset)
- `settings.openai_chat_completion_max_concurrency` → `OPENAI_CHAT_COMPLETION_MAX_CONCURRENCY` (batch size)
- `settings.question_generation_model` → `QUESTION_GENERATION_MODEL` (LLM for questions)
- `settings.question_generation_temperature` → `QUESTION_GENERATION_TEMPERATURE` (generation randomness)
- `settings.question_generation_max_completion_tokens` → `QUESTION_GENERATION_MAX_COMPLETION_TOKENS` (max tokens)
- `settings.prompt_mode` → `PROMPT_MODE` (basic/complex prompts)

**`questions convert_q_to_qa`**
- `settings.rag_dataset` → `RAG_DATASET` (default dataset)
- `settings.question_generation_model` → `QUESTION_GENERATION_MODEL` (LLM to filter by)

**`questions load_csv`**
- No default arguments - `dataset` parameter is required
- Loads queries from CSV evaluation datasets with proper prompt tagging for isolation from generated queries


#### 3. Embedding Commands (`embed`)

Generate embeddings for chunks and queries.

##### `embed chunks`
Generate embeddings for document chunks.

```bash
trcli embed chunks [OPTIONS]

# Parameters:
#   --dataset: Dataset to embed chunks for (default: from config)
#   --max_items: Maximum number of items to embed (-1 for all)
#   --batch_size: Batch size for embedding (default: 200)
#   --emb_model: Embedding model to use (default: text-embedding-3-small)

# Examples:
trcli embed chunks --dataset=squad
trcli embed chunks --max_items=1000 --batch_size=100
trcli embed chunks --emb_model=text-embedding-3-large
```

##### `embed queries`
Generate embeddings for queries.

```bash
trcli embed queries [OPTIONS]

# Parameters:
#   --dataset: Dataset to embed queries for (default: from config)
#   --llm: Language model used to generate queries (default: gpt-4o-mini)
#   --max_items: Maximum number of items to embed (-1 for all)
#   --batch_size: Batch size for embedding (default: 200)
#   --emb_model: Embedding model to use (default: text-embedding-3-small)

# Examples:
trcli embed queries --dataset=squad
trcli embed queries --llm=gpt-4o --max_items=500
```

##### `embed all`
Generate embeddings for both chunks and queries.

```bash
trcli embed all [OPTIONS]

# Uses same parameters as chunks and queries commands
# Examples:
trcli embed all --dataset=squad --max_items=1000
```

For using without arguments:
**`embed chunks/queries/all`**
- `settings.rag_dataset` → `RAG_DATASET` (default dataset)
- `settings.openai_embedding_batch_size` → `OPENAI_EMBEDDING_BATCH_SIZE` (how many text items are sent to the model in single request)
- `settings.openai_embedding_model` → `OPENAI_EMBEDDING_MODEL` (embedding model)
- `settings.rag_generation_model` → `RAG_GENERATION_MODEL` (for query embeddings)
- `settings.openai_api_key` → `OPENAI_API_KEY` (required for API access)
- `settings.openai_api_base_embedding` → `OPENAI_API_BASE_EMBEDDING` (API endpoint)

#### 4. RAG Commands (`rag`)

Query the RAG system and run inference.

##### `rag query`
Query the RAG system with a single question.

```bash
trcli rag query "QUESTION" [OPTIONS]

# Parameters:
#   question: The question to ask (required, in quotes)
#   --k: Number of documents to retrieve (default: 20)
#   --emb_model: Embedding model to use (default: text-embedding-3-small)
#   --dataset: Dataset to query against (default: natural-questions)
#   --retrieval_sources: Sources for retrieval (default: ["chunk", "query"])
#   --output_json: Output in JSON format (default: false)

# Examples:
trcli rag query "What is the capital of France?"
trcli rag query "How does machine learning work?" --k=10 --dataset=squad
trcli rag query "What is AI?" --output_json=true
trcli rag query "Explain quantum computing" --retrieval_sources=["chunk"]
```

##### `rag query_batch`
Query the RAG system with multiple questions.

```bash
trcli rag query_batch [QUESTIONS] [OPTIONS]

# Parameters:
#   questions: List of questions to ask
#   --k: Number of documents to retrieve (default: 20)
#   --emb_model: Embedding model to use
#   --dataset: Dataset to query against
#   --retrieval_sources: Sources for retrieval
#   --batch_size: Batch size for processing (default: 10)
#   --output_json: Output in JSON format

# Examples:
trcli rag query_batch ["What is AI?", "How does ML work?"]
trcli rag query_batch ["Question 1", "Question 2"] --batch_size=5 --output_json=true
```


For using without arguments
**`rag query/query_batch`**
- `settings.rag_retrieval_k` → `RAG_RETRIEVAL_K` (number of documents to retrieve)
- `settings.openai_embedding_model` → `OPENAI_EMBEDDING_MODEL` (for query embedding)
- `settings.rag_dataset` → `RAG_DATASET` (default dataset)
- `settings.rag_retrieval_sources` → `RAG_RETRIEVAL_SOURCES` (chunk/query sources)
- `settings.rag_generation_model` → `RAG_GENERATION_MODEL` (for answer generation)
- `settings.rag_generation_temperature` → `RAG_GENERATION_TEMPERATURE` (generation randomness)
- `settings.rag_generation_max_tokens` → `RAG_GENERATION_MAX_TOKENS` (max response length)
- `settings.openai_api_key` → `OPENAI_API_KEY` (required for API access)


#### 5. Evaluation Commands (`eval`)

Evaluate RAG system performance with metrics.

##### `eval run`
Run evaluation on the RAG system. The retrieval results, parsed ouptputs and metrics will be available in the `/data/eval` folder.

```bash
trcli eval run [OPTIONS]

# Parameters:
#   --dataset: Dataset to evaluate on (default: from config)
#   --batch_size: Batch size for processing queries (default: 50)
#   --ks: List of k values for retrieval (default: [1, 5, 20])
#   --rag_llm: Language model for answer generation (default: gpt-4o-mini)
#   --qgen_llm: Language model for question generation (default: gpt-4o-mini)
#   --emb_model: Embedding model to use (default: text-embedding-3-small)
#   --retrieval_sources: Sources for retrieval (default: ["chunk", "query"])
#   --deduplication_factor: Factor for deduplication (default: 1)
#   --max_queries: Maximum number of queries to evaluate (-1 for all)
#   --generate_answer: Whether to generate answers (default: true)
#   --write_spreadsheet: Write results to Google Spreadsheet (default: from config)
#   --spreadsheet_sheet_name: Name of the sheet to write to (default: from config)
#   --output_json: Output results in JSON format (default: false)
#   --silent: Run silently without output (default: true)
#   --run_ragas: Whether to automatically run RAGAS evaluation after standard evaluation (default: from config)

# Examples:
trcli eval run --dataset=squad --max_queries=100
trcli eval run --ks=[5,10,20] --batch_size=25 --silent=false
trcli eval run --dataset=multihoprag --output_json=true
trcli eval run --generate_answer=false --max_queries=50
trcli eval run --write_spreadsheet=true --spreadsheet_sheet_name="Results"
trcli eval run --run_ragas=true  # Run with RAGAS evaluation
```

##### `eval run_ragas`
Run advanced RAGAS evaluation on existing evaluation results using LLM-based metrics.

```bash
trcli eval run_ragas --json_path [JSON_PATH] [OPTIONS]

# Parameters:
#   --json_path: Path to existing evaluation JSON file (optional)
#   --metrics_to_use: List of RAGAS metrics to compute (default: from config)
#   --ks: List of k values for evaluation (default: from config)
#   --update_existing: Update the original JSON file (default: true)
#   --save_results: Save updated results to files (default: true)
#   --output_json: Output in JSON format (default: false)
#   --experiment_name: Name for spreadsheet updates (auto-detected if not provided)

# Available RAGAS Metrics:
#   Generation metrics: faithfulness, answer_relevancy, nv_accuracy, summary_score
#   Classical metrics: rouge_score, bleu_score, semantic_similarity, exact_match
#   Retrieval metrics: context_precision, context_recall, factual_correctness

# Examples:
trcli eval run_ragas data/eval/20250101_120000_natural-questions.json
trcli eval run_ragas --metrics_to_use=[faithfulness,context_precision,answer_relevancy]
trcli eval run_ragas experiment_results.json --ks [1,5,20] --save_results=true
trcli eval run_ragas --experiment_name "my_experiment" --output_json=true
```

##### `eval recalculate_metrics`
Recalculate metrics for an existing experiment.

```bash
trcli eval recalculate_metrics EXP_NAME [OPTIONS]

# Parameters:
#   exp_name: Name of the experiment to recalculate (required)
#   --ks: List of k values for retrieval (default: [1, 5, 20])
#   --write_to_file: Write results to file (default: true)

# Examples:
trcli eval recalculate_metrics "experiment_20241201"
trcli eval recalculate_metrics "test_run" --ks=[1,10] --write_to_file=false
```

##### `eval run_triplet`
Run triplet-augmented evaluation using dataset-origin questions with eval-origin query index.

Triplet evaluation injects stored triplet contexts (question + retrieved contexts + reference answer) as the retrieval result, enabling fine-grained control over what context the LLM sees.

```bash
trcli eval run_triplet [OPTIONS]

# Parameters:
#   --dataset: Dataset to evaluate on (default: from config)
#   --batch_size: Batch size for processing (default: 50)
#   --ks: List of k values for retrieval (default: [1, 5, 20])
#   --rag_llm: Language model for answer generation (default: gpt-4o-mini)
#   --qgen_llm: Language model for question generation (default: gpt-4o-mini)
#   --emb_model: Embedding model to use (default: text-embedding-3-small)
#   --retrieval_sources: Sources for retrieval (default: ["chunk", "query"])
#   --deduplication_factor: Factor for deduplication (default: 1)
#   --max_queries: Maximum number of queries to evaluate (-1 for all)
#   --generate_answer: Whether to generate answers (default: true)
#   --triplet_json_path: Path to triplets JSON file (optional, uses DB if not set)
#   --output_json: Output in JSON format (default: false)
#   --silent: Run silently without output (default: true)

# Examples:
trcli eval run_triplet --dataset=squad --max_queries=100
trcli eval run_triplet --triplet_json_path=data/eval/experiment/triplets.json
trcli eval run_triplet --ks=[1,5,10] --generate_answer=true
```

##### `eval run_two_stage_triplet`
Run two-stage (query+chunk) triplet-augmented evaluation.

This combines query-based triplet retrieval (stage 1) with chunk-based retrieval (stage 2) for hybrid context injection.

```bash
trcli eval run_two_stage_triplet [OPTIONS]

# Parameters:
#   --dataset: Dataset to evaluate on (default: from config)
#   --batch_size: Batch size for processing (default: 50)
#   --ks: List of k values for final context (default: [1, 5, 20])
#   --rag_llm: Language model for answer generation (default: gpt-4o-mini)
#   --qgen_llm: Language model for question generation (default: gpt-4o-mini)
#   --emb_model: Embedding model to use (default: text-embedding-3-small)
#   --deduplication_factor: Factor for deduplication (default: 1)
#   --max_queries: Maximum number of queries to evaluate (-1 for all)
#   --generate_answer: Whether to generate answers (default: true)
#   --k_query: Number of queries to retrieve in stage 1 (default: max(ks))
#   --k_chunk: Number of chunks to retrieve in stage 2 (default: max(ks))
#   --triplet_json_path: Path to triplets JSON file (optional)
#   --output_json: Output in JSON format (default: false)
#   --silent: Run silently without output (default: true)

# Examples:
trcli eval run_two_stage_triplet --dataset=squad
trcli eval run_two_stage_triplet --k_query=5 --k_chunk=10 --ks=[5,10,15]
trcli eval run_two_stage_triplet --triplet_json_path=data/eval/triplets.json
```

For using without arguments:
**`eval run`**
- `settings.rag_dataset` → `RAG_DATASET` (dataset to evaluate)
- `settings.rag_batch_size` → `RAG_BATCH_SIZE` (processing batch size)
- `settings.rag_retrieval_ks` → `RAG_RETRIEVAL_KS` (k values for retrieval)
- `settings.rag_generation_model` → `RAG_GENERATION_MODEL` (answer generation LLM)
- `settings.question_generation_model` → `QUESTION_GENERATION_MODEL` (question generation LLM)
- `settings.openai_embedding_model` → `OPENAI_EMBEDDING_MODEL` (embedding model)
- `settings.rag_retrieval_sources` → `RAG_RETRIEVAL_SOURCES` (retrieval sources)
- `settings.rag_deduplication_factor` → `RAG_DEDUPLICATION_FACTOR` (deduplication factor)
- `settings.rag_generate_answer` → `RAG_GENERATE_ANSWER` (whether to generate answers)
- `settings.rag_write_spreadsheet` → `RAG_WRITE_SPREADSHEET` (write to Google Spreadsheet)
- `settings.spreadsheet_id` → `SPREADSHEET_ID` (Google Spreadsheet ID)
- `settings.spreadsheet_gid` → `SPREADSHEET_GID` (sheet name or GID)
- `settings.ragas_enabled` → `RAGAS_ENABLED` (whether to run RAGAS evaluation automatically)

**`eval run_ragas`**
- `settings.ragas_metrics` → `RAGAS_METRICS` (list of RAGAS metrics to compute)
- `settings.eval_context_ks` → `EVAL_CONTEXT_KS` (k values for context evaluation)
- `settings.eval_llm` → `EVAL_LLM` (LLM for RAGAS evaluation)
- `settings.eval_embedding_model` → `EVAL_EMBEDDING_MODEL` (embedding model for RAGAS)
- `settings.rag_write_spreadsheet` → `RAG_WRITE_SPREADSHEET` (write to Google Spreadsheet)

**`eval run_triplet` / `eval run_two_stage_triplet`**
- `settings.rag_dataset` → `RAG_DATASET` (dataset to evaluate)
- `settings.rag_batch_size` → `RAG_BATCH_SIZE` (processing batch size)
- `settings.rag_retrieval_ks` → `RAG_RETRIEVAL_KS` (k values for retrieval)
- `settings.rag_generation_model` → `RAG_GENERATION_MODEL` (answer generation LLM)
- `settings.question_generation_model` → `QUESTION_GENERATION_MODEL` (question generation LLM)
- `settings.openai_embedding_model` → `OPENAI_EMBEDDING_MODEL` (embedding model)
- `settings.rag_deduplication_factor` → `RAG_DEDUPLICATION_FACTOR` (deduplication factor)
- `settings.rag_triplet_json_path` → `RAG_TRIPLET_JSON_PATH` (triplets JSON source)
- `settings.rag_n_triplet_contexts` → `RAG_N_TRIPLET_CONTEXTS` (contexts per triplet)
- `settings.rag_triplet_apply_formatting` → `RAG_TRIPLET_APPLY_FORMATTING` (triplet, chunk, or qa)
- `settings.eval_whitelist_origins` → `EVAL_WHITELIST_ORIGINS` (allowed query origins)


#### 6. Database Commands (`db`)

Manage database operations, view statistics, and export data between databases.

##### `db stats`
Show comprehensive database statistics.

```bash
trcli db stats

# Shows:
# - Total counts for documents, chunks, queries, answers, etc.
# - Dataset breakdown
# - LLM breakdown
# - Embedding model breakdown
```

##### `db datasets`
List available datasets in the database.

```bash
trcli db datasets

# Shows each dataset with document and query counts
```

##### `db delete_queries_by_llm`
Delete queries generated by a specific LLM.

```bash
trcli db delete_queries_by_llm [OPTIONS]

# Parameters:
#   --llm: Language model name (default: from config)

# Examples:
trcli db delete_queries_by_llm --llm=gpt-4o-mini
trcli db delete_queries_by_llm --llm=Qwen/Qwen2.5-7B-Instruct
```

For using without arguments:
**`db delete_queries_by_llm`**
- `settings.question_generation_model` → `QUESTION_GENERATION_MODEL` (default LLM to target)
- `settings.rag_dataset` → `RAG_DATASET` (dataset scope)


##### `db delete_vectors_by_model`
Delete vectors generated by a specific embedding model.

```bash
trcli db delete_vectors_by_model [OPTIONS]

# Parameters:
#   --emb_model: Embedding model name (default: from config)
#   --llm: Language model name (default: from config)
#   --delete_chunks: Also delete chunk vectors (default: false)
#   --dataset: Dataset to target (default: from config)

# Examples:
trcli db delete_vectors_by_model --emb_model=text-embedding-3-small
trcli db delete_vectors_by_model --delete_chunks=true --dataset=squad
```

For using without arguments:
**`db delete_vectors_by_model`**
- `settings.openai_embedding_model` → `OPENAI_EMBEDDING_MODEL` (default embedding model)
- `settings.rag_generation_model` → `RAG_GENERATION_MODEL` (default LLM)
- `settings.rag_dataset` → `RAG_DATASET` (dataset scope)


##### `db export_to_postgres`
Export all data from SQLite to PostgreSQL.

```bash
trcli db export_to_postgres [OPTIONS]

# Parameters:
#   --source_database_url: Source database URL (default: SQLite from config)
#   --target_database_url: Target PostgreSQL URL (default: PostgreSQL from config)
#   --recreate_tables: Whether to drop and recreate tables (default: false)
#   --commit_every: Commit every N records (default: 1000)
#   --skip_orphaned_records: Skip records with invalid foreign keys (default: true)

# Examples:
trcli db export_to_postgres --recreate_tables=true
trcli db export_to_postgres --source_database_url="sqlite:///custom.db"
trcli db export_to_postgres --commit_every=500
```

For using without arguments:
**`db export_to_postgres`**
- `settings.sqlite_database_url` → Source SQLite database URL
- `settings.postgres_database_url_async` → Target PostgreSQL database URL



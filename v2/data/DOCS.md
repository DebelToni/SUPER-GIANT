# SUPER GIANT v2 Data Processing Pipeline Documentation

## Overview

This directory contains the data generation and processing pipeline for creating distillation training data. The system generates synthetic Q&A pairs using a teacher model and prepares them for knowledge distillation training.

## Core Components

### 1. Data Generation Pipeline

#### `teacher_generate.py`
- **Purpose**: Two-stage pipeline for generating Q&A pairs with teacher model outputs
- **Key Features**:
  - **Stage A**: Parallel question generation using teacher model
  - **Stage B**: Answer generation with top-K logprobs for distillation
  - **Append Mode**: Can continue from existing datasets
  - **Role-based Encoding**: Explicit user/assistant role tokens and loss masking

**Key Functions**:
- `generate_questions()`: Creates questions in parallel using teacher model
- `generate_answers()`: Generates answers with top-K logprobs
- `encode_with_roles()`: Encodes Q&A pairs with role IDs and loss masks
- `parse_topk_json()`: Extracts top-K logprobs from teacher model responses

**Data Flow**:
```
Teacher Model → Questions → Answers → Arrow Files → Distillation Training
```

#### `Teacher_config.yml`
- **Purpose**: Central configuration for data generation
- **Key Settings**:
  - `model_name`: Teacher model identifier
  - `model_base_url`: vLLM API endpoint
  - `num_questions_total`: Target dataset size
  - `top_logprobs`: Number of top-K alternatives to capture
  - `question_system`/`answer_system`: Generation prompts
  - `user_token`/`ai_token`: Role markers for chat format

### 2. Tokenizer Management

#### `Create_custom_chat_tokenizer.py`
- **Purpose**: Creates a custom tokenizer with chat-specific tokens
- **Features**:
  - Adds `[USER]` and `[AI]` special tokens for role marking
  - Configures post-processing templates for chat format
  - Handles BOS/EOS token alignment
  - Saves tokenizer to `neo-english-cust` directory

**Template Processing**:
- Single: `<|bos|> [USER] $A <|endoftext|>`
- Pair: `<|bos|> [USER] $A <|endoftext|> [AI] $B:1 <|endoftext|>:1`

#### `Save_tokenizer_locally.py`
- **Purpose**: Simple tokenizer download and caching
- **Function**: Downloads base tokenizer and saves locally for offline use

### 3. Data Inspection Tools

#### `peek_arrow_sample.py`
- **Purpose**: Interactive data inspection and validation
- **Features**:
  - Random sampling from Arrow datasets
  - Displays Q&A pairs with formatting
  - Shows top-K logprobs for generated tokens
  - Validates role IDs and loss masks
  - Cross-checks question/answer consistency

## Data Format

### Questions Schema (`questions.arrow`)
- `qid`: Unique question identifier (int64)
- `question_text`: Raw question text (string)
- `meta`: Generation metadata (JSON string)

### Answers Schema (`answers.arrow`)
- `qid`: Question identifier (int64)
- `question_text`: Original question (string)
- `answer_text`: Generated answer (string)
- `topk_json_per_token`: Top-K logprobs per generated token (list of JSON strings)
- `student_input_ids`: Tokenized input sequence (list of int32)
- `student_role_ids`: Role identifiers (1=user, 2=assistant, 0=other) (list of int8)
- `student_loss_mask`: Training mask (1=learn, 0=ignore) (list of int8)
- `meta`: Generation parameters (JSON string)

## Role-Based Encoding

The system uses explicit role tokens and loss masking:

```python
# Example encoding for: "[USER] What is AI?\n[AI] Artificial Intelligence..."
input_ids:    [BOS, USER_TOKEN, what, is, AI, ?, EOS, AI_TOKEN, artificial, intelligence, ...]
role_ids:     [0,   1,         1,    1,  1,  1,  0,   2,        2,          2,           ...]
loss_mask:    [0,   0,         0,    0,  0,  0,  0,   1,        1,          1,           ...]
```

- **Role IDs**: 1=user, 2=assistant, 0=other
- **Loss Mask**: 1=positions to learn (assistant responses), 0=ignore (user input, separators)

## Usage Workflow

### 1. Setup Tokenizer
```bash
python Create_custom_chat_tokenizer.py
python Save_tokenizer_locally.py
```

### 2. Generate Training Data
```bash
python teacher_generate.py --config Teacher_config.yml
```

### 3. Inspect Generated Data
```bash
python peek_arrow_sample.py --questions teacher_out/questions.arrow --answers teacher_out/answers.arrow
```

## Configuration Parameters

### Generation Parameters
- `temperature`: 1.5 (creative generation)
- `top_p`: 0.7 (nucleus sampling)
- `top_logprobs`: 8 (top-K alternatives for distillation)
- `max_new_tokens_answer`: 256 (answer length limit)

### Parallelization
- `parallel_calls`: 32 (concurrent API requests)
- `questions_per_chunk`: 1000 (batch size for generation)

### Output Directories
- `out_dir`: teacher_out/
- `questions_arrow`: teacher_out/questions.arrow
- `answers_arrow`: teacher_out/answers.arrow

## Dependencies

- **vLLM/OpenAI API**: Teacher model inference
- **PyArrow**: Efficient data storage
- **Transformers**: Tokenizer handling
- **OmegaConf**: Configuration management
- **Requests**: HTTP client for API calls

## Integration with Training

The generated Arrow files are consumed by the distillation training pipeline:
- `distill_prepare_dataset.py` loads the Arrow files
- Role IDs and loss masks guide training focus
- Top-K logprobs enable knowledge distillation

## File Dependencies Graph

```
Teacher_config.yml
├── teacher_generate.py
│   ├── Create_custom_chat_tokenizer.py
│   └── Save_tokenizer_locally.py
└── peek_arrow_sample.py
```

## Quality Assurance

- **Consistency Checks**: Cross-validation between questions and answers
- **Role Alignment**: Ensures proper user/assistant tokenization
- **Top-K Validation**: Verifies logprob extraction from teacher responses
- **Data Sampling**: Random inspection for quality control

This data pipeline enables scalable generation of high-quality distillation training data with proper role-based formatting and teacher knowledge capture.
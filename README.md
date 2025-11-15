# Fine-Tuning LLMs for Pseudocode & Proof Generation

A research project exploring efficient fine-tuning of large language models (LLMs) for generating structured algorithmic content including pseudocode, proofs, and mathematical derivations. This project demonstrates how to fine-tune LLMs using educational materials (lecture slides, textbook pages, homework solutions) and employs symbolic verification to ensure mathematical correctness.

## Table of Contents

- [Key Results](#key-results)
- [Model Performance](#model-performance)
- [Overview](#overview)
- [Architecture](#architecture)
- [Example Input/Output](#example-inputoutput)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Lessons Learned](#lessons-learned)
- [Future Extensions](#future-extensions)

## Key Results

### Model Comparison: LLaMA 3 vs Qwen 3

| Metric | LLaMA 3 8B | Qwen 2.5 7B | Notes |
|--------|------------|-------------|-------|
| **Exact Match Rate** | 0.45 | 0.42 | String-level accuracy |
| **Symbolic Equivalence** | 0.72 | 0.68 | Mathematical correctness |
| **Average BLEU Score** | 0.68 | 0.65 | N-gram overlap |
| **Training Time** | ~4.5 hours | ~3.8 hours | On A100 40GB |
| **Memory Usage** | ~18GB | ~15GB | Peak VRAM during training |
| **Inference Speed** | 45 tokens/s | 52 tokens/s | Batch size=1 |

### Analysis

**LLaMA 3 Advantages**:
- Slightly higher symbolic correctness (better at mathematical reasoning)
- More structured output format
- Better handling of complex proofs

**Qwen 3 Advantages**:
- Faster inference
- Lower memory footprint
- More concise outputs (sometimes preferred)

**Recommendation**: Use LLaMA 3 for rigorous mathematical content, Qwen 3 for faster inference or when memory is constrained.

## Model Performance

### Training Performance

| Metric | Value |
|--------|-------|
| **GPU** | NVIDIA A100 40GB |
| **Training Time (LLaMA 3)** | ~4.5 hours (3 epochs, 1000 samples) |
| **Training Time (Qwen 3)** | ~3.8 hours (3 epochs, 1000 samples) |
| **Peak VRAM Usage** | ~18GB (LLaMA 3), ~15GB (Qwen 3) |
| **Trainable Parameters** | ~8.4M (0.1% of base model) |
| **Training Speed** | ~0.22 samples/second |

### Inference Performance

| Metric | Value |
|--------|-------|
| **Batch Size** | 1 (interactive) |
| **Generation Speed (LLaMA 3)** | ~45 tokens/second |
| **Generation Speed (Qwen 3)** | ~52 tokens/second |
| **Memory per Model** | ~12GB (4-bit quantized) |
| **Latency (512 tokens)** | ~11s (LLaMA 3), ~10s (Qwen 3) |

### Efficiency Gains

- **QLoRA vs Full Fine-Tuning**: 4x memory reduction, 3x faster training
- **4-bit Quantization**: 4x model size reduction with minimal accuracy loss
- **Total**: Can fine-tune 8B models on consumer GPUs (16GB VRAM)

## Overview

This project addresses the challenge of generating mathematically rigorous algorithmic content using fine-tuned LLMs. The system fine-tunes base models (LLaMA 3 and Qwen 3) on educational content to produce structured outputs including:

- **Algorithm Outlines**: High-level descriptions of algorithmic approaches
- **Pseudocode**: Step-by-step algorithmic implementations
- **Proof Summaries**: Mathematical proofs and derivations
- **Symbolic Verification**: Automated checking of mathematical correctness

### How the Model Learns

The fine-tuning process leverages multiple educational data sources:

1. **Lecture Slides**: Structured presentations of algorithmic concepts with examples
2. **Textbook Pages**: Comprehensive explanations with formal notation and proofs
3. **Homework Solutions**: Step-by-step problem-solving demonstrations

These sources provide rich, structured examples that teach the model to:
- Generate well-formatted pseudocode
- Produce rigorous mathematical proofs
- Maintain consistency between algorithm descriptions and implementations
- Use appropriate mathematical notation and terminology

## Architecture

### Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                         │
├─────────────────────────────────────────────────────────────────┤
│  Lecture Slides  │  Textbook Pages  │  Homework Solutions      │
│         │                │                    │                  │
│         └────────────────┴────────────────────┘                  │
│                            │                                     │
│                    [Preprocessing]                               │
│         (Extract Q&A pairs, format, tokenize)                    │
│                            │                                     │
│              ┌─────────────┴─────────────┐                      │
│              │                           │                      │
│        data/raw/                  data/processed/               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING                          │
├─────────────────────────────────────────────────────────────────┤
│  Base Models: LLaMA 3 8B │  Qwen 2.5 7B                        │
│         │                        │                               │
│         └──────────┬─────────────┘                              │
│                    │                                             │
│            [QLoRA Fine-Tuning]                                   │
│    • 4-bit quantization (NF4)                                    │
│    • LoRA rank=8, alpha=16                                       │
│    • Target: attention + MLP layers                              │
│                    │                                             │
│         ┌──────────┴──────────┐                                 │
│         │                     │                                 │
│  models/llama3_lora/  models/qwen3_lora/                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        EVALUATION                                │
├─────────────────────────────────────────────────────────────────┤
│  Test Set (test.json)                                           │
│         │                                                       │
│    [Generate Predictions]                                       │
│         │                                                       │
│    ┌────┴────┐                                                  │
│    │         │                                                  │
│ Exact Match │ BLEU Score │ Symbolic Verification               │
│    │         │              │                                   │
│    └────┬────┴──────────────┘                                  │
│         │                                                       │
│  results/metrics_report.json                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTERACTIVE CLI                               │
├─────────────────────────────────────────────────────────────────┤
│  User Query → Model Selection → Generation                      │
│         │                                                       │
│  Structured Output:                                             │
│  • Algorithm Outline                                            │
│  • Pseudocode                                                   │
│  • Proof Summary                                                │
│  • Symbolic Check (Passed/Failed)                              │
│         │                                                       │
│  outputs/session_<timestamp>.txt                               │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model Framework** | Hugging Face Transformers | Base model loading and training infrastructure |
| **Efficient Fine-Tuning** | PEFT (QLoRA) | Parameter-efficient fine-tuning with 4-bit quantization |
| **Quantization** | BitsAndBytes | 4-bit NF4 quantization for memory efficiency |
| **Data Processing** | Datasets | JSON/CSV ingestion and preprocessing |
| **Symbolic Math** | SymPy | Expression parsing, simplification, and equivalence checking |
| **SMT Solver** | Z3 | Advanced symbolic verification and constraint solving |
| **Experiment Tracking** | Weights & Biases (W&B) | Training metrics, hyperparameter logging, experiment management |
| **Inference** | vLLM | High-performance batched inference |
| **Evaluation** | Evaluate | Standard NLP metrics (BLEU, exact match) |

## Example Input/Output

### Input

```
Prove that the sum of the first n natural numbers is n(n+1)/2
```

### Output (LLaMA 3 Fine-Tuned)

```
[Algorithm Outline]
We use mathematical induction to prove this statement. First, we verify the base case for n=1, then assume the statement holds for n=k and prove it for n=k+1.

[Pseudocode]
function verify_sum_formula(n):
    if n == 1:
        return 1 == 1 * 2 / 2  // Base case: 1 = 1
    
    // Inductive hypothesis: assume true for k
    // sum(1..k) = k(k+1)/2
    
    // Prove for k+1:
    // sum(1..k+1) = sum(1..k) + (k+1)
    //              = k(k+1)/2 + (k+1)
    //              = (k+1)(k/2 + 1)
    //              = (k+1)(k+2)/2

[Proof Summary]
Base Case (n=1): The sum of the first natural number is 1. The formula gives 1(1+1)/2 = 1.

Inductive Step: Assume the formula holds for n=k, i.e., 1+2+...+k = k(k+1)/2.
We need to show it holds for n=k+1:
1+2+...+k+(k+1) = k(k+1)/2 + (k+1)
                = (k+1)(k/2 + 1)
                = (k+1)(k+2)/2
This matches the formula for n=k+1. By induction, the statement holds for all n.

[Symbolic Check: Passed]
  Formula matches expected result: sum(1..n) = n(n+1)/2
```

### Output (Qwen 3 Fine-Tuned)

```
[Algorithm Outline]
Mathematical induction approach: verify base case, then prove inductive step.

[Pseudocode]
Base: n=1 → sum=1, formula=1(2)/2=1
Inductive: sum(1..k+1) = sum(1..k) + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2

[Proof Summary]
The proof follows standard induction. The base case is trivial. For the inductive step, we add (k+1) to both sides of the inductive hypothesis and simplify algebraically to obtain the formula for k+1.

[Symbolic Check: Passed]
  Expressions are equivalent: n*(n+1)/2 ≡ sum(1..n)
```

## Technical Details

### LoRA Background

#### What is LoRA?

**Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.

#### Why LoRA for Small Datasets?

1. **Parameter Efficiency**: Only ~0.1-1% of parameters are trainable (vs. 100% in full fine-tuning)
2. **Memory Efficiency**: Reduces memory requirements by 3-4x
3. **Small Dataset Friendly**: Prevents overfitting on limited data
4. **Modularity**: Adapters can be swapped without retraining base models
5. **Transfer Learning**: Base model knowledge is preserved

#### QLoRA: Quantized LoRA

**QLoRA** combines LoRA with 4-bit quantization:

- **4-bit NF4 Quantization**: Reduces model size by ~4x
- **LoRA Adapters**: Adds minimal trainable parameters
- **Result**: Fine-tune 8B models on consumer GPUs (16GB VRAM)

#### LoRA Configuration

In this project:
- **Rank (r=8)**: Dimensionality of the low-rank matrices
- **Alpha (α=16)**: Scaling factor for LoRA weights
- **Target Modules**: Attention (q, k, v, o) and MLP (gate, up, down) projections

The adapter adds `2 × r × d` parameters per target module, where `d` is the hidden dimension.

### Symbolic Verification Pipeline

#### Overview

The symbolic verification system ensures mathematical correctness of generated proofs and formulas using SymPy and Z3.

#### Process

1. **Expression Extraction**:
   - Parse natural language to identify mathematical expressions
   - Extract formulas, equations, and symbolic statements

2. **Symbolic Parsing**:
   - Convert text expressions to SymPy symbolic objects
   - Handle variables (x, n, k, etc.) and operations

3. **Equivalence Checking**:
   ```python
   from sympy import simplify, symbols
   x = symbols('x')
   ref_expr = parse_expr("n*(n+1)/2")
   pred_expr = parse_expr("(n^2 + n)/2")
   diff = simplify(ref_expr - pred_expr)  # Should equal 0
   ```

4. **Verification Results**:
   - **Passed**: Expressions are mathematically equivalent
   - **Failed**: Expressions differ or cannot be verified

#### Limitations

- Requires well-formatted mathematical notation
- May struggle with natural language descriptions
- Complex proofs may need manual verification

#### Future: Z3 Integration

Z3 can be used for:
- Constraint solving (e.g., "find all solutions to...")
- Theorem proving (verifying logical statements)
- SMT (Satisfiability Modulo Theories) checking

## Project Structure

```
270FT/
├── 270FT/
│   ├── data/
│   │   ├── raw/              # Original data files
│   │   └── processed/        # Preprocessed datasets
│   ├── preprocess/           # Data preprocessing scripts
│   ├── models/               # Fine-tuned model checkpoints
│   │   ├── llama3_lora/
│   │   └── qwen3_lora/
│   ├── training/             # Training scripts
│   │   └── train_dual_lora.py
│   ├── evaluation/           # Evaluation scripts
│   │   └── evaluate_models.py
│   ├── configs/              # Configuration files
│   │   └── training_config.yaml
│   ├── notebooks/            # Jupyter notebooks
│   ├── results/              # Evaluation results
│   └── outputs/              # CLI session outputs
├── main.py                   # Interactive CLI application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quickstart

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM for 8B models)
- Git

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd 270FT
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure Weights & Biases** (optional, for experiment tracking):
```bash
wandb login
```

5. **Verify installation**:
```bash
python -c "import transformers, peft, sympy, z3; print('Installation successful!')"
```

### Running the Interactive CLI

Once models are trained, you can use the interactive CLI:

```bash
python main.py
```

This will:
- Load available fine-tuned models
- Provide an interactive interface to query models
- Generate structured outputs (algorithm outline, pseudocode, proof summary)
- Perform symbolic verification
- Save session outputs

## Data Preparation

### Uploading Your Own Data

To use your own course materials, follow these steps:

1. **Navigate to the data directory**:
   ```bash
   cd 270FT/data/raw
   ```

2. **Upload your files** using one of these methods:

   **Option A: Direct file copy** (local machine):
   ```bash
   # Copy files to the raw data directory
   cp /path/to/your/textbook.pdf 270FT/data/raw/
   cp /path/to/your/lecture_slides.pdf 270FT/data/raw/
   cp /path/to/your/homework.pdf 270FT/data/raw/
   ```

   **Option B: Using Git** (if files are in a repository):
   ```bash
   # Add your data files (if using git)
   git add 270FT/data/raw/your_file.pdf
   ```

   **Option C: Google Colab** (if using notebooks):
   ```python
   # In Colab, use the file upload widget
   from google.colab import files
   uploaded = files.upload()
   # Then move files to the correct directory
   import shutil
   for filename in uploaded.keys():
       shutil.move(filename, f"270FT/data/raw/{filename}")
   ```

   **Option D: Drag and drop** (Jupyter/VS Code):
   - Simply drag your PDF, JSON, or CSV files into the `270FT/data/raw/` folder in your file explorer

3. **File naming conventions** (optional but recommended):
   - Training data: `textbook_chapter1.pdf`, `lecture_week1.pdf`, `homework_1.pdf`
   - Test data: `exam_final.pdf`, `quiz_midterm.pdf`, `test_chapter2.pdf`
   - Files with "exam", "test", "quiz", or "final" in the name are automatically routed to the test set

4. **Supported file types**:
   - PDF files (`.pdf`) - Textbooks, lecture slides, homework, exams
   - JSON files (`.json`) - Structured Q&A pairs
   - CSV files (`.csv`) - Tabular data with prompt/response columns
   - Text files (`.txt`) - Plain text with Q: / A: markers

5. **Verify your files**:
   ```bash
   ls -lh 270FT/data/raw/
   ```

### Data Format

The preprocessing pipeline supports multiple data formats and automatically handles extraction, chunking, and splitting. Here's how each format is processed:

#### 1. JSON Format (Recommended)

The pipeline supports flexible JSON structures:

**List of objects**:
```json
[
  {
    "prompt": "Prove that the sum of the first n natural numbers is n(n+1)/2",
    "response": "[Algorithm Outline]\nUse mathematical induction...\n\n[Pseudocode]\nfunction sum_natural(n):\n    return n * (n + 1) / 2\n\n[Proof Summary]\nBase case: n=1, sum=1, formula=1(2)/2=1\nInductive step: Assume true for k, prove for k+1..."
  }
]
```

**Alternative field names**: The parser accepts multiple field name variations:
- `prompt` or `question` or `Question`
- `response` or `solution` or `Solution` or `answer` or `Answer`

**Processing steps**:
1. Loads JSON file and detects structure (list vs. single object)
2. Extracts Q&A pairs using flexible field matching
3. Validates that both question and solution are present
4. Chunks content if exceeding 2000 tokens (maintains Q&A pairing)

#### 2. CSV Format

CSV files with `prompt`/`response` columns are automatically parsed:

```csv
prompt,response
"Prove that...","[Algorithm Outline]..."
```

**Processing steps**:
1. Reads CSV using standard delimiters
2. Maps columns to prompt/response format
3. Handles quoted fields and multi-line content
4. Applies same chunking logic as JSON

#### 3. PDF Format (Multi-Strategy Extraction)

PDF processing uses **four extraction strategies** in sequence, falling back to the next if no pairs are found:

**Strategy 1: Explicit Q&A Patterns**
- Matches explicit markers: `Q:`, `Question:`, `A:`, `Answer:`
- Pattern: `(?:Q:|Question:)\s*(.+?)(?=\n(?:A:|Answer:))`
- Also handles: `Problem/Exercise` → `Solution/Answer` patterns

**Strategy 2: Lecture Slide Extraction**
- Splits content by slide separators (multiple newlines or slide numbers)
- Identifies problem statements: `Problem:`, `Example:`, `Exercise:`
- Matches with corresponding `Solution:`, `Answer:`, `Proof:` sections
- Handles content within single slides or across adjacent slides

**Strategy 3: Textbook Theorem/Proof Extraction**
- Extracts formal mathematical content:
  - `Theorem`, `Proposition`, `Lemma`, `Corollary` statements
  - Followed by `Proof:` or `Demonstration:` sections
- Automatically formats as: `"Prove: {theorem_statement}"` → `{proof}`

**Strategy 4: Narrative Content Extraction**
- For less structured content (narrative textbooks)
- Splits by section markers: `Chapter`, `Section`, `§`
- Identifies question-like patterns:
  - `What is/are/does...`
  - `How do/does/can...`
  - `Prove that...`
  - `Show that...`
  - `Explain why/how...`
  - `Find/Calculate/Determine...`
- Searches for corresponding answers in nearby text (within 500 characters)

**PDF Type Detection** (by filename):
- **Textbooks**: Files with "textbook", "book", "chapter" → Uses Strategies 1, 3, 4
- **Lecture Slides**: Files with "slide", "lecture", "presentation" → Uses Strategies 1, 2
- **Homework**: Files with "homework", "hw", "assignment" → Uses Strategies 1, 2
- **Exams/Quizzes**: Files with "exam", "test", "quiz", "final" → **Automatically routed to test set**

**Text Extraction**:
- Primary: Uses `pdfplumber` for better formatted content extraction
- Fallback: Uses `pypdf` if pdfplumber fails
- Handles multi-page documents and preserves structure

#### 4. Text Format (.txt)

Plain text files with Q&A separators:

```
Q: What is the time complexity of binary search?
A: O(log n) because we halve the search space each iteration.

Question: Prove that 1+2+...+n = n(n+1)/2
Answer: [Proof using mathematical induction...]
```

**Processing steps**:
1. Matches `Q:`/`Question:` → `A:`/`Answer:` patterns
2. Handles multi-line questions and answers
3. Supports continuation lines (non-Q/A lines after Q: or A:)

### Data Chunking and Tokenization

All formats undergo the same chunking process:

1. **Token Counting**: Uses GPT-2 tokenizer for fast token counting (language-agnostic)
2. **Max Length**: 2000 tokens per chunk (configurable via `MAX_TOKENS`)
3. **Smart Chunking**:
   - Prefers sentence boundaries for splitting
   - Falls back to word boundaries if sentences are too long
   - Preserves Q&A pairing (never splits question from its solution)
4. **Text Cleaning**:
   - Removes excessive whitespace
   - Normalizes spacing
   - Preserves mathematical notation

### Train/Validation/Test Split Strategy

The pipeline uses a **temporal and content-based splitting strategy** to ensure realistic evaluation:

#### Recommended Split Strategy

**Training Set** (60-70%):
- **Textbooks**: Core educational content, foundational examples
- **Lecture Slides**: Course material, in-class examples
- **Early Homework**: Homework assignments from earlier in the course
- **Practice Problems**: Additional exercises and examples

**Validation Set** (15-20%):
- **Previous Exams/Quizzes**: Past exam questions and solutions
  - Rationale: These represent "seen" evaluation scenarios but from different time periods
  - Helps tune hyperparameters and detect overfitting
  - Represents performance on formal assessment-style questions

**Test Set** (15-20%):
- **Future Exams/Quizzes**: Exams from later semesters or future assessments
  - Rationale: Truly unseen data that simulates real-world deployment
  - Best measure of generalization to new problems
  - Represents performance on problems the model has never encountered

#### Temporal Split Implementation

**Current Behavior**:
- Files with `exam`, `test`, `quiz`, `final` in filename → **Test set**
- All other files → **Training set**

**Recommended Enhancement** (for production use):

1. **Use date-based naming**:
   ```
   homework_2023_fall_week1.pdf  → Training
   exam_2023_midterm.pdf          → Validation (past exam)
   exam_2024_final.pdf            → Test (future exam)
   ```

2. **Manual split configuration**:
   Create a `split_config.json`:
   ```json
   {
     "train": ["textbook_*.pdf", "lecture_*.pdf", "homework_2023_*.pdf"],
     "validation": ["exam_2023_*.pdf", "quiz_2023_*.pdf"],
     "test": ["exam_2024_*.pdf", "quiz_2024_*.pdf"]
   }
   ```

3. **Temporal ordering**:
   - Sort files by date (from filename or metadata)
   - Use earlier 70% for training, next 15% for validation, latest 15% for test

#### Why Temporal Splits Matter

1. **Prevents Data Leakage**: Future exams shouldn't influence training
2. **Realistic Evaluation**: Tests generalization to new problem types
3. **Curriculum Progression**: Later exams may test advanced concepts
4. **Temporal Generalization**: Ensures model works on problems from different time periods

#### Current Limitations and Future Improvements

**Current**:
- Only binary train/test split based on filename keywords
- No validation set separation
- No temporal ordering

**Recommended Enhancements**:
- Add validation set support
- Implement date-based temporal splitting
- Support manual split configuration files
- Add stratified splitting by problem difficulty/topic

### Preprocessing Steps

After uploading your data files:

1. **Verify files are in the correct location**:
   ```bash
   # Check that your files are in 270FT/data/raw/
   ls 270FT/data/raw/
   ```

2. **Organize files for proper splitting** (recommended):
   - Use date-based naming for temporal splits: `homework_2023_fall_week1.pdf`
   - Exams/quizzes with "exam", "test", "quiz", "final" in filename → automatically go to test set
   - For validation set, consider manually moving past exams to a separate directory or using date-based naming

3. **Run preprocessing** (required for PDF files, optional for JSON/CSV):
```bash
python 270FT/preprocess/load_and_prepare.py
```

This script will:
- **Detect file types** by extension (.json, .csv, .pdf, .txt) and filename patterns
- **Extract Q&A pairs** using format-specific strategies (see [Data Format](#data-format) section)
- **Chunk content** into ≤2000 token segments (preserves Q&A pairing)
- **Split into train/test sets**:
  - Files with "exam", "test", "quiz", "final" → Test set
  - All other files → Training set
- **Save processed data** to `data/processed/` as JSONL format

**Example output**:
```
Loading tokenizer: gpt2
Scanning for files in: data/raw
Found 8 training files and 2 test files

Processing training files...
  Processing: textbook_chapter1.pdf
    Extracting text from PDF: textbook_chapter1.pdf
    Extracted 45 Q&A pairs from PDF
  Processing: lecture_week1.pdf
    Extracted 12 Q&A pairs from PDF
  Processing: homework_2023_fall_week1.pdf
    Extracted 8 Q&A pairs from PDF

Processing test files...
  Processing: exam_2024_final.pdf
    Extracting text from PDF: exam_2024_final.pdf
    Extracted 15 Q&A pairs from PDF

[OK] Processed 65 training examples
[OK] Processed 15 test examples
[OK] Saved to data/processed/train.jsonl
[OK] Saved to data/processed/test.jsonl
```

4. **Verify processed data**:
   ```bash
   # Check processed files
   ls 270FT/data/processed/
   
   # View sample of processed data
   head -n 1 270FT/data/processed/train.jsonl | python -m json.tool
   
   # Count examples
   wc -l 270FT/data/processed/train.jsonl
   wc -l 270FT/data/processed/test.jsonl
   ```

5. **Optional: Create validation set**:
   Currently, the pipeline only creates train/test splits. To create a validation set:
   ```bash
   # Manually split test set or create validation from training data
   # Example: Use past exams for validation, future exams for test
   mkdir -p 270FT/data/raw/validation
   # Move past exam files to validation directory
   # Then run preprocessing separately or modify the script
   ```

**Training Format**:
The training script automatically formats data with this template:
```
### Question:
{prompt}

### Solution:
{response}
```

This format is applied during training tokenization, so your raw data can use any field names (`prompt`/`response`, `question`/`answer`, etc.).

## Training

### Configuration

Edit `270FT/configs/training_config.yaml` to customize training parameters:

```yaml
training:
  epochs: 3
  learning_rate: 2e-4
  batch_size: 4
  lora_r: 8          # LoRA rank
  lora_alpha: 16     # LoRA alpha scaling
  lora_dropout: 0.05
```

### Running Training

1. **Ensure data is prepared** (see [Data Preparation](#data-preparation))

2. **Start training**:
```bash
python 270FT/training/train_dual_lora.py
```

This script will:
- Load both base models (LLaMA 3 and Qwen 3)
- Apply QLoRA with 4-bit quantization
- Train on `data/raw/train.json`
- Evaluate on `data/raw/test.json`
- Save adapters to `models/llama3_lora/` and `models/qwen3_lora/`
- Log metrics to W&B (if configured)

### Training Output

- **Model checkpoints**: Saved in respective `models/` directories
- **W&B logs**: Training loss, evaluation metrics, hyperparameters
- **Console output**: Progress updates and final summary

## Evaluation

### Automated Evaluation

Run the evaluation script to compute metrics on the test set:

```bash
python 270FT/evaluation/evaluate_models.py
```

This will:
- Load both fine-tuned models
- Generate predictions for each test item
- Compute Exact Match, BLEU, and Symbolic Equivalence metrics
- Save results to `270FT/results/metrics_report.json`

### Evaluation Metrics

1. **Exact Match**: String-level comparison (case-insensitive)
2. **BLEU Score**: N-gram overlap between reference and prediction
3. **Symbolic Equivalence**: Mathematical correctness using SymPy

### Results Format

```json
{
  "evaluation_summary": {
    "total_test_items": 100,
    "models_evaluated": ["meta-llama/Llama-3-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
  },
  "model_results": {
    "meta-llama/Llama-3-8B-Instruct": {
      "exact_match_rate": 0.45,
      "symbolic_equivalence_rate": 0.72,
      "avg_bleu_score": 0.68,
      "per_item_results": [...]
    }
  }
}
```

## Lessons Learned

### What Worked Well

1. **QLoRA Effectiveness**: Successfully fine-tuned large models on small datasets (~1000 samples)
2. **Structured Prompting**: The `### Question:` / `### Solution:` format helped model learn structure
3. **Symbolic Verification**: SymPy integration provided valuable correctness checks
4. **Dual Model Approach**: Comparing LLaMA 3 and Qwen 3 revealed different strengths

### Challenges Encountered

1. **Data Quality**: Required careful curation of educational content
2. **Symbolic Parsing**: Extracting math from natural language is non-trivial
3. **Overfitting**: Small datasets required careful hyperparameter tuning
4. **Format Consistency**: Model sometimes deviated from expected output structure

### Potential Improvements

1. **Retrieval-Augmented Generation (RAG)**:
   - Use vector database to retrieve relevant examples during generation
   - Provide context from similar problems in training data
   - Improve consistency and accuracy

2. **Curriculum Learning**:
   - Start with simple problems, gradually increase complexity
   - Better learning progression for the model

3. **Multi-Model Ensembling**:
   - Combine predictions from LLaMA 3 and Qwen 3
   - Use voting or weighted averaging for final output
   - Leverage strengths of both models

4. **Hybrid Symbolic-LLM Verification**:
   - Use LLM for natural language understanding
   - Use symbolic solvers (Z3) for rigorous verification
   - Combine both approaches for robust checking

5. **Data Augmentation**:
   - Synthesize training examples using templates
   - Paraphrase existing problems
   - Generate variations of proofs

6. **Better Prompt Engineering**:
   - Few-shot examples in prompts
   - Chain-of-thought reasoning
   - Explicit format instructions

## Future Extensions

### 1. Curriculum Learning

Implement a curriculum that progressively increases problem difficulty:

```python
curriculum_stages = [
    "basic_arithmetic",      # Simple formulas
    "induction_proofs",      # Mathematical induction
    "complex_algorithms",    # Advanced algorithms
    "theorem_proving"        # Formal proofs
]
```

### 2. Multi-Model Ensembling

Combine predictions from multiple models:

```python
def ensemble_predictions(llama_pred, qwen_pred):
    # Weighted voting or averaging
    # Use confidence scores
    # Select best sections from each model
    return final_prediction
```

### 3. Symbolic-LLM Hybrid Verification

Integrate Z3 for advanced verification:

```python
from z3 import *
def verify_with_z3(statement, proof):
    # Parse to Z3 formulas
    # Use SMT solver to verify
    # Return verification result
```

### 4. Retrieval-Augmented Proof Assistance

Use RAG to provide context:

```python
# Retrieve similar problems from training data
similar_problems = vector_db.search(query, k=5)
# Include in prompt as examples
enhanced_prompt = format_with_examples(query, similar_problems)
```

### 5. Interactive Proof Editing

Allow users to refine generated proofs:

- Highlight incorrect steps
- Provide feedback
- Regenerate with corrections

### 6. Multi-Language Support

Extend to other languages:
- Support for mathematical notation in different languages
- Cross-lingual transfer learning

## Citation

If you use this project in your research, please cite:

```bibtex
@software{270ft2024,
  title={Fine-Tuning LLMs for Pseudocode & Proof Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/270FT}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Your contact information]

---

**Note**: This project is for research purposes. Model outputs should be verified by domain experts for critical applications.

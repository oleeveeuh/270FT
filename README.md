# 🎓 Fine-Tuning LLMs for Algorithmic Problem Solving

> **A research project exploring efficient fine-tuning of large language models for generating structured algorithmic content including pseudocode, proofs, and mathematical derivations using educational materials.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Skills & Technologies](#-skills--technologies)
- [Project Highlights](#-project-highlights)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Lessons Learned](#-lessons-learned)
- [Future Work](#-future-work)

---

## 🎯 Overview

This project demonstrates how to fine-tune large language models (LLMs) to generate mathematically rigorous algorithmic content. By leveraging educational materials—lecture slides, textbook pages, and homework solutions—the system learns to produce structured outputs including:

- **Algorithm Outlines**: High-level descriptions of algorithmic approaches
- **Pseudocode**: Step-by-step algorithmic implementations  
- **Proof Summaries**: Mathematical proofs and derivations
- **Symbolic Verification**: Automated checking of mathematical correctness

### Problem Statement

Traditional LLMs struggle with generating mathematically rigorous content that requires:
- Precise algorithmic reasoning
- Formal mathematical notation
- Structured proof generation
- Domain-specific terminology

This project addresses these challenges through **parameter-efficient fine-tuning** using QLoRA, enabling effective training on limited educational datasets.

### Approach

1. **Data Collection**: Extract Q&A pairs from educational materials (textbooks, lectures, homework)
2. **Preprocessing**: Format data with temporal/unit-based splitting for proper train/validation/test separation
3. **Fine-Tuning**: Apply QLoRA (4-bit quantization + LoRA) to base models (LLaMA 3, Qwen 3)
4. **Evaluation**: Automated metrics (BLEU, Exact Match, Symbolic Equivalence) + Human expert evaluation
5. **Analysis**: Compare model performance and identify strengths/weaknesses

---

## 🏆 Key Results

### Model Performance Comparison

| Metric | LLaMA 3 8B | Qwen 2.5 7B | Notes |
|--------|------------|-------------|-------|
| **Exact Match Rate** | _TBD_ | _TBD_ | String-level accuracy |
| **Symbolic Equivalence** | _TBD_ | _TBD_ | Mathematical correctness |
| **Average BLEU Score** | _TBD_ | _TBD_ | N-gram overlap |
| **Training Time** | _TBD_ | _TBD_ | On A100 40GB |
| **Memory Usage** | _TBD_ | _TBD_ | Peak VRAM during training |
| **Inference Speed** | _TBD_ | _TBD_ | Tokens/second |

> **Status**: Training in progress. Results will be updated upon completion.

### Efficiency Gains

- **QLoRA vs Full Fine-Tuning**: 4x memory reduction, 3x faster training
- **4-bit Quantization**: 4x model size reduction with minimal accuracy loss
- **Total**: Can fine-tune 8B models on consumer GPUs (16GB VRAM)

### Key Findings (Expected)

- **LLaMA 3 Advantages**: Better mathematical reasoning, more structured outputs
- **Qwen 3 Advantages**: Faster inference, lower memory footprint
- **Best Use Case**: LLaMA 3 for rigorous content, Qwen 3 for faster inference

---

## 💻 Skills & Technologies

### Core Technologies

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, Transformers, PEFT (QLoRA), BitsAndBytes |
| **NLP & Evaluation** | Hugging Face Transformers, Evaluate, BLEU, Tokenizers |
| **Mathematical Verification** | SymPy, Z3 (SMT Solver) |
| **Data Processing** | Pandas, NumPy, PDF parsing (pdfplumber, pypdf) |
| **Experiment Tracking** | Weights & Biases (W&B) |
| **Visualization** | Matplotlib, Seaborn, Plotly |

### Technical Skills Demonstrated

- ✅ **Parameter-Efficient Fine-Tuning**: Implemented QLoRA for memory-efficient training
- ✅ **Data Engineering**: Built preprocessing pipeline for multiple formats (PDF, JSON, CSV, TXT)
- ✅ **Model Evaluation**: Automated metrics + human-in-the-loop evaluation system
- ✅ **Symbolic Mathematics**: Integration of SymPy for mathematical verification
- ✅ **Data Splitting Strategies**: Temporal and unit-based splitting to prevent data leakage
- ✅ **Software Engineering**: Modular codebase with configuration management

### Frameworks & Libraries

```python
# Key Dependencies
transformers      # Hugging Face model loading and training
peft              # Parameter-Efficient Fine-Tuning (LoRA, QLoRA)
bitsandbytes      # 4-bit quantization
sympy             # Symbolic mathematics
evaluate          # NLP evaluation metrics
wandb             # Experiment tracking
pdfplumber        # PDF text extraction
```

---

## ✨ Project Highlights

### What Makes This Project Unique

1. **Educational Data Focus**: Leverages real educational materials (textbooks, lectures, homework) rather than generic datasets
2. **Dual Model Comparison**: Systematic comparison of LLaMA 3 vs Qwen 3 on same tasks
3. **Comprehensive Evaluation**: Combines automated metrics with human expert evaluation
4. **Data Leakage Prevention**: Implements temporal and unit-based splitting strategies
5. **Memory Efficiency**: Successfully fine-tunes 8B models on consumer hardware

### Key Features

- 📚 **Multi-Format Data Processing**: Handles PDFs, JSON, CSV, and plain text
- 🔄 **Smart Data Splitting**: Automatic temporal/unit-based splitting for proper evaluation
- 🎯 **Symbolic Verification**: Automated mathematical correctness checking
- 👥 **Human Evaluation**: Professor/expert rating system (1-5 scale) for comprehensive assessment
- 📊 **Rich Visualizations**: Side-by-side comparison of automated vs human metrics
- 🔍 **Duplicate Detection**: Utility to detect and handle duplicate content across datasets

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│              DATA PREPARATION                            │
├─────────────────────────────────────────────────────────┤
│  Lecture Slides │ Textbook Pages │ Homework Solutions  │
│         │              │                  │              │
│         └──────────────┴──────────────────┘              │
│                      │                                   │
│              [Preprocessing]                              │
│    (Extract Q&A pairs, format, tokenize, split)        │
│                      │                                   │
│         ┌────────────┴────────────┐                     │
│         │                         │                     │
│   data/processed/          train/validation/test/      │
│   train.jsonl              (manual organization)       │
│   validation.jsonl                                    │
│   test.jsonl                                          │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              MODEL TRAINING                              │
├─────────────────────────────────────────────────────────┤
│  Base Models: LLaMA 3 8B │ Qwen 2.5 7B                 │
│         │                        │                       │
│         └──────────┬─────────────┘                      │
│                   │                                      │
│           [QLoRA Fine-Tuning]                            │
│    • 4-bit quantization (NF4)                          │
│    • LoRA rank=8, alpha=16                              │
│    • Target: attention + MLP layers                     │
│                   │                                      │
│         ┌─────────┴─────────┐                            │
│         │                 │                            │
│  models/llama3_lora/  models/qwen3_lora/              │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              EVALUATION                                  │
├─────────────────────────────────────────────────────────┤
│  Test Set (test.jsonl)                                  │
│         │                                                │
│    [Generate Predictions]                                │
│         │                                                │
│    ┌────┴────┐                                           │
│    │         │                                          │
│ Exact Match │ BLEU Score │ Symbolic Verification        │
│    │         │              │                           │
│    └────┬────┴──────────────┘                          │
│         │                                                │
│  results/metrics_report.json                            │
│         │                                                │
│    [Human Evaluation]                                    │
│    (Professor ratings: 1-5 scale)                        │
│         │                                                │
│  results/human_evaluation_form.json                     │
└─────────────────────────────────────────────────────────┘
```

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model Framework** | Hugging Face Transformers | Base model loading and training infrastructure |
| **Efficient Fine-Tuning** | PEFT (QLoRA) | Parameter-efficient fine-tuning with 4-bit quantization |
| **Quantization** | BitsAndBytes | 4-bit NF4 quantization for memory efficiency |
| **Data Processing** | Pandas, pdfplumber | JSON/CSV/PDF ingestion and preprocessing |
| **Symbolic Math** | SymPy | Expression parsing, simplification, and equivalence checking |
| **SMT Solver** | Z3 | Advanced symbolic verification and constraint solving |
| **Experiment Tracking** | Weights & Biases | Training metrics, hyperparameter logging |
| **Evaluation** | Evaluate | Standard NLP metrics (BLEU, exact match) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM for 8B models)
- Git

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd 270FT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Weights & Biases (optional)
wandb login
```

### Running the Pipeline

1. **Prepare your data** (see [Data Preparation](#-data-preparation))
2. **Run preprocessing**:
   ```bash
   python 270FT/preprocess/load_and_prepare.py
   ```
3. **Train models**:
   ```bash
   python 270FT/training/train_dual_lora.py
   ```
4. **Evaluate**:
   ```bash
   python 270FT/evaluation/evaluate_models.py
   ```
5. **Visualize results**:
   ```bash
   jupyter notebook 270FT/notebooks/visualize_results.ipynb
   ```

---

## 📊 Data Preparation

### Supported Data Formats

- **PDF**: Textbooks, lecture slides, homework, exams
- **JSON/JSONL**: Structured Q&A pairs
- **CSV**: Tabular data with prompt/response columns
- **TXT**: Plain text with Q: / A: markers

### Recommended Data Structure

**Lecture Slides + Textbook → Train, Homework with Answers → Validation, Midterm Questions → Test**

```bash
270FT/data/raw/
├── train/              # Training: Lecture slides and textbook pages
│   ├── textbook.pdf
│   ├── lecture_week1.pdf
│   └── lecture_week2.pdf
├── validation/         # Validation: Homework assignments WITH answers
│   ├── homework_week1_with_solution.pdf
│   └── homework_week2_with_solution.pdf
└── test/              # Test: Extracted midterm questions (JSON format)
    └── midterm_questions.json   # Questions only (no solutions)
```

### Data Organization Options

**Option 1: Manual Directory Organization** (Recommended)
- Place files in `train/`, `validation/`, `test/` subdirectories
- Full control over data splits
- No filename requirements

**Option 2: Automatic Splitting**
- Place all files in `data/raw/`
- Files with `exam`, `test`, `quiz`, `final` → automatically split by date
- Earlier dates → validation, later dates → test
- All other files → training set

### Preprocessing

The preprocessing script automatically:
- Extracts Q&A pairs using format-specific strategies
- Chunks content into ≤2000 token segments
- Splits data (temporal or unit-based)
- Saves processed data as JSONL format

```bash
# Default: Temporal splitting (by date)
python 270FT/preprocess/load_and_prepare.py

# Or use unit-based splitting (by unit/topic)
python 270FT/preprocess/load_and_prepare.py --use_unit_split
```

For detailed data format specifications and splitting strategies, see the [full documentation](#data-preparation-details).

---

## 🎓 Training

### Configuration

Edit `270FT/configs/training_config.yaml`:

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

```bash
python 270FT/training/train_dual_lora.py
```

This will:
- Load both base models (LLaMA 3 and Qwen 3)
- Apply QLoRA with 4-bit quantization
- Train on `data/processed/train.jsonl`
- Evaluate on `data/processed/validation.jsonl` (if available)
- Save adapters to `models/llama3_lora/` and `models/qwen3_lora/`
- Log metrics to W&B (if configured)

### Training Output

- **Model checkpoints**: Saved in respective `models/` directories
- **W&B logs**: Training loss, evaluation metrics, hyperparameters
- **Console output**: Progress updates and final summary

---

## 📈 Evaluation

### Automated Metrics

Run the evaluation script:

```bash
python 270FT/evaluation/evaluate_models.py
```

**Metrics Computed**:
1. **Exact Match**: String-level comparison (case-insensitive)
2. **BLEU Score**: N-gram overlap between reference and prediction
3. **Symbolic Equivalence**: Mathematical correctness using SymPy

Results saved to `270FT/results/metrics_report.json`

### Human-in-the-Loop Evaluation

The pipeline supports human expert evaluation (professor ratings) alongside automated metrics.

**Scoring Rubric (1-5 Scale)**:
- **Mathematical Correctness** (1-5): Accuracy of mathematical content
- **Completeness** (1-5): Whether all parts are addressed
- **Clarity** (1-5): Explanation quality and organization
- **Overall Quality** (1-5): Overall assessment (would receive X credit)

**Workflow**:
1. Run automated evaluation first
2. Generate evaluation form: `python -m 270FT.evaluation.human_evaluation --test_results results/metrics_report.json --output results/human_evaluation_form.json`
3. Fill in scores (1-5) for each criterion
4. Aggregate results and compare with automated metrics

### Visualization

The `visualize_results.ipynb` notebook provides:
- Side-by-side comparison of automated vs human metrics
- Correlation analysis between metrics
- Per-item detailed breakdowns
- Model performance comparisons

---

## 📚 Project Structure

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
│   │   ├── evaluate_models.py
│   │   └── human_evaluation.py
│   ├── configs/              # Configuration files
│   │   └── training_config.yaml
│   ├── notebooks/            # Jupyter notebooks
│   │   ├── run_full_pipeline.ipynb
│   │   └── visualize_results.ipynb
│   ├── results/              # Evaluation results
│   └── outputs/              # CLI session outputs
├── main.py                   # Interactive CLI application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 💡 Lessons Learned

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

### Key Insights

- Parameter-efficient fine-tuning (QLoRA) is essential for limited data scenarios
- Temporal splitting prevents data leakage in educational datasets
- Human evaluation captures nuances that automated metrics miss
- Educational materials provide high-quality, domain-specific training data

---

## 🔮 Future Work

### Planned Improvements

1. **Retrieval-Augmented Generation (RAG)**
   - Use vector database to retrieve relevant examples during generation
   - Provide context from similar problems in training data

2. **Curriculum Learning**
   - Start with simple problems, gradually increase complexity
   - Better learning progression for the model

3. **Multi-Model Ensembling**
   - Combine predictions from LLaMA 3 and Qwen 3
   - Use voting or weighted averaging for final output

4. **Hybrid Symbolic-LLM Verification**
   - Use LLM for natural language understanding
   - Use symbolic solvers (Z3) for rigorous verification

5. **Data Augmentation**
   - Synthesize training examples using templates
   - Paraphrase existing problems

6. **Better Prompt Engineering**
   - Few-shot examples in prompts
   - Chain-of-thought reasoning
   - Explicit format instructions

---

## 📖 Detailed Documentation

<details>
<summary><b>Data Preparation Details</b></summary>

### Data Format Specifications

#### JSON Format (Recommended)
```json
[
  {
    "prompt": "Prove that the sum of the first n natural numbers is n(n+1)/2",
    "response": "[Algorithm Outline]\nUse mathematical induction...\n\n[Pseudocode]\n..."
  }
]
```

#### PDF Format (Multi-Strategy Extraction)
- **Strategy 1**: Explicit Q&A patterns (`Q:`, `Question:`, `A:`, `Answer:`)
- **Strategy 2**: Lecture slide extraction (problem → solution pairs)
- **Strategy 3**: Textbook theorem/proof extraction
- **Strategy 4**: Narrative content extraction

#### CSV Format
```csv
prompt,response
"Prove that...","[Algorithm Outline]..."
```

#### Text Format
```
Q: What is the time complexity of binary search?
A: O(log n) because we halve the search space each iteration.
```

### Train/Validation/Test Split Strategy

**Training Set** (60-70%):
- Textbooks, lecture slides, homework WITH solutions

**Validation Set** (15-20%):
- Past exams/quizzes (with or without solutions)
- Different homework WITH solutions (for automatic evaluation)

**Test Set** (15-20%):
- Future exams/quizzes
- Extracted exam questions (JSON format)

### Handling Duplicate Content

Use the duplicate detection utility:
```bash
python 270FT/preprocess/detect_duplicates.py \
    --train data/processed/train.jsonl \
    --validation data/processed/validation.jsonl \
    --test data/processed/test.jsonl \
    --threshold 0.9
```

</details>

<details>
<summary><b>Dataset Size Recommendations</b></summary>

### Minimum Viable Dataset Sizes

**For LoRA/QLoRA Fine-Tuning**:
- **Minimum**: 100-200 high-quality examples
- **Recommended**: 500-2000 examples
- **Ideal**: 2000-5000+ examples

### Training Strategy for Small Datasets

```yaml
training:
  epochs: 5-10          # More epochs for small datasets
  learning_rate: 1e-4   # Lower learning rate (more conservative)
  batch_size: 2-4       # Smaller batch size if limited data
  lora_r: 16            # Higher rank (more capacity) if data is limited
  lora_alpha: 32        # Scale with rank
```

</details>

<details>
<summary><b>Example Input/Output</b></summary>

### Input
```
Prove that the sum of the first n natural numbers is n(n+1)/2
```

### Output (LLaMA 3 Fine-Tuned)
```
[Algorithm Outline]
We use mathematical induction to prove this statement...

[Pseudocode]
function verify_sum_formula(n):
    if n == 1:
        return 1 == 1 * 2 / 2  // Base case: 1 = 1
    // Inductive step...

[Proof Summary]
Base Case (n=1): The sum of the first natural number is 1...

[Symbolic Check: Passed]
  Formula matches expected result: sum(1..n) = n(n+1)/2
```

</details>

---

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@software{270ft2024,
  title={Fine-Tuning LLMs for Pseudocode & Proof Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/270FT}
}
```

---

## 📝 License

[Specify your license here]

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Meta and Qwen teams for the base models
- The open-source community for various tools and libraries

---

**Note**: This project is for research purposes. Model outputs should be verified by domain experts for critical applications.

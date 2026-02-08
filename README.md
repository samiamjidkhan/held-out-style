# held-out-style

**A minimal implementation of character training with held-out evaluation.**

Can an 8B model learn an author's style from raw transcripts using synthetic preference data, and generalize to style dimensions it was never trained on?

This implements Constitutional AI + DPO for style transfer, based on [OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining).

## Motivation

[Will Brown's post on X](https://x.com/willccbb/status/2018060932723995087)

## Key Idea: Train/Eval Rubric Split

We split the evaluation rubric:
- **Train rubric** (2 dimensions): Used to score preference pairs during data generation
- **Eval rubric** (2 dimensions): Never seen during training, used only for final evaluation

If the model improves on both, it learned the style—not just the specific dimensions we trained on.

## Results: Norm Macdonald

**Setup:**
- Base model: Llama 3.1 8B Instruct
- Judge model: Llama 3.1 70B Instruct (AWQ INT4)
- 50 eval prompts scored on 4 dimensions (2 train + 2 held-out)

**Scores (1-5 scale):**

| Variant | Train Rubric | Eval Rubric |
|---------|-------------|-------------|
| base | 2.22 | 1.86 |
| dpo | 3.94 | 4.05 |
| merged (DPO+SFT) | 4.92 | 5.00 |

**Training data:**
- 10 style principles extracted from transcripts
- 841 prompts generated
- 511 preference pairs (filtered by score gap ≥1.2)
- DPO: 16 steps, LoRA rank 64
- SFT: 16 steps on DPO checkpoint
- Final adapter: 1.0×DPO + 0.25×SFT merge

**Trained adapter:** [samiamjidkhan/norm-macdonald-style](https://huggingface.co/samiamjidkhan/norm-macdonald-style)

## Pipeline

```
Corpus (transcripts)
       │
       ▼
Extract 10 style principles (70B)
       │
       ▼
Generate 841 prompts (70B)
       │
       ▼
For each prompt:
  • Chosen: 70B with style constitution + thinking prefill
  • Rejected: 70B without style conditioning
  • Score both on train rubric (70B judge)
  • Keep if score gap ≥ 1.2
       │
       ▼
511 preference pairs
       │
       ▼
DPO training (8B + LoRA)
       │
       ▼
SFT training (on DPO checkpoint)
       │
       ▼
Merge adapters → final model
       │
       ▼
Evaluate on train + eval rubrics
```

## Usage

```bash
pip install -e .

# Full pipeline
python run.py run-all norm_macdonald

# Or step by step:
python run.py extract-style norm_macdonald
python run.py generate-data norm_macdonald
python run.py train norm_macdonald
python run.py eval norm_macdonald
```

Requires A100 80GB for the 70B judge model.

## Adding a Style

1. Add transcripts to `corpus/your_style/*.txt`
2. Create `configs/styles/your_style.yaml`:

```yaml
style:
  name: "your_style"
  description: "One sentence description"
  corpus_dir: "corpus/your_style"

train_rubric:
  dimensions:
    - name: "dimension_1"
      description: "What this measures"
      anchors:
        1: "Score 1 means..."
        3: "Score 3 means..."
        5: "Score 5 means..."

eval_rubric:
  dimensions:
    - name: "held_out_dimension"
      description: "Related but different aspect"
      anchors:
        1: "..."
        3: "..."
        5: "..."

seed_questions:
  - "Example prompt for this style"

eval_prompts:
  - "Prompts for final evaluation"
```

3. Run `python run.py run-all your_style`

## Limitations

- **No true teacher/student split**: Both chosen and rejected come from the same 70B model (different prompting). A real 70B→8B capability gap would produce stronger training signal.
- **Single style tested**: Only validated on Norm Macdonald. Needs more styles to confirm methodology generalizes.
- **No robustness evaluation**: We measure quality, not resistance to adversarial prompts.

## Files

```
held-out-style/
├── configs/
│   ├── styles/norm_macdonald.yaml
│   ├── pipeline.yaml
│   ├── training.yaml
│   └── eval.yaml
├── corpus/norm_macdonald/*.txt
├── src/
│   ├── corpus.py
│   ├── eval.py
│   ├── extract.py
│   ├── generate.py
│   ├── inference.py
│   ├── judge.py
│   ├── rubric.py
│   └── train.py
├── data/
│   ├── principles/                   # Extracted style principles
│   ├── datasets/                     # Training data
│   ├── checkpoints/                  # Model weights (gitignored)
│   └── results/                      # Evaluation outputs
├── run.py
└── pyproject.toml
```

## Related Work

- [Character training: Understanding and crafting a language model's personality](https://interconnects.ai/p/character-training)
- [Opening the black box of character training](https://interconnects.ai/p/opening-the-black-box-of-character)
- [Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI](https://arxiv.org/abs/2511.01689)
- [RLHF Book: Chapter 17 - Character Training](https://rlhfbook.com/c/17-product)
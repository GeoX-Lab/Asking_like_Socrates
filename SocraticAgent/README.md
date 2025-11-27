

<!-- python SocraticAgent/example_preproc.py --data_path SocraticAgent/demo_data/VRSBench_train.json --image_root SocraticAgent/demo_data/images


python SocraticAgent/generation.py --data-path SocraticAgent/demo_data/VRSBench_train_convert.parquet --debug --debug-samples 2 --concurrency 2 --max-loop 6 --verify-inst "- If the answer is a pure number, it must exactly match the GT value.\n- For other cases, semantic equivalence is sufficient; if the model’s answer is more complete or detailed, it is still acceptable.\n" --image-meta-pre "The provided imagery is remote sensing image\n"

python SocraticAgent/generation.py --data-path SocraticAgent/demo_data/VRSBench_train_convert.parquet --concurrency 10 --max-loop 6 --verify-inst "- If the answer is a pure number, it must exactly match the GT value.\n- For other cases, semantic equivalence is sufficient; if the model’s answer is more complete or detailed, it is still acceptable.\n" --image-meta-pre "The provided imagery is remote sensing image\n"


python SocraticAgent/postproc.py --data_path SocraticAgent/demo_data/VRSBench_train_convert.jsonl -->


# SocraticAgent: Synthesizing Evidence-of-Thought Traces

**SocraticAgent** is a multi-agent data synthesis framework introduced in our paper *"Asking like Socrates: Socrates helps VLMs understand remote sensing images"*.

It simulates a "Socratic" dialogue between a **Reasoner** (text-only, logic-driven) and a **Perceiver** (visual expert) to solve Remote Sensing VQA tasks. This process generates high-quality, step-by-step reasoning traces (Evidence-of-Thought) which are then used to cold-start the **RS-EoT-7B** model.

## 📂 Directory Structure

```text
SocraticAgent
├── demo_data/               # Contains sample data and images
├── example_preproc.py       # Step 1: Convert standard VQA data to pipeline format
├── generation.py            # Step 2: Core multi-agent generation loop
└── postproc.py              # Step 3: Format traces
````

## 🚀 Usage Pipeline

The pipeline consists of three main steps: **Preprocessing**, **Trace Generation**, and **Post-processing**.

### 0\. Prerequisites

Before running the generation script, ensure you have configured your API keys. Create a `.env` file in the root directory or export the necessary environment variables for the models you intend to use (e.g., OPENAI_API_KEY, OPENAI_BASE_URL, ARK_API_KEY, etc.).

### 1\. Preprocessing

Convert VQA datasets into the intermediate format required by the generator. We provide a sample dataset in `demo_data`.

```bash
python SocraticAgent/example_preproc.py \
    --data_path SocraticAgent/demo_data/VRSBench_train.json \
    --image_root SocraticAgent/demo_data/images
```

*This will generate a `.parquet` file in the `demo_data` directory.*

### 2\. Trace Generation (SocraticAgent)

Launch the multi-agent system.

#### 🛠️ Debug Mode

Run with a small sample size (`--debug-samples 2`) and lower concurrency to verify your setup.

```bash
python SocraticAgent/generation.py \
    --data-path SocraticAgent/demo_data/VRSBench_train_convert.parquet \
    --debug \
    --debug-samples 2 \
    --concurrency 2 \
    --max-loop 6 \
    --verify-inst "- If the answer is a pure number, it must exactly match the GT value.\n- For other cases, semantic equivalence is sufficient; if the model’s answer is more complete or detailed, it is still acceptable.\n" \
    --image-meta-pre "The provided imagery is remote sensing image\n"
```

#### ⚡ Full Production Mode

Run on the full dataset with higher concurrency.

```bash
python SocraticAgent/generation.py \
    --data-path SocraticAgent/demo_data/VRSBench_train_convert.parquet \
    --concurrency 10 \
    --max-loop 6 \
    --verify-inst "- If the answer is a pure number, it must exactly match the GT value.\n- For other cases, semantic equivalence is sufficient; if the model’s answer is more complete or detailed, it is still acceptable.\n" \
    --image-meta-pre "The provided imagery is remote sensing image\n"
```

### 3\. Post-processing

Finally, convert the raw multi-agent dialogue logs into the final training format. This script concatenates the reasoning steps into a single `<think>...</think>` block and formats the final answer.

```bash
python SocraticAgent/postproc.py \
    --data_path SocraticAgent/demo_data/demo_output/VRSBench_train_convert.jsonl
```
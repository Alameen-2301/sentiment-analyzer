# Sentiment Analyzer

A small Python project for exploring binary sentiment classification with the Hugging Face Transformers pipeline.

The script runs a set of English example sentences through a pretrained model and prints a **POSITIVE** or **NEGATIVE** label with its model score. It is an inference demo; it does not train or fine-tune a model.

## How it works

1. Create a `pipeline("sentiment-analysis")`.
2. Run each sentence in the `sentences` list through the pipeline.
3. Print the predicted label and the score as a percentage.

The implementation uses the pipeline's default model rather than specifying a checkpoint. See the [Hugging Face pipeline documentation](https://huggingface.co/docs/transformers/en/pipeline_tutorial) for model selection.

## Quick start

Clone the repository, then install the dependencies in a virtual environment:

```bash
git clone https://github.com/Alameen-2301/sentiment-analyzer.git
cd sentiment-analyzer
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

```bash
python -m pip install torch transformers
python sentiment.py
```

The first run needs internet access to download the model. To try your own inputs, edit the `sentences` list in [sentiment.py](sentiment.py) and run the script again.

## What to inspect

The bundled examples include clearly positive and negative sentences, negation, and mixed sentiment. Compare the predicted label with your own interpretation, especially for inputs such as `"The weather today is okay"`.

## Limitations

- The classifier provides two labels; neutral or mixed statements are forced into one of them.
- Sarcasm, context, and negation can lead to misleading predictions.
- A high model score does not guarantee a correct prediction.
- The checkpoint and dependency versions are not pinned, so outputs may vary between environments.

## Built with

Python · Hugging Face Transformers · PyTorch

# GeneNet Research Beta

## Files

- `app.py` — Streamlit beta application
- `gene_model_v4_orfformer.pt` — frozen private checkpoint
- `requirements.txt` — Python dependencies

## Local run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints.

## Private-code deployment

Keep the GitHub repository PRIVATE.

The public app exposes the interface and prediction outputs. It does not need
to publish your GitHub repository or training/validation scripts.

For the first beta, keep these private:
- model weights
- training code
- validation code
- architecture/inference source
- benchmark construction scripts

## Beta workflow

FASTA upload
-> frozen GeneNet scan
-> contiguous coding-like regions
-> ranked table
-> downloadable:
   - genenet_candidates.fasta
   - genenet_candidates_context.fasta
   - genenet_results.csv
   - ZIP bundle including BED coordinates

## Scientific scope

GeneNet is presented as a coding-evidence detector.

It does NOT claim:
- exact gene start/stop boundaries
- confirmed genes
- biological function
- clinical/diagnostic use

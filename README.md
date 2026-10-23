# 🧬 GeneNet

**AI-assisted coding-region detection for bacterial genomes**

GeneNet is an open-source deep learning research tool that scans bacterial DNA and identifies **coding-like regions** based on learned sequence patterns.

Rather than replacing conventional gene callers, GeneNet is designed to provide an **independent neural coding-evidence signal** that can help researchers prioritize genomic regions for further investigation.

The project includes the trained model, inference pipeline, and an interactive Streamlit research application.

---

## 🔬 What GeneNet Does

GeneNet accepts bacterial genomic DNA in FASTA format and performs base-level neural inference across the sequence.

It produces:

- Ranked coding-like regions
- Genomic coordinates
- Region lengths
- Mean and maximum GeneNet scores
- Exact candidate DNA sequences
- Candidate sequences with surrounding genomic context
- CSV results
- BED genomic coordinates
- Downloadable FASTA files

Candidate sequences can then be investigated using tools such as:

- BLASTX
- BLASTN
- ORF/CDS prediction tools
- InterPro
- Protein/domain annotation pipelines

### Basic workflow

```text
Bacterial genome
       ↓
    GeneNet
       ↓
Learned coding evidence
       ↓
Candidate coding-like regions
       ↓
ORF / BLASTX / protein annotation
       ↓
Biological investigation
```

---

# 🧠 Model Architecture

GeneNet combines several neural sequence-modeling components.

```text
DNA sequence
     ↓
Nucleotide embedding
     ↓
Multi-scale convolutions
  ├── kernel 3
  ├── kernel 9
  └── kernel 27
     ↓
Feature merging
     ↓
Dilated residual blocks
  ├── dilation 1
  ├── dilation 2
  ├── dilation 4
  ├── dilation 8
  └── dilation 16
     ↓
2-layer Bidirectional GRU
     ↓
Neural prediction head
     ↓
Per-base coding probability
```

The multi-scale convolutional layers capture sequence patterns at different local scales, while the dilated residual network expands the sequence context available to the model.

A bidirectional GRU then integrates information across the sequence before producing base-level coding evidence.

---

# 🔎 Genome Scanning

GeneNet scans sequences using overlapping windows:

```text
Window size: 1024 bp
Stride:      256 bp
```

Predictions from overlapping windows are averaged to produce one coding probability for each nucleotide.

Contiguous positions exceeding the frozen model threshold are grouped into candidate coding-like regions.

The current primary beta reporting range uses regions of at least:

```text
180 bp
```

Shorter regions can still be explored experimentally.

---

# 📊 Research Validation

GeneNet has been evaluated on bacterial genomes not used during model development.

### External whole-genome coding-region recognition

Across three external bacterial genomes, GeneNet achieved approximately:

```text
Macro F1 ≈ 0.990
```

for per-base coding-region recognition.

Examples included:

```text
Listeria      F1 ≈ 0.994
Helicobacter  F1 ≈ 0.985
Campylobacter F1 ≈ 0.990
```

These values measure **per-base coding-region recognition**, not exact gene-boundary prediction.

---

## Challenging Reference CDS

A separate benchmark examined reference CDS that were missed by conventional Pyrodigal prediction.

Across 15 unseen bacterial genomes:

```text
Reference CDS missed by both
Pyrodigal single + meta modes: 659

Detected by GeneNet:           439

Rescue rate:                  66.62%
```

This suggests that GeneNet can provide coding evidence complementary to conventional gene-prediction approaches.

This does **not** establish that every rescued region represents an experimentally confirmed gene.

---

## Ordinary Reference CDS

For sampled ordinary reference CDS ≥180 bp:

```text
Detected: 401 / 408

Sensitivity: 98.28%
```

GeneNet therefore performs strongly on ordinary coding regions while retaining useful signal on more difficult reference CDS.

---

# 🧪 Example: GeneNet → BLASTX

A typical GeneNet candidate can be exported directly for downstream analysis.

For example:

```text
GeneNet candidate
Length: 632 bp
Mean score: ~0.94

        ↓ BLASTX

dTMP kinase
Query coverage: ~99%
Identity: ~99.5%
E-value: ~8e-152
```

In this case, almost the entire GeneNet region corresponds to a single recognizable protein-coding sequence.

BLASTX can also expose an important limitation of GeneNet.

For example:

```text
GeneNet candidate
████████████████████████████████████

BLASTX
████████████████████████   Protein A
                         ███████████ Protein B
```

This can indicate that GeneNet has detected a continuous coding territory containing more than one neighboring coding unit.

Therefore GeneNet should not be interpreted as an exact gene-boundary caller.

---

# ⚠️ Important Scientific Interpretation

GeneNet predicts:

> **coding-like sequence evidence**

It does **not** independently establish:

- an experimentally confirmed gene
- an exact CDS start position
- an exact CDS stop position
- protein function
- biological mechanism

A high GeneNet score means that the DNA sequence resembles patterns the model learned to associate with bacterial coding DNA.

For exact ORF/CDS boundaries and biological interpretation, GeneNet results should be combined with established downstream tools and biological evidence.

---

# 🚀 Streamlit Research Application

The included Streamlit application provides an interactive interface for running GeneNet.

### Input

```text
FASTA (.fasta / .fa / .fna)
```

### Output

GeneNet generates:

```text
genenet_results.csv

genenet_candidates.fasta

genenet_candidates_context.fasta

genenet_regions.bed
```

A ZIP archive containing the complete result set can also be downloaded.

---

# 📁 Repository Structure

```text
GeneNet/
│
├── app.py
│
├── gene_model_v4_orfformer.pt
│
├── requirements.txt
├── LICENSE
└── README.md
```

### `app.py`

Contains the GeneNet architecture, frozen-model loading, FASTA parsing, genome scanning, candidate extraction and Streamlit interface.

### `gene_model_v4_orfformer.pt`

Frozen trained GeneNet checkpoint used by the research beta.

### `requirements.txt`

Python dependencies required to run the application.

---

# 💻 Running Locally

Clone the repository:

```bash
git clone https://github.com/N1S55M-X/genenet-beta.git
cd genenet-beta
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run GeneNet:

```bash
streamlit run app.py
```

Then open the local Streamlit address displayed in your terminal.

---

# 🧬 Example Research Workflow

```text
1. Obtain bacterial FASTA
          ↓
2. Scan with GeneNet
          ↓
3. Rank coding-like candidates
          ↓
4. Export candidate/context FASTA
          ↓
5. Examine ORFs
          ↓
6. Run BLASTX / homology searches
          ↓
7. Investigate domains and proteins
          ↓
8. Develop a biological hypothesis
```

GeneNet is intended to contribute an additional learned signal to this workflow rather than replace established genomic annotation methods.

---

# 🎯 Research Motivation

Conventional microbial gene prediction relies heavily on engineered biological and statistical assumptions about coding sequences.

GeneNet explores a complementary question:

> **Can a neural sequence model learn coding evidence directly from bacterial DNA and identify useful coding-like regions independently of a conventional gene caller?**

The project is an ongoing research experiment into learned representations of microbial coding structure.

---

# 🛠️ Current Limitations

GeneNet Research Beta currently has several important limitations:

- It is designed for bacterial/microbial DNA.
- It is not an exact gene-boundary caller.
- Adjacent coding regions can sometimes be merged.
- False-positive rates vary across genomic contexts and species.
- Very short coding regions remain more challenging.
- Functional annotation requires external tools.
- Large genomes can require substantial CPU/GPU computation.

These limitations are active areas for future investigation.

---

# 🔭 Future Research

Potential research directions include:

- Better separation of adjacent coding regions
- Improved boundary refinement
- Better short-CDS detection
- Species-aware calibration
- Expanded independent genome validation
- Integration with ORF evidence
- Protein/domain annotation workflows
- Uncertainty calibration
- Investigation of difficult and poorly characterized coding regions

---

# 🤝 Contributions

GeneNet is an open research project.

Testing on additional bacterial genomes, benchmarking, bug reports, reproducibility checks and scientific feedback are welcome.

If you test GeneNet on a new organism, opening a GitHub Issue with your observations is especially useful.

---

# 📜 License

GeneNet is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the `LICENSE` file for the complete license terms.

---

# ⚠️ Research Disclaimer

GeneNet is a research beta and should not be treated as a clinical, diagnostic or experimentally validated gene-annotation system.

Predicted regions represent learned coding-like sequence evidence and should be independently evaluated using appropriate genomic, protein and experimental evidence.

---

# 👨‍🔬 Author

**Mohammed Naseem Hussain**

GitHub: **N1S55M-X**

GeneNet was developed as an independent research project exploring deep learning approaches to microbial coding-region detection.

---

## ⭐ GeneNet

**Raw DNA → learned coding evidence → candidate regions → biological investigation**

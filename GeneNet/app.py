import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# GeneNet Research Beta
# ============================================================
# Input:
#   bacterial FASTA (.fasta/.fa/.fna)
#
# Output:
#   1) ranked coding-like regions
#   2) exact detected-region FASTA
#   3) detected-region + flank FASTA
#   4) CSV coordinates/scores
#
# IMPORTANT:
# GeneNet is a coding-evidence model, not an exact gene-boundary
# caller and not a functional annotation system.
# ============================================================

APP_TITLE = "GeneNet Research Beta"

MODEL_PATH = Path("gene_model_v4_orfformer.pt")

WINDOW = 1024
STRIDE = 256
BATCH_SIZE = 32

# Frozen model threshold comes FROM the checkpoint.
MIN_PRIMARY_BP = 180

# Context exported around every coding-like region.
DEFAULT_FLANK_BP = 1000

# Safety / usability limits for a public beta.
MAX_UPLOAD_MB = 20
MAX_DISPLAY_ROWS = 500

# ------------------------------------------------------------
# PAGE
# ------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GeneNet Research Beta")
st.caption("AI-assisted bacterial coding-region detection")

st.info(
    "Upload bacterial DNA in FASTA format. GeneNet scans the sequence and "
    "returns coding-like DNA regions that can be downloaded and reused in "
    "BLASTX, ORF-finding, annotation, or other downstream tools."
)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.15):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(x))
        x = self.conv1(x)
        x = self.dropout(x)
        x = F.gelu(self.norm2(x))
        x = self.conv2(x)
        return residual + x


class GeneNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(5, 32)

        self.conv3 = nn.Conv1d(32, 48, 3, padding=1)
        self.conv9 = nn.Conv1d(32, 48, 9, padding=4)
        self.conv27 = nn.Conv1d(32, 48, 27, padding=13)

        self.merge = nn.Conv1d(144, 128, 1)

        self.residual = nn.Sequential(
            ResidualDilatedBlock(128, 1),
            ResidualDilatedBlock(128, 2),
            ResidualDilatedBlock(128, 4),
            ResidualDilatedBlock(128, 8),
            ResidualDilatedBlock(128, 16),
        )

        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20,
        )

        self.head = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        a = F.gelu(self.conv3(x))
        b = F.gelu(self.conv9(x))
        c = F.gelu(self.conv27(x))

        x = torch.cat([a, b, c], dim=1)
        x = self.merge(x)
        x = self.residual(x)

        x = x.transpose(1, 2)
        x, _ = self.bigru(x)

        return self.head(x).squeeze(-1)


@st.cache_resource(show_spinner="Loading frozen GeneNet model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. "
            "Place it in the same folder as app.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    required = {"gene_model_state_dict", "gene_threshold", "base_map"}
    missing = required - set(checkpoint.keys())
    if missing:
        raise KeyError(
            f"Checkpoint is missing required fields: {sorted(missing)}"
        )

    model = GeneNetV2().to(device)
    model.load_state_dict(
        checkpoint["gene_model_state_dict"],
        strict=True,
    )
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    threshold = float(checkpoint["gene_threshold"])
    base_map = dict(checkpoint["base_map"])

    # Sanity check
    with torch.inference_mode():
        dummy = torch.randint(
            0, 5, (1, WINDOW),
            dtype=torch.long,
            device=device,
        )
        out = model(dummy)

    if tuple(out.shape) != (1, WINDOW):
        raise RuntimeError("Model sanity check failed.")

    return model, threshold, base_map, device


# ------------------------------------------------------------
# FASTA
# ------------------------------------------------------------

def parse_fasta(text):
    """
    Minimal FASTA parser.
    Returns:
        list of (record_name, DNA_sequence)
    """
    records = []
    name = None
    chunks = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if name is not None:
                seq = "".join(chunks).upper()
                records.append((name, seq))

            name = line[1:].strip() or f"contig_{len(records)+1}"
            chunks = []

        else:
            if name is None:
                raise ValueError(
                    "Invalid FASTA: sequence found before a >header."
                )

            seq_line = "".join(line.split()).upper()
            chunks.append(seq_line)

    if name is not None:
        seq = "".join(chunks).upper()
        records.append((name, seq))

    if not records:
        raise ValueError("No FASTA records found.")

    clean_records = []

    for name, seq in records:
        if not seq:
            continue

        # Normalize U as T. Everything else outside A/C/G/T/N
        # is treated as N rather than crashing inference.
        seq = seq.replace("U", "T")
        seq = "".join(
            base if base in {"A", "C", "G", "T", "N"} else "N"
            for base in seq
        )

        clean_records.append((name, seq))

    if not clean_records:
        raise ValueError("No non-empty DNA sequences found.")

    return clean_records


def safe_name(text):
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            cleaned.append(ch)
        else:
            cleaned.append("_")

    result = "".join(cleaned).strip("_")
    return result[:80] or "sequence"


# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

def encode_sequence(sequence, base_map):
    unknown_id = int(base_map.get("N", 4))

    return np.fromiter(
        (base_map.get(base, unknown_id) for base in sequence),
        dtype=np.int64,
        count=len(sequence),
    )


@torch.inference_mode()
def scan_sequence(
    sequence,
    model,
    threshold,
    base_map,
    device,
    progress_callback=None,
):
    """
    Frozen GeneNet scan using the same:
        WINDOW = 1024
        STRIDE = 256
    and overlap averaging used in validation.

    Returns one probability per input base.
    """

    sequence = sequence.upper()
    n = len(sequence)

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    probability_sum = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    if n <= WINDOW:
        starts = [0]
    else:
        starts = list(range(0, n - WINDOW + 1, STRIDE))
        last = n - WINDOW

        if not starts or starts[-1] != last:
            starts.append(last)

    total_batches = max(
        1,
        (len(starts) + BATCH_SIZE - 1) // BATCH_SIZE
    )

    for batch_number, batch_begin in enumerate(
        range(0, len(starts), BATCH_SIZE),
        start=1,
    ):
        batch_starts = starts[
            batch_begin:batch_begin + BATCH_SIZE
        ]

        encoded = []
        usable_lengths = []

        for start in batch_starts:
            frag = sequence[start:start + WINDOW]
            usable = len(frag)
            usable_lengths.append(usable)

            if usable < WINDOW:
                frag += "N" * (WINDOW - usable)

            encoded.append(
                encode_sequence(frag, base_map)
            )

        x = torch.as_tensor(
            np.stack(encoded),
            dtype=torch.long,
            device=device,
        )

        probs = torch.sigmoid(model(x)).cpu().numpy()

        for row, start in enumerate(batch_starts):
            usable = usable_lengths[row]

            probability_sum[
                start:start + usable
            ] += probs[row, :usable]

            counts[
                start:start + usable
            ] += 1.0

        if progress_callback is not None:
            progress_callback(
                batch_number / total_batches
            )

    counts[counts == 0] = 1.0

    return probability_sum / counts


# ------------------------------------------------------------
# REGION EXTRACTION
# ------------------------------------------------------------

def contiguous_positive_regions(
    probabilities,
    threshold,
):
    """
    Convert the per-base GeneNet signal into raw contiguous
    threshold-positive regions.

    No ORF prediction.
    No boundary correction.
    No hidden merging rules.
    """

    mask = probabilities >= threshold
    regions = []

    start = None

    for i, positive in enumerate(mask):
        if positive and start is None:
            start = i

        elif not positive and start is not None:
            regions.append((start, i))
            start = None

    if start is not None:
        regions.append((start, len(mask)))

    return regions


def region_stats(probabilities, start, end, threshold):
    p = probabilities[start:end]

    if len(p) == 0:
        return 0.0, 0.0, 0.0

    return (
        float(np.mean(p)),
        float(np.max(p)),
        float(np.mean(p >= threshold)),
    )


def evidence_label(mean_score):
    """
    Display-only wording.
    Does NOT create a new biological validation threshold.
    """
    if mean_score >= 0.80:
        return "Strong"
    elif mean_score >= 0.50:
        return "Moderate"
    return "Coding-like"


def build_candidates(
    contig_name,
    sequence,
    probabilities,
    threshold,
    min_bp,
    flank_bp,
    id_counter,
):
    raw_regions = contiguous_positive_regions(
        probabilities,
        threshold,
    )

    rows = []
    candidate_payloads = []

    for start, end in raw_regions:
        length = end - start

        if length < min_bp:
            continue

        mean_score, max_score, positive_fraction = region_stats(
            probabilities,
            start,
            end,
            threshold,
        )

        candidate_id = f"GN_{id_counter:06d}"
        id_counter += 1

        context_start = max(0, start - flank_bp)
        context_end = min(
            len(sequence),
            end + flank_bp,
        )

        core_seq = sequence[start:end]
        context_seq = sequence[
            context_start:context_end
        ]

        rows.append(
            {
                "candidate_id": candidate_id,
                "contig": contig_name,

                # Human-readable coordinates
                "start_1based": start + 1,
                "end_1based": end,

                # Machine-friendly coordinates
                "start_0based": start,
                "end_0based_exclusive": end,

                "length_bp": length,
                "mean_genenet_score": round(
                    mean_score, 6
                ),
                "max_genenet_score": round(
                    max_score, 6
                ),
                "positive_fraction": round(
                    positive_fraction, 6
                ),
                "evidence": evidence_label(
                    mean_score
                ),
                "model_threshold": threshold,
                "primary_range": (
                    "yes"
                    if length >= MIN_PRIMARY_BP
                    else "experimental"
                ),
                "context_start_1based":
                    context_start + 1,
                "context_end_1based":
                    context_end,
                "context_length_bp":
                    context_end - context_start,
            }
        )

        candidate_payloads.append(
            {
                "candidate_id": candidate_id,
                "contig": contig_name,
                "start": start,
                "end": end,
                "context_start": context_start,
                "context_end": context_end,
                "core_seq": core_seq,
                "context_seq": context_seq,
                "mean_score": mean_score,
                "max_score": max_score,
            }
        )

    return rows, candidate_payloads, id_counter


# ------------------------------------------------------------
# DOWNLOAD FILES
# ------------------------------------------------------------

def wrap_fasta(seq, width=80):
    return "\n".join(
        seq[i:i + width]
        for i in range(0, len(seq), width)
    )


def build_core_fasta(payloads):
    parts = []

    for c in payloads:
        header = (
            f">{c['candidate_id']}"
            f"|contig={safe_name(c['contig'])}"
            f"|start={c['start']+1}"
            f"|end={c['end']}"
            f"|length={c['end']-c['start']}"
            f"|mean_score={c['mean_score']:.4f}"
        )

        parts.append(
            header + "\n" +
            wrap_fasta(c["core_seq"])
        )

    return "\n".join(parts) + (
        "\n" if parts else ""
    )


def build_context_fasta(payloads, flank_bp):
    parts = []

    for c in payloads:
        header = (
            f">{c['candidate_id']}_CONTEXT"
            f"|contig={safe_name(c['contig'])}"
            f"|context_start={c['context_start']+1}"
            f"|context_end={c['context_end']}"
            f"|core_start={c['start']+1}"
            f"|core_end={c['end']}"
            f"|requested_flank={flank_bp}"
        )

        parts.append(
            header + "\n" +
            wrap_fasta(c["context_seq"])
        )

    return "\n".join(parts) + (
        "\n" if parts else ""
    )


def build_bed(rows):
    """
    BED uses:
      0-based start
      half-open end
    """
    lines = []

    for r in rows:
        lines.append(
            "\t".join(
                [
                    safe_name(r["contig"]),
                    str(r["start_0based"]),
                    str(
                        r["end_0based_exclusive"]
                    ),
                    r["candidate_id"],
                    str(
                        int(
                            min(
                                1000,
                                round(
                                    r[
                                        "mean_genenet_score"
                                    ] * 1000
                                ),
                            )
                        )
                    ),
                    ".",
                ]
            )
        )

    return "\n".join(lines) + (
        "\n" if lines else ""
    )


def make_zip(csv_bytes, core_fasta, context_fasta, bed_text):
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "genenet_results.csv",
            csv_bytes,
        )
        zf.writestr(
            "genenet_candidates.fasta",
            core_fasta,
        )
        zf.writestr(
            "genenet_candidates_context.fasta",
            context_fasta,
        )
        zf.writestr(
            "genenet_regions.bed",
            bed_text,
        )

    buffer.seek(0)
    return buffer.getvalue()


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------

try:
    model, gene_threshold, base_map, device = load_model()

except Exception as exc:
    st.error(
        "GeneNet could not be loaded."
    )
    st.code(str(exc))
    st.stop()


with st.sidebar:
    st.header("Run settings")

    st.write(
        f"**Frozen GeneNet threshold:** "
        f"`{gene_threshold:.4f}`"
    )

    st.write(
        f"**Primary reporting minimum:** "
        f"`{MIN_PRIMARY_BP} bp`"
    )

    min_bp = st.number_input(
        "Minimum returned region length (bp)",
        min_value=1,
        max_value=10000,
        value=MIN_PRIMARY_BP,
        step=10,
        help=(
            "180 bp is the current primary beta "
            "reporting minimum. Lower values are "
            "experimental."
        ),
    )

    flank_bp = st.number_input(
        "Context DNA on each side (bp)",
        min_value=0,
        max_value=10000,
        value=DEFAULT_FLANK_BP,
        step=100,
        help=(
            "Extra DNA exported around each detected "
            "region so downstream tools can inspect "
            "possible full ORFs/boundaries."
        ),
    )

    st.divider()

    st.write(f"**Inference device:** `{device}`")
    st.write(f"**Window / stride:** `{WINDOW} / {STRIDE}`")

    st.warning(
        "GeneNet reports coding-like DNA evidence. "
        "It does not by itself confirm a complete "
        "gene, exact CDS boundary, or biological function."
    )



# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "scan_payloads" not in st.session_state:
    st.session_state.scan_payloads = None
if "scan_rows" not in st.session_state:
    st.session_state.scan_rows = None
if "scan_flank_bp" not in st.session_state:
    st.session_state.scan_flank_bp = None
if "scan_filename" not in st.session_state:
    st.session_state.scan_filename = None
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


def restart_app():
    """Clear only user/run data. The cached model remains loaded."""
    st.session_state.scan_results = None
    st.session_state.scan_payloads = None
    st.session_state.scan_rows = None
    st.session_state.scan_flank_bp = None
    st.session_state.scan_filename = None
    st.session_state.uploader_version += 1


with st.sidebar:
    st.divider()
    if st.button(
        "↻ Start new genome",
        use_container_width=True,
        help="Clear the current genome and results and start a new scan.",
    ):
        restart_app()
        st.rerun()


uploaded = st.file_uploader(
    "Upload bacterial FASTA",
    type=["fasta", "fa", "fna", "fas", "txt"],
    key=f"genome_upload_{st.session_state.uploader_version}",
)

# ------------------------------------------------------------
# UPLOAD PREVIEW + SCAN
# ------------------------------------------------------------

records = None

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_mb = len(file_bytes) / (1024 * 1024)

    if file_mb > MAX_UPLOAD_MB:
        st.error(
            f"File is {file_mb:.1f} MB. "
            f"Current beta limit is {MAX_UPLOAD_MB} MB."
        )
        st.stop()

    try:
        fasta_text = file_bytes.decode(
            "utf-8",
            errors="replace",
        )
        records = parse_fasta(fasta_text)

    except Exception as exc:
        st.error("Could not parse the FASTA file.")
        st.code(str(exc))
        st.stop()

    total_bp = sum(len(seq) for _, seq in records)

    col1, col2, col3 = st.columns(3)

    col1.metric("FASTA records", len(records))
    col2.metric("Total DNA", f"{total_bp:,} bp")
    col3.metric("Minimum output region", f"{min_bp:,} bp")

    run_scan = st.button(
        "🧬 Scan with GeneNet",
        type="primary",
        use_container_width=True,
    )

    if run_scan:
        all_rows = []
        all_payloads = []
        id_counter = 1

        overall_progress = st.progress(0)
        status = st.empty()

        for record_index, (name, seq) in enumerate(
            records,
            start=1,
        ):
            status.write(
                f"Scanning `{name}` "
                f"({len(seq):,} bp) — "
                f"record {record_index}/{len(records)}"
            )

            def update_local_progress(local_fraction):
                completed_before = (
                    record_index - 1
                ) / len(records)

                record_share = 1 / len(records)

                overall = (
                    completed_before
                    + local_fraction * record_share
                )

                overall_progress.progress(
                    min(1.0, overall)
                )

            probabilities = scan_sequence(
                sequence=seq,
                model=model,
                threshold=gene_threshold,
                base_map=base_map,
                device=device,
                progress_callback=update_local_progress,
            )

            rows, payloads, id_counter = build_candidates(
                contig_name=name,
                sequence=seq,
                probabilities=probabilities,
                threshold=gene_threshold,
                min_bp=int(min_bp),
                flank_bp=int(flank_bp),
                id_counter=id_counter,
            )

            all_rows.extend(rows)
            all_payloads.extend(payloads)

        overall_progress.progress(1.0)

        if not all_rows:
            status.warning(
                "Finished, but no regions passed the selected "
                "minimum length."
            )
            st.session_state.scan_results = None
            st.session_state.scan_payloads = None
            st.session_state.scan_rows = None
        else:
            results = pd.DataFrame(all_rows)

            results = results.sort_values(
                by=[
                    "mean_genenet_score",
                    "length_bp",
                ],
                ascending=[False, False],
            ).reset_index(drop=True)

            # THIS is the important fix:
            # keep the complete scan in Streamlit session state.
            st.session_state.scan_results = results
            st.session_state.scan_payloads = all_payloads
            st.session_state.scan_rows = all_rows
            st.session_state.scan_flank_bp = int(flank_bp)
            st.session_state.scan_filename = uploaded.name

            status.success(
                f"Finished. Found {len(all_rows):,} "
                "returned coding-like regions. "
                "Results are now kept for this session."
            )

elif st.session_state.scan_results is None:
    st.subheader("Output")
    st.write(
        "After scanning, you will receive a ranked table "
        "plus downloadable FASTA/CSV files."
    )
    st.code(
        """Candidate     Start      End      Length    Score
GN_000001     1284501    1284930   430 bp    0.97
GN_000002     2010444    2010831   388 bp    0.93"""
    )


# ------------------------------------------------------------
# PERSISTENT RESULTS
# ------------------------------------------------------------

if st.session_state.scan_results is not None:
    results = st.session_state.scan_results
    all_payloads = st.session_state.scan_payloads
    all_rows = st.session_state.scan_rows
    result_flank_bp = st.session_state.scan_flank_bp

    st.divider()
    st.subheader("Results")

    if st.session_state.scan_filename:
        st.caption(
            f"Current scan: {st.session_state.scan_filename} · "
            "Results stay available while you download or inspect candidates."
        )

    a, b, c = st.columns(3)

    a.metric(
        "Returned regions",
        f"{len(results):,}",
    )

    b.metric(
        "Mean region length",
        f"{results['length_bp'].mean():,.0f} bp",
    )

    c.metric(
        "Mean GeneNet score",
        f"{results['mean_genenet_score'].mean():.3f}",
    )

    display_columns = [
        "candidate_id",
        "contig",
        "start_1based",
        "end_1based",
        "length_bp",
        "mean_genenet_score",
        "max_genenet_score",
        "evidence",
        "primary_range",
    ]

    st.dataframe(
        results[
            display_columns
        ].head(MAX_DISPLAY_ROWS),
        use_container_width=True,
        hide_index=True,
    )

    if len(results) > MAX_DISPLAY_ROWS:
        st.caption(
            f"Showing first {MAX_DISPLAY_ROWS:,} ranked "
            "regions. Downloads contain all results."
        )

    # --------------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------------

    csv_text = results.to_csv(index=False)
    csv_bytes = csv_text.encode("utf-8")

    core_fasta = build_core_fasta(all_payloads)

    context_fasta = build_context_fasta(
        all_payloads,
        int(result_flank_bp),
    )

    bed_text = build_bed(all_rows)

    zip_bytes = make_zip(
        csv_bytes,
        core_fasta,
        context_fasta,
        bed_text,
    )

    st.subheader("Download for downstream analysis")

    st.write(
        "**Downloads no longer clear the scan.** "
        "Your current results remain on screen until you press "
        "**Start new genome**."
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇ Download exact candidates FASTA",
            data=core_fasta,
            file_name="genenet_candidates.fasta",
            mime="text/plain",
            use_container_width=True,
            on_click="ignore",
        )

        st.download_button(
            "⬇ Download results CSV",
            data=csv_bytes,
            file_name="genenet_results.csv",
            mime="text/csv",
            use_container_width=True,
            on_click="ignore",
        )

    with d2:
        st.download_button(
            "⬇ Download candidates + context FASTA",
            data=context_fasta,
            file_name="genenet_candidates_context.fasta",
            mime="text/plain",
            use_container_width=True,
            on_click="ignore",
        )

        st.download_button(
            "⬇ Download everything (.zip)",
            data=zip_bytes,
            file_name="genenet_results_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            on_click="ignore",
        )

    with st.expander("What can I do with these files?"):
        st.markdown(
            """
**`genenet_candidates.fasta`**  
Exact coding-like DNA regions returned by GeneNet. These can be used in tools such as BLASTN or BLASTX.

**`genenet_candidates_context.fasta`**  
Each candidate plus surrounding genomic DNA. This is preferable when you want another tool to search for complete ORFs, start/stop boundaries, or translated protein candidates.

**`genenet_results.csv`**  
Coordinates, lengths, GeneNet scores, and metadata for every returned region.

**`genenet_regions.bed`**  
Included in the ZIP bundle for loading the regions into genome browsers and other coordinate-based tools.
"""
        )

    # --------------------------------------------------------
    # INDIVIDUAL CANDIDATE VIEW
    # --------------------------------------------------------

    st.subheader("Inspect one candidate")

    selected_id = st.selectbox(
        "Candidate",
        results["candidate_id"].tolist(),
        key="candidate_selector",
    )

    selected_row = results[
        results["candidate_id"] == selected_id
    ].iloc[0]

    selected_payload = next(
        x for x in all_payloads
        if x["candidate_id"] == selected_id
    )

    x1, x2, x3, x4 = st.columns(4)

    x1.metric(
        "Length",
        f"{selected_row['length_bp']} bp",
    )
    x2.metric(
        "Mean score",
        f"{selected_row['mean_genenet_score']:.3f}",
    )
    x3.metric(
        "Max score",
        f"{selected_row['max_genenet_score']:.3f}",
    )
    x4.metric(
        "Evidence",
        selected_row["evidence"],
    )

    st.write(
        f"**Coordinates:** "
        f"{selected_row['contig']} : "
        f"{selected_row['start_1based']:,}–"
        f"{selected_row['end_1based']:,}"
    )

    st.text_area(
        "Exact GeneNet-positive DNA",
        value=selected_payload["core_seq"],
        height=180,
        key=f"core_{selected_id}",
    )

    st.text_area(
        "Candidate + surrounding context DNA",
        value=selected_payload["context_seq"],
        height=180,
        key=f"context_{selected_id}",
    )

    candidate_single_fasta = (
        f">{selected_id}"
        f"|contig={safe_name(selected_row['contig'])}"
        f"|start={selected_row['start_1based']}"
        f"|end={selected_row['end_1based']}\n"
        f"{wrap_fasta(selected_payload['core_seq'])}\n"
    )

    st.download_button(
        f"⬇ Download {selected_id}.fasta",
        data=candidate_single_fasta,
        file_name=f"{selected_id}.fasta",
        mime="text/plain",
        on_click="ignore",
    )

    st.divider()

    if st.button(
        "↻ Start new genome",
        key="restart_bottom",
        use_container_width=True,
    ):
        restart_app()
        st.rerun()

    st.caption(
        "Research beta. GeneNet reports learned coding-like "
        "sequence evidence. Returned regions are not by "
        "themselves experimentally confirmed genes, exact "
        "CDS boundaries, or functional annotations."
    )


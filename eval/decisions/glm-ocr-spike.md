# GLM-OCR feasibility spike — Apple Silicon (roadmap Phase 1.3)

_Run 2026-08-08 on the machine below. **Investigation only — no pipeline code was
changed.** Every claim here is either a command run on this host (with its output)
or a cited URL. Anything I could not verify is labelled **unverified**._

**Recommendation: GO-LATER** — the serving path works here today with zero new
Python dependencies, but three defects (below) block making it a default. See
[Recommendation](#recommendation).

| | |
|---|---|
| Host | Apple M2 Max, 96 GiB unified memory, macOS 15.5 (24F74) |
| Ollama | 0.32.6 (`ollama --version`) |
| Python env | `.venv` — docling 2.118.1, docling-core 2.91.0, docling-parse 7.11.0, torch 2.4.1, transformers 4.51.0, pymupdf 1.28.2 |
| Model pulled | `glm-ocr:latest`, id `6effedd0dc8a`, 2.2 GB — **kept** (it works) |

---

## Q1 — Serving options on Apple Silicon, Aug 2026

### (a) Ollama — WORKS, this is the viable path

`glm-ocr` is in the **official Ollama library namespace** (no user prefix):
<https://ollama.com/library/glm-ocr>. Tags on that page: `latest` 2.2 GB,
`q8_0` 1.6 GB, `bf16` 2.2 GB — all 128K context, text+image input. The upstream
repo links its own Ollama guide
(<https://github.com/zai-org/GLM-OCR/blob/main/examples/ollama-deploy/README.md>),
so the tag is vendor-blessed, not a random community upload.

```
$ ollama pull glm-ocr          # ~17 min here at ~2.0 MB/s; "success"
$ ollama list | grep ocr
glm-ocr:latest    6effedd0dc8a    2.2 GB
```

```
$ ollama show glm-ocr
  architecture        glmocr
  parameters          1.1B
  context length      131072
  embedding length    1536
  quantization        F16
  requires            0.15.5
  Capabilities: completion, vision, tools
```

Loaded footprint, from `ollama ps` during a run: **2.8 GB, 100% GPU** at
`num_ctx=16384`. Disk: 2.1 GB.

**Caveat found by experiment — the Ollama build ignores the prompt.**

```
$ ollama show glm-ocr --template
{{ .Prompt }}
$ ollama show glm-ocr --parameters
temperature   0
```

The Modelfile template is bare and defines no stop strings. I sent three
different prompts (`Text Recognition:`, a hand-wrapped
`<|user|>\nText Recognition:<|assistant|>`, and `Convert this page to markdown.`)
against the same page: **byte-identical output all three times**. Consequence:
GLM-OCR's documented alternate modes (`Formula Recognition:`,
`Table Recognition:`, and the JSON information-extraction schemas) are **not
reachable through Ollama**. Only the default full-page recognition behaviour is.

Upstream's own Ollama README says to prefer the native `/api/generate` endpoint
over the OpenAI-compatible one for vision, and recommends vLLM/SGLang for
production. Docling uses `/v1/chat/completions` — it worked here regardless
(Q2), but that is a documented mismatch to keep an eye on.

### (b) llama.cpp / GGUF — supported upstream, not tested here

Support landed via PR #19677; usage thread:
<https://github.com/ggml-org/llama.cpp/discussions/19721>. Needs the decoder
GGUF **plus** an `mmproj` projector from `ggml-org/GLM-OCR-GGUF`, and
**flash-attention must be off**:

```
llama-server -m glmocr-Q4_K_M.gguf --mmproj mmproj-glmocr-Q4_1.gguf \
  -c 12000 -ngl 99 --flash-attn off -fit off
```

Reported working on M-series Macs in that thread (one user: ~3 min for a 7-page
document on an M1 Air at Q8). **Not exercised on this host** — Ollama (which is
llama.cpp underneath) already gave us a working server, so a second one buys
nothing except control over the chat template, which is the one thing that would
fix the prompt-ignored caveat above. Worth revisiting only if we need the
`Table Recognition:` / extraction modes.

### (c) MLX — a port exists; needs a dependency we did not install

- `mlx-community/GLM-OCR-bf16`, 2.21 GB, BF16, MIT
  (<https://huggingface.co/mlx-community/GLM-OCR-bf16>). Card states it was
  converted with **mlx-vlm 0.3.11**.
- Our pinned docling already knows about it — `vlm_model_specs.py:436`
  (`GLMOCR_MLX`, `repo_id="mlx-community/GLM-OCR-bf16"`, MPS-only) and
  `stage_model_specs.py:1367-1368` ("Native GLM-OCR support was added to
  mlx-vlm in v0.3.11").
- Neither `mlx` nor `mlx-vlm` is in `.venv`, and installing packages was out of
  scope for this spike, so **this path is unverified on this host**. It is the
  most promising *future* path: in-process, no server, full prompt control,
  and docling drives it through the same preset with
  `engine_options=MlxVlmEngineOptions(...)`.

### (d) vLLM on macOS — NO

<https://docs.vllm.ai/en/stable/getting_started/installation/cpu/> and
`docs/getting_started/installation/cpu/apple.inc.md` in the vLLM repo: macOS is
**source-build, CPU-only, FP32/FP16, no prebuilt Apple Silicon wheels**. A 2026
write-up measures the CPU backend at 20–30× slower than llama.cpp's Metal
backend. A community Metal plugin exists
(<https://github.com/vllm-project/vllm-metal>, MLX compute backend) but that is
a third serving stack to own. Confirmed dead end for us; not pursued further.

---

## Q2 — Docling integration

### Our pinned version already ships it. No upgrade needed.

```
$ .venv/bin/pip show docling
Name: docling
Version: 2.118.1
```

GLM-OCR is present in that installed tree:

| File (in `.venv/lib/python3.12/site-packages/docling/`) | What |
|---|---|
| `datamodel/vlm_model_specs.py:416` | `GLMOCR_TRANSFORMERS` (`zai-org/GLM-OCR`, MPS listed as a supported device) |
| `datamodel/vlm_model_specs.py:436` | `GLMOCR_MLX` (`mlx-community/GLM-OCR-bf16`) |
| `datamodel/vlm_model_specs.py:446/449` | `GLMOCR_VLLM`, `GLMOCR_VLLM_API` |
| `datamodel/stage_model_specs.py:1354` | preset `glm_ocr` with `api_overrides` for `API`, `API_OPENAI`, **`API_OLLAMA`**, `API_LMSTUDIO` |
| `models/inference_engines/vlm/base.py:29` | `VlmEngineType` incl. `API_OLLAMA` |

Verified by instantiating it (no network, no code change):

```
$ .venv/bin/python -c "
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
o = VlmConvertOptions.from_preset('glm_ocr',
      engine_options=ApiVlmEngineOptions(engine_type=VlmEngineType.API_OLLAMA))
print(o.engine_options.url, o.model_spec.api_overrides[VlmEngineType.API_OLLAMA].params)"
http://localhost:11434/v1/chat/completions {'model': 'glm-ocr', 'max_tokens': 4096}
```

Preset defaults: prompt `Text Recognition:`, `scale=2.0`, `max_tokens=4096`,
`temperature=0.0`, `response_format=MARKDOWN`, `stop_strings=['<|user|>','<|endoftext|>']`.

**So the roadmap's "requires Docling ≥ v2.84" is satisfied — and then some. The
upgrade question is moot; there is nothing to upgrade and nothing to break.**
(I could not read the docling CHANGELOG to pin the exact version that first
added the preset — GitHub returned a render error for `CHANGELOG.md`. It does
not matter for the decision, but it is **unverified** which release introduced it.)

### End-to-end, on this host, with the pinned version

```python
# exactly what was run (read-only script, nothing written into the repo)
vlm  = VlmConvertOptions.from_preset("glm_ocr",
           engine_options=ApiVlmEngineOptions(engine_type=VlmEngineType.API_OLLAMA))
opts = VlmPipelineOptions(vlm_options=vlm, enable_remote_services=True)
conv = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts)})
conv.convert(pdf)
```

Two gotchas, both hit for real:

1. **`enable_remote_services=True` is mandatory even for `localhost`.** Without
   it: `docling.exceptions.OperationNotAllowed: Connections to remote services
   is only allowed when set explicitly.` If we ever ship this, that flag has to
   be documented — it *sounds* like it sends documents off-box and it does not.
2. Harmless shutdown noise: `Exception ignored in VlmConvertModel.__del__ …
   'NoneType' object has no attribute 'warning'` (docling bug, at exit only).

### Alternative worth knowing about (not tested)

`DCC-BS/docling-glm-ocr` (<https://github.com/DCC-BS/docling-glm-ocr>,
PyPI `docling-glm-ocr`) plugs GLM-OCR in as an **`ocr_options` engine inside the
classic PDF pipeline** — layout/table models keep running, and GLM-OCR only
recognises the OCR regions docling asks for. That is architecturally closer to
what localGPT wants (replace the OCR box, keep the rest) than `VlmPipeline`,
which replaces the whole pipeline. It targets a vLLM endpoint. **Unverified — I
did not install it.** Flagging it because it is the natural answer to defect #2
below.

---

## Q3 — End-to-end parse test

Method: render the page with PyMuPDF at the preset's `scale=2.0`, POST as a
base64 PNG. Scripts live in the scratchpad (`glmocr_smoke.py` → OpenAI-compatible
endpoint, `glmocr_native.py` → `/api/chat` with `num_ctx`), not in the repo.

### `eval/corpora/atlas7_service_manual.pdf` page 1 — verbatim model output

```
Atlas-7 Espresso Machine · Service Manual

Model: Atlas-7 Dual Boiler (2026 revision C)
Manufacturer: Meridian Coffee Systems, Tacoma WA

1. OPERATING SPECIFICATIONS
The brew boiler operates at a pressure of 9.2 bar during extraction.
The steam boiler is maintained at 1.45 bar. The PID controller keeps brew water at 93.5 degrees Celsius with a tolerance of 0.4 degrees.
The vibratory pump is rated for 52 watts continuous duty.

2. MAINTENANCE SCHEDULE
Descaling must be performed every 60 days when water hardness exceeds 120 ppm. The group head gasket (part MG-311) should be replaced every 14 months. Backflushing with Cafiza detergent is recommended weekly.
```

Against the PDF's own text layer: **every planted fact is exact** — 9.2 bar,
1.45 bar, 93.5 °C, 0.4 °C, 52 W, 60 days, 120 ppm, `MG-311`, 14 months, "Cafiza".
No hallucinated content. Only difference from the source is that hard line
wraps are joined into paragraphs, which is what we want for chunking.
Page 2 likewise exact (`TS-71`, `E11/E23/E42/E57`, 12 bar, 200 ml, 36-month,
8 percent, "under the drip tray on the left rail").

### Defect #1 — the page is sometimes transcribed twice

Deterministic, page-dependent, and it survives everything I threw at it:

```
atlas7 p1:  5.48s  chars=1284  half-vs-half similarity=0.97  DUPLICATED
atlas7 p2:  2.16s  chars=1282  half-vs-half similarity=0.97  DUPLICATED
northwind p1:  4.39s  chars=846  similarity=0.05  ok
northwind p2:  4.27s  chars=737  similarity=0.06  ok
northwind p3:  4.37s  chars=835  similarity=0.08  ok
invoice_scan p1:  3.68s  chars=1337  similarity=0.81  DUPLICATED
```

The second copy is usually wrapped in a ```` ```markdown ```` fence, and on
atlas7 p2 it actually contains a running header the *first* copy dropped. Ruled
out: `stop` strings (`<|user|>`, `<|endoftext|>` — passed explicitly, no effect),
`num_ctx` (4096 vs 16384 — identical output, so it is not context shift),
prompt wording (ignored entirely, see Q1a). Cause is the bare Ollama template /
missing stop token, i.e. it is a **packaging** problem, not a model problem —
which is consistent with the llama.cpp thread where a clean install fixed
repeat artifacts for another user. Cheap mitigation: de-duplicate identical
halves in post-processing. Real fix: MLX or a self-hosted llama-server with the
correct chat template.

### Scanned + tabular test vs. what localGPT runs today

Fixture (scratch, generated for this spike): a bordered 6×5 parts-invoice table
plus prose, rasterised at 150 dpi, greyscaled, rotated 0.6°, JPEG q45 —
**0 characters of text layer**, so it takes the OCR branch.

**GLM-OCR (12.6 s cold / 3.7 s warm):**

```
| Part No. | Description                  | Torque (Nm) | Qty | Price (EUR) |
| MG-311   | Group head gasket, 8.5 mm    | 12.5        | 2   | 4.20        |
| TS-71    | Brew thermistor, PT1000      | 3.0         | 1   | 18.75       |
| OPV-12   | Over-pressure valve, 12 bar  | 22.0        | 1   | 31.40       |
| FM-9     | Flow meter, 0.45 ml/pulse    | 6.5         | 1   | 27.90       |
| PMP-52   | Vibratory pump, 52 W         | n/a         | 1   | 63.05       |
Subtotal 145.30  VAT 19% 27.61  Total 172.91
```

**All 30 cells correct.**

**Current chain (`rag_system.ingestion.document_converter`, unmodified, 6.7 s):**

```
|      |                             | CCC CCE   |          |
| oa   | [ompmergmennsem             | _ CE      | CES      |
|      | Brew thermistor, PT1000     | 3.0       | a &#124; |
|      | Over-pressure valve, 12 bar | 22.0      |          |
| pus? | Vibratory pump, 52 W         | &#124;    |          |
```

Every price gone, every quantity gone, 4 of 5 part numbers destroyed
(`MG-311`→`oa`, `PMP-52`→`pus?`), the `FM-9` row dropped entirely, header row
garbage. Prose paragraphs outside the table came through fine in both.

### Defect #2 — docling's markdown path loses the table

Running the same scanned PDF through `VlmPipeline` + `API_OLLAMA`, the
`DoclingDocument` has **`tables: 0`** and the pipe table is flattened into a
single paragraph:

```
Part No. Description Torque (Nm) Qty Price (EUR) MG-311 Group head gasket, 8.5 mm 12.5 2 4.20 TS-71 …
```

The values are all still there and in reading order — far better than tesseract's
output — but the structure the model produced is thrown away by the time it
reaches `export_to_markdown()`, which is exactly what
`_perform_conversion()` in our converter consumes (it also hands the
`DoclingDocument` to structure-aware chunkers). Whether the duplication is what
breaks the markdown parse is **untested**. The raw HTTP response *does* contain
a clean pipe table, so a thin path that keeps the model's markdown would not
have this problem.

### Latency measured here (M2 Max, scale 2.0, `num_ctx=16384`)

| Scenario | Wall time |
|---|---|
| Cold (first request, model load) | 25.1 s |
| Warm, sequential, single page | **2.2 – 5.5 s/page** (4 consecutive runs: 8.3, 4.3, 3.7, 3.7 s) |
| Warm, 3-page PDF via docling (`concurrency=4`) | **4.38 s total → 1.46 s/page** |
| Current chain (tesseract-cli) same scanned page | 6.7 s/page |

Render-scale sweep on the scanned invoice: `scale=1.0` → 7.1 s, all values still
correct but emitted as plain rows instead of a markdown table; `scale=2.0` →
7.3 s, proper table; `scale=3.0` → 17.1 s, **no accuracy gain**. Docling's
preset default of 2.0 is the right operating point; do not raise it.

Upstream claims 1.86 pages/s for PDFs (Table 6,
<https://arxiv.org/html/2603.10910v1>) — that is on server GPUs, not comparable,
but our 1.46 pages/s wall with concurrency 4 is in the same order.

---

## Q4 — Cost/benefit for localGPT

### What our OCR chain actually resolves to on this machine (surprise)

`build_ocr_options()` walks `OcrMac → EasyOCR → RapidOCR → tesserocr →
tesseract-cli`. Run for real:

```
$ .venv/bin/python -c "from rag_system.ingestion.document_converter import build_ocr_options; print(build_ocr_options())"
OCR engine: TesseractCliOcrOptions
mode=FULL_PAGE lang=['fra','deu','spa','eng'] scale=3.0 force_full_page_ocr=True
```

`ocrmac`, `easyocr` and `tesserocr` are not installed; `rapidocr` **3.9.2 is**,
but the probe tests for the stale module name `rapidocr_onnxruntime`, so it is
skipped. **So the premise "OcrMac fallback" is not what runs here — everything
falls through to Tesseract CLI with four languages at once**, which is both the
slowest and the least accurate configuration in that list. Two much cheaper
wins than GLM-OCR exist: `pip install ocrmac` (Apple Vision, native, fast), and
fixing the RapidOCR module-name probe. Those should be measured before, or
alongside, any VLM work.

### Where GLM-OCR would and would not move the needle

| Document class | Verdict |
|---|---|
| Digital-born PDFs with a text layer | **No change.** The probe skips OCR entirely; GLM-OCR never runs unless we force it. Forcing it would be strictly worse (slower, and it re-flows text we already have perfectly). |
| Scanned pages, prose only | Moderate win. Tesseract handles clean prose acceptably; GLM-OCR is better on degraded input but this is not where the gap is. |
| **Scanned/photographed pages with tables** | **Large, demonstrated win** — 30/30 cells vs. near-total loss (above). This is the case that justifies the whole exercise. |
| Dense tables generally | Strong on paper: TableTEDS 93.96 / TableTEDS-S 96.39, best in class on OmniDocBench v1.5 (arXiv 2603.10910). |
| Formulas | FormulaCDM 93.90 — but our stack has nothing downstream that uses LaTeX, so this is not a localGPT win today. |
| **Handwriting** | **Do not promise this.** Reported 86.1 vs Gemini 3 Pro's 94.5 on handwritten KIE. Weakest category. |

### Correction to the roadmap's evidence line

Roadmap 1.3 says "95.22 OmniDocBench vs GPT-5.2's 86.59". The 95.22 figure is
**OmniDocBench v1.6_full**, where GLM-OCR is **third**, behind PaddleOCR-VL-1.6
(96.34) and MinerU2.5-Pro (95.75). The "#1, 94.62" claim is the older **v1.5**
table (arXiv 2603.10910, where PaddleOCR-VL-1.5 is 94.50 — a 0.12-point gap,
i.e. inside the noise our own roadmap says not to trust). I could not verify a
"GPT-5.2 86.59" row anywhere. **The roadmap row should be corrected: GLM-OCR is
an excellent small OCR model, not the uncontested leader, and PaddleOCR-VL-1.6
now scores higher on the current benchmark.** Docling 2.118.1 also ships presets
for `lightonocr`, `dots_ocr`, `chandra_ocr2`, `falcon_ocr` and `nanonets_ocr2` —
if we build a scanned-document eval, GLM-OCR should be one column in it, not the
foregone conclusion.

---

## Recommendation

### GO-LATER — prototype behind a flag, do not adopt as default yet

**Why not NO-GO:** the Apple Silicon serving path the roadmap called "unproven"
is now proven. It costs **zero new Python dependencies** (our pinned docling
already drives it), one 2.2 GB Ollama model, 2.8 GB of GPU memory while loaded,
and it fixes a failure mode that today loses *all* numeric content in scanned
tables. Warm latency (2–5 s/page, 1.5 s/page with concurrency) is acceptable for
an ingestion-time, opt-in path.

**Why not GO:** three things must land first, and none of them is code we should
write blind:

1. **Duplication** — pages are sometimes emitted twice through the Ollama build.
   Needs either a de-dup post-process (cheap, ugly) or the MLX / custom-template
   serving path (correct).
2. **Table structure is lost** in docling's markdown→`DoclingDocument` step
   (`tables: 0`), which is most of the value. Needs either the raw-markdown path
   or the `ocr_options`-style plugin shape.
3. **No eval to decide with.** Phase 0 has retrieval and groundedness metrics but
   **no ingestion-quality corpus**: `eval/corpora/` contains only digital-born
   PDFs with clean text layers, so today nothing in the harness would even
   exercise the OCR branch, let alone score it. Adopting 1.3 without that would
   violate the roadmap's own decision gate.

**Sequencing:** do the cheap OCR fixes first (`ocrmac`, RapidOCR probe name),
build a small scanned/tabular corpus with known ground truth, then A/B
GLM-OCR against the fixed baseline — and against `lightonocr` / `dots_ocr`,
which are already one line away in the same preset registry.

### Integration path when it goes ahead (config sketch, no code)

Route **per page**, not per document — today's probe is "any page has text →
no OCR for the whole document", so a scanned insert inside a digital PDF gets
nothing at all. That is a bug worth fixing independently of GLM-OCR.

```
# opt-in; default off — the current chain stays the fallback
PARSER_VLM              = off | glm-ocr        # default: off
PARSER_VLM_ENDPOINT     = http://localhost:11434
PARSER_VLM_MODEL        = glm-ocr
PARSER_VLM_SCALE        = 2.0                  # do not raise; 3.0 costs 2.3x for no gain
PARSER_VLM_CONCURRENCY  = 4
PARSER_VLM_TIMEOUT_S    = 90
PARSER_VLM_MIN_CHARS    = 32                   # per-page text-layer threshold
```

Behaviour: per-page probe → pages under `PARSER_VLM_MIN_CHARS` go to the VLM;
everything else keeps the existing text-layer path. On timeout, HTTP error, or
empty output, fall back to the current OCR chain for that page — never fail the
ingest. De-duplicate identical output halves before chunking. Docling wiring is
`VlmConvertOptions.from_preset("glm_ocr", engine_options=ApiVlmEngineOptions(
engine_type=VlmEngineType.API_OLLAMA))` plus **`enable_remote_services=True`**.

### Open risks

- **"Remote services" flag.** Turning on a docling option literally named
  `enable_remote_services` in a product called *localGPT* needs a comment in the
  code and a line in the docs saying it points at `localhost:11434`. If a user
  ever overrides `PARSER_VLM_ENDPOINT`, documents leave the box.
- **Ollama build ignores prompts.** We are locked to one recognition mode, and a
  future Modelfile change upstream could silently alter output. Pin the tag.
- **Second runtime dependency at ingest time.** Ingestion would fail-soft but get
  much slower if Ollama is down or busy serving the chat model — GLM-OCR
  competes for the same GPU as the generation model.
- **Ollama version floor.** The model requires Ollama ≥ 0.15.5; we would need to
  state that (this host runs 0.32.6).
- **Docker.** `docker-compose.local-ollama.yml` exists, but nothing here was
  tested in a container, and the 2.2 GB model would need to be present in
  whichever Ollama the container talks to.
- **Benchmark drift.** OmniDocBench moved from v1.5 to v1.6_full between the
  roadmap being written and this spike, and GLM-OCR moved from 1st to 3rd. Do
  not re-cite leaderboard positions in our docs; cite our own eval or nothing.
- **Table loss may be a docling bug we cannot fix from config.** If the
  markdown→document parse is the blocker, the integration shape has to change
  (raw markdown, or the `ocr_options` plugin), which is more work than the
  "config sketch" above implies.

---

## Reproducing this

Scratch scripts (not in the repo):
`/private/tmp/claude-501/-Users-prompt-videos-localgpt-08082026/4d62420b-7ab2-4be1-90f2-708d7bae9146/scratchpad/`
— `glmocr_smoke.py` (OpenAI-compatible endpoint), `glmocr_native.py`
(`/api/chat`, exposes `num_ctx`), `make_scan_fixture.py` (builds the scanned
invoice fixture). The pulled model `glm-ocr:latest` was kept, since it works.

### Sources

- <https://ollama.com/library/glm-ocr>
- <https://huggingface.co/zai-org/GLM-OCR>
- <https://github.com/zai-org/GLM-OCR> · <https://github.com/zai-org/GLM-OCR/blob/main/examples/ollama-deploy/README.md>
- <https://arxiv.org/html/2603.10910v1> (GLM-OCR technical report) · <https://arxiv.org/html/2603.10910v2>
- <https://huggingface.co/mlx-community/GLM-OCR-bf16>
- <https://github.com/ggml-org/llama.cpp/discussions/19721> · <https://huggingface.co/blog/ggml-org/using-ocr-models-with-llama-cpp>
- <https://docs.vllm.ai/en/stable/getting_started/installation/cpu/> · <https://github.com/vllm-project/vllm-metal>
- <https://github.com/DCC-BS/docling-glm-ocr> (unverified alternative)
- <https://docling-project.github.io/docling/reference/pipeline_options/>

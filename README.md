# GPTQ

![license](https://img.shields.io/badge/license-license-blue) ![offline-first](https://img.shields.io/badge/offline--first-air--gap-green) ![audit](https://img.shields.io/badge/audit-SHA3--256-orange) ![category](https://img.shields.io/badge/category-chip_quantization-lightgrey)

> Anticloud-hardened packaging of the upstream project `GPTQ` in category **CHIP QUANTIZATION**. Upstream source is vendored in `UPSTREAM_CLONE/` at the pinned commit below; the 12-improvement overlay lives in `anticloud/`. Every fact in this file traces to a file on disk in this project directory.

**Category:** CHIP QUANTIZATION · **Upstream:** https://github.com/AutoGPTQ/AutoGPTQ · **Upstream pin:** `9f7d37072917ab3a7545835f23e808294a542153` · **Vendor:** Anticloud FZ LLE

---

## What This Project Does

# GPTQ

This project contains the code for the ICLR 2023 paper [GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323). 
The current release includes the following features:

* An efficient implementation of the GPTQ algorithm: `gptq.py`
* Compressing all models from the OPT and BLOOM families to 2/3/4 bits, including weight grouping: `opt.py`, `bloom.py`, `zeroShot/`
* Evaluating the perplexity of quantized models on several language generation tasks: `opt.py`, `bloom.py`
* Evaluating the performance of quantized models on several ZeroShot tasks: `zeroShot/`
* A 3-bit quantized matrix full-precision vector product CUDA kernel: `quant_cuda_kernel.cu`, `quant_cuda.cpp`, `setup_cuda.py`
* Benchmarking code for individual matrix-vector products and for language generation with quantized models: `test_kernel.py`, `opt.py`

## New Features

Update July 2023:

* Added `--static-groups` options which determines all group-grids in advance rather than dynamically during quantization, which has the effect that `--act-order` does not require any inference changes (that may cause slowdown) when used together with this option.

Together with the camera ready version of the paper we have added several updates to this project:

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag `--new-eval`.
* Optimized 3bit kernels, which are considerably faster especially on the A100, e.g. 1.9x -> 3.25x generation speedup for OPT-175B; can be activated via `--faster-kernel`.
* A minimal LlaMa integration (for more complete features see the [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) project), which demonstrates two new tricks:`--act-order` (quantizing columns in order of decreasing activation size) and `--true-sequential` (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general.

Here is a summary of LLaMa results:

| Wiki2 PPL | FP16 | 4bit-RTN | 4bit-GPTQ | 3bit-RTN | 3bit-GPTQ | 3g128-GPTQ |
|:---------:|:----:|:--------:|:---------:|:--------:|:---------:|:----------:|
| LLaMa-7B  | 5.68 | 6.29     | **6.09**  | 25.54    | **8.07**  | 6.61       |
| LLaMa-13B | 5.09 | 5.53     | **5.36**  | 11.40    | **6.63**  | 5.62       |
| LLaMa-30B | 4.10 | 4.54     | **4.45**  | 14.89    | **5.69**  | 4.80       |
| LLaMa-65B | 3.53 | 3.92     | **3.84**  | 10.59    | **5.04**  | 4.17       |

Here is a sample command:

```
python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval
```

The `--act-order` heuristic also dramatically improves accuracy on the OPT-66B outlier model: 9.55 to 9.34 and 14.16 to 9.95 PPL on Wiki2 for 4bit and 3bit, respectively.

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2 (the LLaMa integration currently requires a main install from source and `sentencepiece`)
* `datasets`: tested on v1.17.0
* (to run 3-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.4)

All experiments were run on a single 80GB NVIDIA A100. However, most experiments will work on a GPU with a lot less memory as well.

## Language Generation

### OPT

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 [--groupsize 1024]
````

To run other OPT models replace `opt-125m` with one of: `opt-350m`, `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-66b`.
For the 175B-parameter mode, you have to request access from Meta and then convert it to a local HuggingFace checkpoint using their scripts in `metaseq`.
Once you have such a checkpoint, simply pass its path instead of `facebook/opt-125m`. 

### BLOOM

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 [--groupsize 1024]
````

To run other BLOOM models replace `bloom-560m` with one of: `bloom-1b1`, `bloom-1b7`, `bloom-3b`, `bloom-7b1`, `bloom`.

## ZeroShot

See `zeroShot/` folder.

## 3-bit CUDA Kernels 

```
# Install kernels
python setup_cuda.py install

# Benchmark performance for FC2 layer of OPT-175B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py

# Benchmark language generation with 3-bit OPT-175B:
# OPT175B denotes the name of the folder with the HuggingFace OPT-175b checkpoint (see above)

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python opt.py OPT175B c4 --wbits 3 --save opt175-3bit.pt
# Benchmark generating a 128 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python opt.py OPT175B c4 --load opt175b-3bit.pt --benchmark 128
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python opt.py OPT175B c4 --benchmark 128
```

Please note that our 3-bit kernels are currently only optimized for OPT-175B running on 1xA100 or 2xA6000 and may thus yield suboptimal performance on smaller models or on other GPUs.

## Cite

If you found this work useful, please consider citing:

```
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2210.17323}
}
```

*Quoted from the upstream `README.md` file in `UPSTREAM_CLONE/`.*
Project-specific facts detected in this directory:

- Ecosystem: **Unknown (no standard manifest detected)** (manifests: none detected; scanned in UPSTREAM_CLONE)
- Top-level source layout: `zeroShot/`
- Snapshot size: **41 files**, **6820 lines of code** (measured; see Benchmarks)
- Primary languages: `.py` (32), `.md` (3), `(none)` (2), `.cpp` (1), `.csv` (1), `.cu` (1)
- Upstream commit pinned for this packaging: `9f7d37072917ab3a7545835f23e808294a542153`

---

## Installation

* Benchmarking code for individual matrix-vector products and for language generation with quantized models: `test_kernel.py`, `opt.py`

*Section quoted from the upstream readme.*
Overlay install (this project):

```sh
python -m pip install -e anticloud/     # overlay package with the 12 improvements
python anticloud/cli.py --help          # 13 subcommands, JSON stdout
```

---

## Usage

Together with the camera ready version of the paper we have added several updates to this project:

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag `--new-eval`.
* Optimized 3bit kernels, which are considerably faster especially on the A100, e.g. 1.9x -> 3.25x generation speedup for OPT-175B; can be activated via `--faster-kernel`.
* A minimal LlaMa integration (for more complete features see the [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) project), which demonstrates two new tricks:`--act-order` (quantizing columns in order of decreasing activation size) and `--true-sequential` (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general.

Here is a summary of LLaMa results:

| Wiki2 PPL | FP16 | 4bit-RTN | 4bit-GPTQ | 3bit-RTN | 3bit-GPTQ | 3g128-GPTQ |
|:---------:|:----:|:--------:|:---------:|:--------:|:---------:|:----------:|
| LLaMa-7B  | 5.68 | 6.29     | **6.09**  | 25.54    | **8.07**  | 6.61       |
| LLaMa-13B | 5.09 | 5.53     | **5.36**  | 11.40    | **6.63**  | 5.62       |
| LLaMa-30B | 4.10 | 4.54     | **4.45**  | 14.89    | **5.69**  | 4.80       |
| LLaMa-65B | 3.53 | 3.92     | **3.84**  | 10.59    | **5.04**  | 4.17       |

Here is a sample command:

```
python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval
```

The `--act-order` heuristic also dramatically improves accuracy on the OPT-66B outlier model: 9.55 to 9.34 and 14.16 to 9.95 PPL on Wiki2 for 4bit and 3bit, respectively.

*Section quoted from the upstream readme.*
Anticloud overlay CLI (available in every project):

```sh
python anticloud/cli.py --help     # 13 subcommands, JSON stdout
python anticloud/cli.py checks     # run the 16-check suite
```

---

## API

The upstream API surface is defined by the `GPTQ` source tree vendored in `UPSTREAM_CLONE/` (Unknown (no standard manifest detected) ecosystem). Public entry points:

- Source modules: `zeroShot/`
- Overlay API: `anticloud/cli.py` exposes 13 subcommands with JSON stdout; `anticloud/bench/runner.py` runs the 16-check suite; `anticloud/provenance/chain.py` exposes the SHA3-256 + Ed25519 provenance chain.

---

## Dependencies

| Metric | Value |
|--------|-------|
| Ecosystem | Unknown (no standard manifest detected) |
| Manifests detected | none |
| Files in snapshot | 41 |
| Lines of code | 6820 |
| Dependency references | 0 |
| Upstream license | Apache-2.0 |
| Overlay license | Anticommons 0.1.0 |

Pinned lockfile: `anticloud/requirements.lock` (hash-pinned, PEP 508). SBOM: `sbom.cdx.json` (CycloneDX 1.5, pinned to the upstream SHA).

---

## Configuration

Together with the camera ready version of the paper we have added several updates to this project:

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag `--new-eval`.
* Optimized 3bit kernels, which are considerably faster especially on the A100, e.g. 1.9x -> 3.25x generation speedup for OPT-175B; can be activated via `--faster-kernel`.
* A minimal LlaMa integration (for more complete features see the [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) project), which demonstrates two new tricks:`--act-order` (quantizing columns in order of decreasing activation size) and `--true-sequential` (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general.

Here is a summary of LLaMa results:

| Wiki2 PPL | FP16 | 4bit-RTN | 4bit-GPTQ | 3bit-RTN | 3bit-GPTQ | 3g128-GPTQ |
|:---------:|:----:|:--------:|:---------:|:--------:|:---------:|:----------:|
| LLaMa-7B  | 5.68 | 6.29     | **6.09**  | 25.54    | **8.07**  | 6.61       |
| LLaMa-13B | 5.09 | 5.53     | **5.36**  | 11.40    | **6.63**  | 5.62       |
| LLaMa-30B | 4.10 | 4.54     | **4.45**  | 14.89    | **5.69**  | 4.80       |
| LLaMa-65B | 3.53 | 3.92     | **3.84**  | 10.59    | **5.04**  | 4.17       |

Here is a sample command:

```
python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval
```

The `--act-order` heuristic also dramatically improves accuracy on the OPT-66B outlier model: 9.55 to 9.34 and 14.16 to 9.95 PPL on Wiki2 for 4bit and 3bit, respectively.

*Section quoted from the upstream readme.*
Overlay configuration (Anticloud):

- `anticloud/` - improvement overlay; environment-driven, no cloud dependency
- `LEDGERS/` - aioss tamper-evident chain files (per-project, verified with `aioss verify --live`)
- `ISOLATED_LAB_RESULTS/` - reproducibility record (environment, reproduction steps, result register, evidence)
- `OFFICIAL_BENCHMARKS/` - 26 framework assessments for this project

---

## Contributing

Upstream contributions: fork the `GPTQ` project, create a feature branch, and open a pull request against upstream. Keep `UPSTREAM_CLONE/` untouched in this packaging; put improvements in the `anticloud/` overlay.

Overlay contributions: run the 16-check suite before opening a pull request:

```sh
python anticloud/bench/runner.py --cwd anticloud
```

---

## License

**Upstream license: Apache-2.0** (evidence: `LICENSE` in the upstream snapshot).

License file excerpt:

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
```

### Anticommons 0.1.0 overlay

The Anticloud integration overlay in `anticloud/` - improvements 1 through 12 listed under Benchmarks - is licensed under **Anticommons 0.1.0**. Upstream code remains under its original Apache-2.0 terms. See `ANTICOMMONS_LICENSE.md` in this directory for the overlay terms and contact.

SPDX: `Apache-2.0` (upstream) + Anticommons 0.1.0 (overlay, dual).

---

## Upstream

- **Project:** `GPTQ` (category: CHIP QUANTIZATION)
- **Upstream URL:** https://github.com/AutoGPTQ/AutoGPTQ
- **Pinned commit (SHA):** `9f7d37072917ab3a7545835f23e808294a542153`
- **Branch:** main
- **Pin provenance:** GitHub API commits/main. The parent-project stamp is explicitly rejected for this project.
- **Snapshot location:** `UPSTREAM_CLONE/` (vendored, not shipped as-is)
- **Benchmark snapshot:** `BENCH.json`

---

## Benchmarks

Measured by the Anticloud assurance suite. Every value below is read from this
project's `BENCH.json`, produced by a real run — the SHA3-256 of that file is
`01487f9453665a5d8f1a9dbbc6f0f9d96811dfdc63cf71374c13b24643f0da04`.

| Framework | Controls | Evidence | Coverage | Result |
|---|---|---|---|---|
| OWASP Top 10 for LLM Applications | 10 controls mapped | 10 with evidence | 100.0% | PASS |
| OWASP Top 10 (2021) | 9 controls mapped | 9 with evidence | 100.0% | PASS |
| SOC 2 Type II readiness | 9 controls mapped | 9 with evidence | 100.0% | PASS |
| NIST AI Risk Management Framework | 8 controls mapped | 8 with evidence | 100.0% | PASS |
| NIST SP 800-53 Rev. 5 | 12 controls mapped | 12 with evidence | 100.0% | PASS |
| NIST Cybersecurity Framework 2.0 | 8 controls mapped | 8 with evidence | 100.0% | PASS |
| FedRAMP Rev. 5 | 10 controls mapped | 10 with evidence | 100.0% | PASS |
| PCI DSS v4.0.1 | 11 controls mapped | 11 with evidence | 100.0% | PASS |
| ISO/IEC 27001:2022 | 9 controls mapped | 9 with evidence | 100.0% | PASS |
| MITRE ATT&CK v16 | 12 controls mapped | 12 with evidence | 100.0% | PASS |
| ML Technology Readiness Level | TRL 8 | 8/8 criteria | | PASS |

**Overall: 16/16 checks passing.**

See `ISOLATED_LAB_RESULTS/03_Result_Register.md` for the 16-check register with pass condition, command and observed value per check.

Framework folders in `OFFICIAL_BENCHMARKS/` state the control set and the
evidence source bound to each control. This project does not claim an audit
opinion, a SOC report, a FedRAMP authorisation or a PCI attestation — those are
issued by an independent assessor.



## Archives and Permanent Records

| Platform | Identifier | Volume |
|---|---|---|
| Harvard Dataverse | DOI 10.7910/DVN/YMJKOG | 145 citable datasets |
| AIOSS verification kit | DOI 10.7910/DVN/OORKNJ | Offline hash verification |
| Zenodo (CERN) | — | 146 records, DOI-registered |
| OSF | — | 144 preregistered records |
| Figshare | author 20849885 | Research data and figures |
| Internet Archive | aioss-format, Anticode | Permanent binary specification |
| ORCID | 0009-0009-2233-6107 | Permanent researcher ID |
| Kaggle | pax-millennium-20 | Reproducible T4 benchmark run |



## Press and Independent Publication

The PAX benchmark release was distributed by Newsfile wire to 336 outlets
(312 Web, 23 Terminal, 1 Application), including Yahoo Finance, The Globe and Mail, Business Insider, National Post, Financial Post, StreetInsider, Digital Journal, Barchart, International Business Times, and fox47 · National Law Review.
Wire distribution makes the announcement dated, public, and indexed, which
makes the claim checkable.


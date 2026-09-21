# Lightweight PC-STE Paper

Publisher-neutral LaTeX manuscript for:

**Lightweight PC-STE: Performance-Preserving Compression of a Shared Multi-Dataset State-Space Encoder for Rotating-Machinery Fault Diagnosis**

## Author order

1. Mohammad Abdalaziz
2. Erick Giovani Sperandio Nascimento
3. Marco Aurélio da Silva Cruz

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex   # the full-page confusion-matrix float needs one extra pass
```

Or simply `latexmk -pdf main.tex`.

### Requirements

Packages, all present in a standard TeX Live / Overleaf installation:
`geometry`, `graphicx`, `booktabs`, `multirow`, `tabularx`, `array`, `amsmath`,
`amssymb`, `microtype`, `enumitem`, `hyperref`, `cleveref`, `natbib`, `xcolor`.
No custom `.cls` or `.sty` file is required. Figures are vector PDFs under
`figures/` and are included by the float files in the same directory.

The current project contains complete draft prose for all five manuscript sections, the abstract, bibliography, publication-result tables, and Figures 1--6. The LaTeX shell is intentionally publisher-neutral so it can later be migrated to `elsarticle`, `IEEEtran`, Springer, or another target-journal template without rewriting the scientific sections.

## Scientific stage separation

- **Primary study:** Full-S1 vs K1 under the sealed nine-cell publication protocol.
- **Efficiency:** publication benchmark with the separate inference-mode timing correction used as the authoritative latency/throughput source.
- **Secondary extension:** Q8(K1), kept separate from the original confirmatory Full-S1 vs K1 comparison.

## Items to finalize with co-authors / target journal

- target journal and document class;
- corresponding author and full affiliation formatting;
- final architecture/overview artwork if desired;
- funding and acknowledgements;
- public repository URL and archival DOI;
- final CRediT roles and competing-interest statement;
- journal-specific word-count, graphical-abstract, and supplementary-material requirements.

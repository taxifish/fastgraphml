# Guides

| Guide | Content |
| ----- | ------- |
| [fastgraphml explained](fastgraphml-explained.pdf) | What the library is for, how it turns a graph into embeddings step by step, the four models, reference tables |
| [Getting started with fastgraphml](fastgraphml-getting-started.pdf) | Installation, first embeddings, your own graph in ArangoDB, the three example projects, troubleshooting |

Both PDFs are generated from the Python sources in `src/` (text, code samples and
the diagrams, which are drawn as SVG):

```bash
pip install pygments
python docs/src/build.py        # needs Google Chrome or Chromium for printing to PDF
```

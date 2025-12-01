from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import plotly.io as pio
from jinja2 import Template

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Audio Analysis Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; }
    h1, h2 { color: #222; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .card { padding: 12px 16px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
    .section { margin-top: 24px; }
  </style>
</head>
<body>
  <h1>Audio Analysis Report</h1>
  <p><strong>Sample:</strong> {{ sample_name }}</p>
  <p><strong>Predicted Genre:</strong> {{ top_genre }} (confidence {{ top_genre_conf|round(3) }})</p>
  <p><strong>Authenticity Score (AI=1):</strong> {{ authenticity_score|round(3) }}</p>

  <div class="section">
    <h2>Key Metrics</h2>
    <div class="cards">
      {% for key, value in metrics.items() %}
      <div class="card">
        <strong>{{ key }}</strong>
        <div>{{ value }}</div>
      </div>
      {% endfor %}
    </div>
  </div>

  <div class="section">
    <h2>Figures</h2>
    {% for name, fig_html in figures.items() %}
      <h3>{{ name }}</h3>
      {{ fig_html | safe }}
    {% endfor %}
  </div>
</body>
</html>
"""


def build_report(
    sample_name: str,
    top_genre: str,
    top_genre_conf: float,
    authenticity_score: float,
    metrics: Dict[str, str],
    figures: Dict[str, object],
    html_path: Path,
    pdf: bool = False,
    pdf_path: Optional[Path] = None,
) -> Path:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig_html: Dict[str, str] = {name: pio.to_html(fig, include_plotlyjs="cdn", full_html=False) for name, fig in figures.items()}
    template = Template(HTML_TEMPLATE)
    rendered = template.render(
        sample_name=sample_name,
        top_genre=top_genre,
        top_genre_conf=top_genre_conf,
        authenticity_score=authenticity_score,
        metrics=metrics,
        figures=fig_html,
    )
    html_path.write_text(rendered, encoding="utf-8")
    if pdf:
        _render_pdf(rendered, pdf_path or html_path.with_suffix(".pdf"))
    return html_path


def _render_pdf(html: str, pdf_path: Path) -> None:
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("weasyprint is required for PDF rendering") from exc
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(pdf_path)

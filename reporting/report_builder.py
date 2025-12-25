from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import plotly.io as pio
from jinja2 import Template

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Audio Analysis Report</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #111827;
      --accent: #22d3ee;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --radius: 14px;
      --shadow: 0 14px 40px rgba(0,0,0,0.25);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 32px;
      font-family: 'Segoe UI', 'Inter', system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at 10% 20%, rgba(34,211,238,0.15), transparent 25%), radial-gradient(circle at 80% 0%, rgba(94,92,255,0.15), transparent 22%), var(--bg);
      color: var(--text);
    }
    h1, h2, h3 { margin: 0 0 8px 0; color: var(--text); }
    p { color: var(--muted); margin: 4px 0; }
    .container { max-width: 1200px; margin: 0 auto; }
    .hero {
      background: linear-gradient(135deg, rgba(34,211,238,0.18), rgba(17,24,39,0.7));
      border: 1px solid rgba(34,211,238,0.25);
      padding: 20px;
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      margin-bottom: 20px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.05);
      padding: 14px 16px;
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .label { color: var(--muted); font-size: 13px; }
    .value { font-size: 20px; font-weight: 600; color: var(--text); }
    .section { margin: 24px 0; }
    .figure {
      background: var(--card);
      border-radius: var(--radius);
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.05);
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }
    @media (max-width: 640px) {
      body { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>Audio Analysis Report</h1>
      <p><strong>Sample:</strong> {{ sample_name }}</p>
      <p><strong>Predicted Genre:</strong> {{ top_genre }} (confidence {{ top_genre_conf|round(3) }})</p>
      <p><strong>Authenticity Score (AI=1):</strong> {{ authenticity_score|round(3) }}</p>
    </div>

    <div class="section">
      <h2>Key Metrics</h2>
      <div class="grid">
        {% for key, value in metrics.items() %}
        <div class="card">
          <div class="label">{{ key }}</div>
          <div class="value">{{ value }}</div>
        </div>
        {% endfor %}
      </div>
    </div>

    <div class="section">
      <h2>Figures</h2>
      {% for name, fig_html in figures.items() %}
        <div class="figure">
          <h3>{{ name }}</h3>
          {{ fig_html | safe }}
        </div>
      {% endfor %}
    </div>
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
    """
    Build an HTML (and optionally PDF) report for audio analysis.
    
    Args:
        sample_name: Name of the audio sample
        top_genre: Predicted genre label
        top_genre_conf: Confidence score for top genre
        authenticity_score: AI authenticity score (0-1)
        metrics: Dictionary of metrics to display
        figures: Dictionary of plotly figures
        html_path: Output path for HTML report
        pdf: Whether to generate PDF output
        pdf_path: Optional custom PDF output path
        
    Returns:
        Path to generated HTML report
        
    Raises:
        RuntimeError: If report generation fails
    """
    try:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Converting {len(figures)} figures to HTML")
        fig_html: Dict[str, str] = {}
        for name, fig in figures.items():
            try:
                fig_html[name] = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
            except Exception as e:
                logger.error(f"Failed to convert figure '{name}' to HTML: {e}")
                fig_html[name] = f"<p>Error rendering figure: {e}</p>"
        
        template = Template(HTML_TEMPLATE)
        rendered = template.render(
            sample_name=sample_name,
            top_genre=top_genre,
            top_genre_conf=top_genre_conf,
            authenticity_score=authenticity_score,
            metrics=metrics,
            figures=fig_html,
        )
        
        logger.info(f"Writing HTML report to {html_path}")
        html_path.write_text(rendered, encoding="utf-8")
        
        if pdf:
            try:
                _render_pdf(rendered, pdf_path or html_path.with_suffix(".pdf"))
            except Exception as e:
                logger.warning(f"Failed to generate PDF: {e}")
        
        logger.info("Report generated successfully")
        return html_path
        
    except Exception as e:
        logger.error(f"Failed to build report: {e}")
        raise RuntimeError(f"Failed to build report: {e}") from e


def _render_pdf(html: str, pdf_path: Path) -> None:
    """
    Render HTML to PDF using WeasyPrint.
    
    Args:
        html: HTML content as string
        pdf_path: Output path for PDF
        
    Raises:
        RuntimeError: If PDF rendering fails
    """
    try:
        from weasyprint import HTML
    except ImportError as exc:
        logger.error("weasyprint is not installed")
        raise RuntimeError("weasyprint is required for PDF rendering") from exc
    
    try:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Rendering PDF to {pdf_path}")
        HTML(string=html).write_pdf(pdf_path)
        logger.info("PDF rendered successfully")
    except Exception as e:
        logger.error(f"Failed to render PDF: {e}")
        raise RuntimeError(f"Failed to render PDF: {e}") from e

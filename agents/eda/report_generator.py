# agents/reporting/report_generator.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Report Generator                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade multi-format report generation:                          ║
║    ✓ HTML Reports (interactive, styled, print-ready)                      ║
║    ✓ PDF Reports (with WeasyPrint fallback to HTML)                       ║
║    ✓ Markdown Reports (portable, version-control friendly)                ║
║    ✓ Executive Summary generation                                         ║
║    ✓ Responsive design with mobile support                                ║
║    ✓ Comprehensive section builders (overview, stats, missing, etc.)      ║
║    ✓ Automatic timestamping & metadata                                    ║
║    ✓ Intelligent fallbacks for missing data sections                      ║
║    ✓ Output path auto-generation with timestamps                          ║
║    ✓ Comprehensive error handling & telemetry                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "report_path": str,
    "format": "html" | "pdf" | "markdown",
    "generated_at": str (ISO format),
    "file_size_bytes": int,
    "sections_included": List[str],
    "telemetry": {
        "elapsed_ms": float,
        "pdf_conversion_attempted": bool,
        "pdf_conversion_success": bool,
        "weasyprint_available": bool,
    },
    "metadata": {
        "title": str,
        "generator": str,
        "version": str,
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import wraps

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
    # Fallback for testing
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data = None
            self.errors = []
            self.warnings = []
        
        def add_error(self, msg: str):
            self.errors.append(msg)
        
        def add_warning(self, msg: str):
            self.warnings.append(msg)

# PDF generation (optional)
try:
    from weasyprint import HTML as WeasyprintHTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.debug("WeasyPrint not available — PDF generation will fallback to HTML")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReportGeneratorConfig:
    """Enterprise configuration for report generation."""
    
    # Output paths
    default_reports_dir: Path = Path("reports")
    
    # Report styling
    primary_color: str = "#1f77b4"
    secondary_color: str = "#2c3e50"
    gradient_start: str = "#667eea"
    gradient_end: str = "#764ba2"
    
    # Content limits
    max_table_rows: int = 10           # Max rows to show in tables
    max_correlation_pairs: int = 10     # Max correlation pairs to display
    max_columns_display: int = 5        # Max columns in detailed views
    
    # PDF generation
    pdf_dpi: int = 96
    pdf_enable_forms: bool = False
    
    # Metadata
    generator_name: str = "DataGenius PRO Master Enterprise ++++"
    generator_version: str = "5.0-kosmos-enterprise"
    
    # Features
    include_css_animations: bool = True
    include_print_styles: bool = True
    responsive_breakpoint: str = "768px"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """Decorator for operation timing with intelligent logging."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.2f}ms")
        return wrapper
    return decorator


def _safe_operation(operation_name: str, default_value: Any = None):
    """Decorator for defensive operations with fallback values."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠ {operation_name} failed: {type(e).__name__}: {str(e)[:80]}")
                return default_value
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Report Generator Agent
# ═══════════════════════════════════════════════════════════════════════════

class ReportGenerator(BaseAgent):
    """
    **ReportGenerator** — Enterprise multi-format EDA report generation.
    
    Responsibilities:
      1. HTML report generation (interactive, styled, responsive)
      2. PDF report generation (with WeasyPrint or fallback)
      3. Markdown report generation (portable, VCS-friendly)
      4. Executive summary compilation
      5. Section builders for all EDA components
      6. Automatic path generation with timestamps
      7. Comprehensive error handling with graceful degradation
      8. File size tracking & metadata
      9. Telemetry for PDF conversion attempts
      10. Zero dependencies on external report templates
    
    Features:
      • Responsive design with mobile breakpoints
      • Print-optimized CSS
      • Gradient card designs
      • Interactive tables with hover effects
      • Intelligent fallbacks for missing sections
      • ISO timestamp generation
    """
    
    def __init__(self, config: Optional[ReportGeneratorConfig] = None) -> None:
        """Initialize generator with optional custom configuration."""
        super().__init__(
            name="ReportGenerator",
            description="Generates comprehensive EDA reports in multiple formats"
        )
        self.config = config or ReportGeneratorConfig()
        self._log = logger.bind(agent="ReportGenerator")
        warnings.filterwarnings("ignore")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            eda_results: Dict[str, Any]
            data_info: Dict[str, Any]
        
        Optional:
            format: str (html, pdf, markdown)
            output_path: Path
        """
        if "eda_results" not in kwargs:
            raise ValueError("Required parameter 'eda_results' not provided")
        
        if "data_info" not in kwargs:
            raise ValueError("Required parameter 'data_info' not provided")
        
        if not isinstance(kwargs["eda_results"], dict):
            raise TypeError(f"'eda_results' must be dict, got {type(kwargs['eda_results']).__name__}")
        
        if not isinstance(kwargs["data_info"], dict):
            raise TypeError(f"'data_info' must be dict, got {type(kwargs['data_info']).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("ReportGenerator.execute")
    def execute(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        format: str = "html",
        output_path: Optional[Path] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Generate comprehensive EDA report in specified format.
        
        Args:
            eda_results: Results from EDA Orchestrator
            data_info: Basic dataset information
            format: Report format ("html", "pdf", "markdown")
            output_path: Optional custom output path
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with report generation details (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0 = time.perf_counter()
        
        try:
            cfg = self.config
            
            # Validate format
            format = format.lower()
            if format not in ("html", "pdf", "markdown"):
                raise ValueError(f"Unsupported format: {format}. Use 'html', 'pdf', or 'markdown'")
            
            # Track telemetry
            pdf_conversion_attempted = False
            pdf_conversion_success = False
            sections_included: List[str] = []
            
            # ─── Generate Report Based on Format
            if format == "html":
                report_path = self._generate_html_report(
                    eda_results=eda_results,
                    data_info=data_info,
                    output_path=output_path
                )
                sections_included = self._get_sections_list(eda_results)
            
            elif format == "pdf":
                pdf_conversion_attempted = True
                report_path, pdf_conversion_success = self._generate_pdf_report(
                    eda_results=eda_results,
                    data_info=data_info,
                    output_path=output_path
                )
                sections_included = self._get_sections_list(eda_results)
            
            elif format == "markdown":
                report_path = self._generate_markdown_report(
                    eda_results=eda_results,
                    data_info=data_info,
                    output_path=output_path
                )
                sections_included = self._get_sections_list(eda_results)
            
            # ─── Gather Metadata
            file_size = report_path.stat().st_size if report_path.exists() else 0
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            
            # ─── Assemble Result
            result.data = {
                "report_path": str(report_path),
                "format": format,
                "generated_at": datetime.now().isoformat(),
                "file_size_bytes": int(file_size),
                "sections_included": sections_included,
                "telemetry": {
                    "elapsed_ms": elapsed_ms,
                    "pdf_conversion_attempted": pdf_conversion_attempted,
                    "pdf_conversion_success": pdf_conversion_success,
                    "weasyprint_available": WEASYPRINT_AVAILABLE,
                },
                "metadata": {
                    "title": "EDA Report — DataGenius PRO",
                    "generator": cfg.generator_name,
                    "version": cfg.generator_version,
                },
                "version": "5.0-kosmos-enterprise",
            }
            
            self._log.success(
                f"✓ Report generated | "
                f"format={format} | "
                f"path={report_path} | "
                f"size={file_size/1024:.1f}KB | "
                f"elapsed={elapsed_ms:.1f}ms"
            )
        
        except Exception as e:
            msg = f"Report generation failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # HTML Report Generation
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("html_generation")
    def _generate_html_report(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        output_path: Optional[Path]
    ) -> Path:
        """
        Generate HTML report with responsive design.
        
        Returns:
            Path to generated HTML file
        """
        cfg = self.config
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = cfg.default_reports_dir / f"eda_report_{timestamp}.html"
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML content
        html_content = self._create_html_template(eda_results, data_info)
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        self._log.info(f"HTML report saved: {output_path}")
        return output_path
    
    # ───────────────────────────────────────────────────────────────────
    # PDF Report Generation
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("pdf_generation")
    def _generate_pdf_report(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        output_path: Optional[Path]
    ) -> tuple[Path, bool]:
        """
        Generate PDF report with WeasyPrint (or fallback to HTML).
        
        Returns:
            Tuple of (report_path, conversion_success)
        """
        cfg = self.config
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = cfg.default_reports_dir / f"eda_report_{timestamp}.pdf"
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML content
        html_content = self._create_html_template(eda_results, data_info)
        
        # Attempt PDF conversion
        if WEASYPRINT_AVAILABLE:
            try:
                WeasyprintHTML(string=html_content).write_pdf(
                    output_path,
                    presentational_hints=True
                )
                self._log.info(f"PDF report saved: {output_path}")
                return output_path, True
            
            except Exception as e:
                self._log.warning(f"⚠ PDF conversion failed: {e}, falling back to HTML")
        
        # Fallback to HTML
        output_path = output_path.with_suffix('.html')
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        self._log.info(f"HTML fallback saved: {output_path}")
        return output_path, False
    
    # ───────────────────────────────────────────────────────────────────
    # Markdown Report Generation
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("markdown_generation")
    def _generate_markdown_report(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        output_path: Optional[Path]
    ) -> Path:
        """
        Generate Markdown report (portable, VCS-friendly).
        
        Returns:
            Path to generated Markdown file
        """
        cfg = self.config
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = cfg.default_reports_dir / f"eda_report_{timestamp}.md"
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate Markdown content
        md_content = self._create_markdown_template(eda_results, data_info)
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        self._log.info(f"Markdown report saved: {output_path}")
        return output_path
    
    # ───────────────────────────────────────────────────────────────────
    # HTML Template Builder
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("html_template", default_value="<html><body>Report generation failed</body></html>")
    def _create_html_template(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any]
    ) -> str:
        """
        Create comprehensive HTML report with modern styling.
        
        Returns:
            Complete HTML document as string
        """
        cfg = self.config
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="{cfg.generator_name} {cfg.generator_version}">
    <title>Raport EDA — DataGenius PRO Master Enterprise ++++</title>
    {self._get_html_styles()}
</head>
<body>
    <div class="container">
        {self._create_html_header(timestamp)}
        {self._create_html_overview(data_info)}
        {self._create_html_statistics(eda_results)}
        {self._create_html_missing_data(eda_results)}
        {self._create_html_outliers(eda_results)}
        {self._create_html_correlations(eda_results)}
        {self._create_html_summary(eda_results)}
        {self._create_html_footer()}
    </div>
</body>
</html>"""
        
        return html
    
    def _get_html_styles(self) -> str:
        """Generate CSS styles for HTML report."""
        cfg = self.config
        
        return f"""<style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-radius: 12px;
        }}
        
        .header {{
            border-bottom: 4px solid {cfg.primary_color};
            padding-bottom: 20px;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            color: {cfg.primary_color};
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.2rem;
            font-weight: 400;
        }}
        
        .header .meta {{
            color: #999;
            font-size: 0.9rem;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e1e8ed;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            color: {cfg.secondary_color};
            font-size: 1.8rem;
            margin-bottom: 25px;
            border-left: 5px solid {cfg.primary_color};
            padding-left: 20px;
            font-weight: 600;
        }}
        
        .section h3 {{
            color: #34495e;
            font-size: 1.4rem;
            margin: 25px 0 15px 0;
            font-weight: 600;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {cfg.gradient_start} 0%, {cfg.gradient_end} 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.35);
        }}
        
        .metric-card .label {{
            font-size: 0.95rem;
            opacity: 0.95;
            margin-bottom: 8px;
            font-weight: 500;
        }}
        
        .metric-card .value {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -1px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}
        
        table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e1e8ed;
        }}
        
        table tr:last-child td {{
            border-bottom: none;
        }}
        
        table tr:hover {{
            background: #f8f9fa;
        }}
        
        .alert {{
            padding: 18px 22px;
            border-radius: 10px;
            margin: 20px 0;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .alert::before {{
            font-size: 1.5rem;
        }}
        
        .alert-info {{
            background: #e7f3ff;
            border-left: 5px solid #0078d4;
            color: #004578;
        }}
        
        .alert-info::before {{ content: "ℹ️"; }}
        
        .alert-warning {{
            background: #fff4e5;
            border-left: 5px solid #ff8c00;
            color: #995200;
        }}
        
        .alert-warning::before {{ content: "⚠️"; }}
        
        .alert-success {{
            background: #e6f7e6;
            border-left: 5px solid #107c10;
            color: #0e5a0e;
        }}
        
        .alert-success::before {{ content: "✅"; }}
        
        .footer {{
            margin-top: 60px;
            padding-top: 30px;
            border-top: 3px solid #e1e8ed;
            text-align: center;
            color: #666;
            font-size: 0.9rem;
        }}
        
        .footer p {{
            margin: 5px 0;
        }}
        
        ul {{
            margin: 15px 0;
            padding-left: 25px;
        }}
        
        li {{
            margin: 8px 0;
            line-height: 1.7;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
            .metric-card {{
                break-inside: avoid;
            }}
            .section {{
                break-inside: avoid;
            }}
        }}
        
        @media (max-width: {cfg.responsive_breakpoint}) {{
            body {{
                padding: 10px;
            }}
            .container {{
                padding: 20px;
            }}
            .header h1 {{
                font-size: 2rem;
            }}
            .metrics {{
                grid-template-columns: 1fr;
            }}
            table {{
                font-size: 0.85rem;
            }}
            table th, table td {{
                padding: 10px;
            }}
        }}
    </style>"""
    
    @_safe_operation("html_header", default_value="<div class='header'><h1>Report</h1></div>")
    def _create_html_header(self, timestamp: str) -> str:
        """Create HTML header section."""
        cfg = self.config
        
        return f"""
        <div class="header">
            <h1>🚀 Raport Analizy Danych (EDA)</h1>
            <div class="subtitle">{cfg.generator_name}</div>
            <div class="meta">
                <strong>Wygenerowano:</strong> {timestamp} | 
                <strong>Wersja:</strong> {cfg.generator_version}
            </div>
        </div>
        """
    
    @_safe_operation("html_overview", default_value="")
    def _create_html_overview(self, data_info: Dict[str, Any]) -> str:
        """Create data overview section."""
        
        n_rows = data_info.get('n_rows', 0)
        n_columns = data_info.get('n_columns', 0)
        memory_mb = data_info.get('memory_mb', 0.0)
        
        return f"""
        <div class="section">
            <h2>📊 Przegląd Danych</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="label">Liczba wierszy</div>
                    <div class="value">{n_rows:,}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Liczba kolumn</div>
                    <div class="value">{n_columns}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Rozmiar pamięci</div>
                    <div class="value">{memory_mb:.2f} MB</div>
                </div>
            </div>
        </div>
        """
    
    @_safe_operation("html_statistics", default_value="")
    def _create_html_statistics(self, eda_results: Dict[str, Any]) -> str:
        """Create statistics section."""
        
        eda_data = eda_results.get("eda_results", {})
        stats = eda_data.get("StatisticalAnalyzer", {})
        
        if not stats:
            return ""
        
        overall = stats.get("overall", {})
        n_numeric = overall.get('n_numeric', 0)
        n_categorical = overall.get('n_categorical', 0)
        sparsity = overall.get('sparsity', 0.0)
        
        html = """
        <div class="section">
            <h2>📈 Statystyki Opisowe</h2>
        """
        
        if overall:
            html += f"""
            <h3>Podsumowanie typów cech</h3>
            <ul>
                <li><strong>Cechy numeryczne:</strong> {n_numeric}</li>
                <li><strong>Cechy kategoryczne:</strong> {n_categorical}</li>
                <li><strong>Sparsity (rzadkość):</strong> {sparsity:.2%}</li>
            </ul>
            """
        
        html += "</div>"
        return html
    
    @_safe_operation("html_missing", default_value="")
    def _create_html_missing_data(self, eda_results: Dict[str, Any]) -> str:
        """Create missing data section."""
        cfg = self.config
        
        eda_data = eda_results.get("eda_results", {})
        missing = eda_data.get("MissingDataAnalyzer", {})
        
        if not missing:
            return ""
        
        summary = missing.get("summary", {})
        total_missing = summary.get("total_missing", 0)
        
        if total_missing == 0:
            return """
            <div class="section">
                <h2>🔍 Brakujące Dane</h2>
                <div class="alert alert-success">
                    Brak brakujących danych w zbiorze — dane są kompletne!
                </div>
            </div>
            """
        
        missing_pct = summary.get("missing_percentage", 0.0)
        
        html = f"""
        <div class="section">
            <h2>🔍 Brakujące Dane</h2>
            <div class="alert alert-warning">
                Wykryto {total_missing:,} brakujących wartości ({missing_pct:.2f}% całego zbioru)
            </div>
        """
        
        # Table of columns with missing data
        columns_missing = missing.get("columns", [])
        if columns_missing:
            html += f"""
            <h3>Top {min(len(columns_missing), cfg.max_table_rows)} kolumn z brakami</h3>
            <table>
                <tr>
                    <th>Kolumna</th>
                    <th>Liczba braków</th>
                    <th>Procent</th>
                    <th>Severity</th>
                    <th>Sugerowana strategiaRetryMContinue </th>
                </tr>
            """
        for col in columns_missing[:cfg.max_table_rows]:
            severity = col.get('severity', 'unknown')
            severity_emoji = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }.get(severity, '⚪')
            
            html += f"""
            <tr>
                <td><strong>{col['column']}</strong></td>
                <td>{col['n_missing']:,}</td>
                <td>{col['missing_percentage']:.2f}%</td>
                <td>{severity_emoji} {severity}</td>
                <td>{col.get('suggested_strategy', 'N/A')}</td>
            </tr>
            """
        
        html += "</table>"
    
    # Recommendations
    recommendations = missing.get("recommendations", [])
    if recommendations:
        html += "<h3>Rekomendacje</h3><ul>"
        for rec in recommendations[:5]:
            html += f"<li>{rec}</li>"
        html += "</ul>"
    
    html += "</div>"
    return html

@_safe_operation("html_outliers", default_value="")
def _create_html_outliers(self, eda_results: Dict[str, Any]) -> str:
    """Create outliers section."""
    cfg = self.config
    
    eda_data = eda_results.get("eda_results", {})
    outliers = eda_data.get("OutlierDetector", {})
    
    if not outliers:
        return ""
    
    summary = outliers.get("summary", {})
    total_rows = summary.get("total_outliers_rows_union", 0)
    n_cols = summary.get("n_columns_with_outliers", 0)
    
    if total_rows == 0:
        return """
        <div class="section">
            <h2>⚠️ Outliers (Wartości odstające)</h2>
            <div class="alert alert-success">
                Brak wykrytych outlierów — dane są jednorodne!
            </div>
        </div>
        """
    
    by_method = summary.get("by_method", {})
    methods_used = summary.get("methods_used", [])
    
    html = f"""
    <div class="section">
        <h2>⚠️ Outliers (Wartości odstające)</h2>
        <div class="alert alert-warning">
            Wykryto outliers w {n_cols} kolumnach, 
            dotykające {total_rows:,} wierszy (unia wszystkich metod)
        </div>
        
        <h3>Wykrycie według metody</h3>
        <table>
            <tr>
                <th>Metoda</th>
                <th>Liczba wierszy z outlierami</th>
            </tr>
    """
    
    for method in methods_used:
        count = by_method.get(method, 0)
        html += f"""
            <tr>
                <td><strong>{method}</strong></td>
                <td>{count:,}</td>
            </tr>
        """
    
    html += "</table>"
    
    # Most affected column
    most = summary.get("most_outliers")
    if most:
        html += f"""
        <h3>Najbardziej dotknięta kolumna</h3>
        <div class="alert alert-info">
            <strong>{most['column']}</strong> zawiera {most['n_outliers']:,} outlierów
        </div>
        """
    
    # Recommendations
    recommendations = outliers.get("recommendations", [])
    if recommendations:
        html += "<h3>Rekomendacje</h3><ul>"
        for rec in recommendations[:5]:
            html += f"<li>{rec}</li>"
        html += "</ul>"
    
    html += "</div>"
    return html

@_safe_operation("html_correlations", default_value="")
def _create_html_correlations(self, eda_results: Dict[str, Any]) -> str:
    """Create correlations section."""
    cfg = self.config
    
    eda_data = eda_results.get("eda_results", {})
    corr = eda_data.get("CorrelationAnalyzer", {})
    
    if not corr:
        return ""
    
    high_corr = corr.get("high_correlations", [])
    
    html = """
    <div class="section">
        <h2>🔗 Korelacje Między Cechami</h2>
    """
    
    if not high_corr:
        html += """
        <div class="alert alert-success">
            Brak silnych korelacji między cechami (|r| > 0.8) — cechy są niezależne!
        </div>
        """
    else:
        html += f"""
        <div class="alert alert-warning">
            Wykryto {len(high_corr)} par silnie skorelowanych cech — 
            może to wskazywać na multikolinearność
        </div>
        
        <h3>Top {min(len(high_corr), cfg.max_correlation_pairs)} silnych korelacji</h3>
        <table>
            <tr>
                <th>Cecha 1</th>
                <th>Cecha 2</th>
                <th>Korelacja (r)</th>
                <th>Siła</th>
            </tr>
        """
        
        for pair in high_corr[:cfg.max_correlation_pairs]:
            r = pair['correlation']
            abs_r = abs(r)
            
            # Determine strength
            if abs_r >= 0.9:
                strength = "🔴 Bardzo silna"
            elif abs_r >= 0.8:
                strength = "🟠 Silna"
            else:
                strength = "🟡 Umiarkowana"
            
            html += f"""
            <tr>
                <td><strong>{pair['feature1']}</strong></td>
                <td><strong>{pair['feature2']}</strong></td>
                <td>{r:.3f}</td>
                <td>{strength}</td>
            </tr>
            """
        
        html += "</table>"
        
        # Recommendations
        recommendations = corr.get("recommendations", [])
        if recommendations:
            html += "<h3>Rekomendacje</h3><ul>"
            for rec in recommendations[:5]:
                html += f"<li>{rec}</li>"
            html += "</ul>"
    
    html += "</div>"
    return html

@_safe_operation("html_summary", default_value="")
def _create_html_summary(self, eda_results: Dict[str, Any]) -> str:
    """Create executive summary section."""
    
    summary = eda_results.get("summary", {})
    
    if not summary:
        return ""
    
    html = """
    <div class="section">
        <h2>📋 Podsumowanie Wykonawcze</h2>
    """
    
    # Data quality
    quality = summary.get("data_quality", "unknown")
    quality_colors = {
        "excellent": ("🟢", "#107c10"),
        "good": ("🟡", "#ff8c00"),
        "fair": ("🟠", "#d83b01"),
        "poor": ("🔴", "#a80000")
    }
    emoji, color = quality_colors.get(quality, ("⚪", "#666"))
    
    html += f"""
    <div style="background: {color}15; border-left: 5px solid {color}; 
                padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3 style="color: {color}; margin-bottom: 10px;">
            {emoji} Ocena jakości danych: {quality.upper()}
        </h3>
    """
    
    severity = summary.get("severity_score", 0.0)
    html += f"""
        <p><strong>Wskaźnik severity:</strong> {severity:.2f} / 1.00</p>
    </div>
    """
    
    # Key findings
    findings = summary.get("key_findings", [])
    if findings:
        html += "<h3>🔍 Kluczowe Odkrycia</h3><ul>"
        for finding in findings:
            html += f"<li>{finding}</li>"
        html += "</ul>"
    
    # Recommendations
    recommendations = summary.get("recommendations", [])
    if recommendations:
        html += "<h3>💡 Rekomendacje</h3><ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
    
    html += "</div>"
    return html

@_safe_operation("html_footer", default_value="<div class='footer'><p>DataGenius PRO</p></div>")
def _create_html_footer(self) -> str:
    """Create HTML footer."""
    cfg = self.config
    
    return f"""
    <div class="footer">
        <p><strong>Raport wygenerowany przez {cfg.generator_name}</strong></p>
        <p>Wersja {cfg.generator_version}</p>
        <p>Built with ❤️ for Enterprise ML Teams</p>
    </div>
    """

# ───────────────────────────────────────────────────────────────────
# Markdown Template Builder
# ───────────────────────────────────────────────────────────────────

@_safe_operation("markdown_template", default_value="# Report\n\nGeneration failed")
def _create_markdown_template(
    self,
    eda_results: Dict[str, Any],
    data_info: Dict[str, Any]
) -> str:
    """
    Create Markdown report (portable, VCS-friendly).
    
    Returns:
        Complete Markdown document as string
    """
    cfg = self.config
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = f"""# 🚀 Raport Analizy Danych (EDA)
{cfg.generator_name}

Wygenerowano: {timestamp}
Wersja: {cfg.generator_version}

📊 Przegląd Danych

Liczba wierszy: {data_info.get('n_rows', 0):,}
Liczba kolumn: {data_info.get('n_columns', 0)}
Rozmiar pamięci: {data_info.get('memory_mb', 0):.2f} MB


📈 Statystyki Opisowe
{self._create_markdown_statistics(eda_results)}

🔍 Brakujące Dane
{self._create_markdown_missing(eda_results)}

⚠️ Outliers (Wartości odstające)
{self._create_markdown_outliers(eda_results)}

🔗 Korelacje Między Cechami
{self._create_markdown_correlations(eda_results)}

📋 Podsumowanie Wykonawcze
{self._create_markdown_summary(eda_results)}

Raport wygenerowany przez {cfg.generator_name} v{cfg.generator_version}
"""
return md
@_safe_operation("markdown_statistics", default_value="Brak danych statystycznych")
def _create_markdown_statistics(self, eda_results: Dict[str, Any]) -> str:
    """Create statistics section for Markdown."""
    
    eda_data = eda_results.get("eda_results", {})
    stats = eda_data.get("StatisticalAnalyzer", {})
    
    if not stats:
        return "Brak danych statystycznych"
    
    overall = stats.get("overall", {})
    n_numeric = overall.get('n_numeric', 0)
    n_categorical = overall.get('n_categorical', 0)
    sparsity = overall.get('sparsity', 0.0)
    
    return f"""

Cechy numeryczne: {n_numeric}
Cechy kategoryczne: {n_categorical}
Sparsity (rzadkość): {sparsity:.2%}
"""
@_safe_operation("markdown_missing", default_value="Brak analizy braków")
def _create_markdown_missing(self, eda_results: Dict[str, Any]) -> str:
    """Create missing data section for Markdown."""
    eda_data = eda_results.get("eda_results", {})
    missing = eda_data.get("MissingDataAnalyzer", {})
    
    if not missing:
        return "Brak analizy braków"
    
    summary = missing.get("summary", {})
    total_missing = summary.get("total_missing", 0)
    
    if total_missing == 0:
        return "✅ **Brak brakujących danych** — dane są kompletne!"
    
    missing_pct = summary.get("missing_percentage", 0.0)
    n_cols = summary.get("n_columns_with_missing", 0)
    
    md = f"""


⚠️ Wykryto {total_missing:,} brakujących wartości ({missing_pct:.2f}%) w {n_cols} kolumnach
Top kolumny z brakami
"""
    columns_missing = missing.get("columns", [])
    if columns_missing:
        for col in columns_missing[:5]:
            md += f"- **{col['column']}**: {col['n_missing']:,} ({col['missing_percentage']:.2f}%) — *{col.get('suggested_strategy', 'N/A')}*\n"
    
    return md

@_safe_operation("markdown_outliers", default_value="Brak analizy outlierów")
def _create_markdown_outliers(self, eda_results: Dict[str, Any]) -> str:
    """Create outliers section for Markdown."""
    
    eda_data = eda_results.get("eda_results", {})
    outliers = eda_data.get("OutlierDetector", {})
    
    if not outliers:
        return "Brak analizy outlierów"
    
    summary = outliers.get("summary", {})
    total_rows = summary.get("total_outliers_rows_union", 0)
    n_cols = summary.get("n_columns_with_outliers", 0)
    
    if total_rows == 0:
        return "✅ **Brak wykrytych outlierów** — dane są jednorodne!"
    
    md = f"""
⚠️ Wykryto outliers w {n_cols} kolumnach, dotykające {total_rows:,} wierszy
Wykrycie według metody
"""
    by_method = summary.get("by_method", {})
    for method, count in by_method.items():
        if count > 0:
            md += f"- **{method}**: {count:,} wierszy\n"
    
    return md

@_safe_operation("markdown_correlations", default_value="Brak analizy korelacji")
def _create_markdown_correlations(self, eda_results: Dict[str, Any]) -> str:
    """Create correlations section for Markdown."""
    cfg = self.config
    
    eda_data = eda_results.get("eda_results", {})
    corr = eda_data.get("CorrelationAnalyzer", {})
    
    if not corr:
        return "Brak analizy korelacji"
    
    high_corr = corr.get("high_correlations", [])
    
    if not high_corr:
        return "✅ **Brak silnych korelacji** między cechami — cechy są niezależne!"
    
    md = f"""
⚠️ Wykryto {len(high_corr)} par silnie skorelowanych cech
Top silne korelacje
"""
    for pair in high_corr[:cfg.max_correlation_pairs]:
        r = pair['correlation']
        md += f"- **{pair['feature1']}** ↔ **{pair['feature2']}**: r = {r:.3f}\n"
    
    return md

@_safe_operation("markdown_summary", default_value="Brak podsumowania")
def _create_markdown_summary(self, eda_results: Dict[str, Any]) -> str:
    """Create summary section for Markdown."""
    
    summary = eda_results.get("summary", {})
    
    if not summary:
        return "Brak podsumowania"
    
    md = ""
    
    # Data quality
    quality = summary.get("data_quality", "unknown")
    severity = summary.get("severity_score", 0.0)
    
    quality_emoji = {
        "excellent": "🟢",
        "good": "🟡",
        "fair": "🟠",
        "poor": "🔴"
    }.get(quality, "⚪")
    
    md += f"""
{quality_emoji} Ocena jakości danych: {quality.upper()}
Wskaźnik severity: {severity:.2f} / 1.00
"""
    # Key findings
    findings = summary.get("key_findings", [])
    if findings:
        md += "### 🔍 Kluczowe Odkrycia\n\n"
        for finding in findings:
            md += f"- {finding}\n"
        md += "\n"
    
    # Recommendations
    recommendations = summary.get("recommendations", [])
    if recommendations:
        md += "### 💡 Rekomendacje\n\n"
        for rec in recommendations:
            md += f"- {rec}\n"
    
    return md

# ───────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────

def _get_sections_list(self, eda_results: Dict[str, Any]) -> List[str]:
    """
    Extract list of included sections from EDA results.
    
    Returns:
        List of section names present in report
    """
    sections: List[str] = ["overview"]
    
    eda_data = eda_results.get("eda_results", {})
    
    if "StatisticalAnalyzer" in eda_data:
        sections.append("statistics")
    
    if "MissingDataAnalyzer" in eda_data:
        sections.append("missing_data")
    
    if "OutlierDetector" in eda_data:
        sections.append("outliers")
    
    if "CorrelationAnalyzer" in eda_data:
        sections.append("correlations")
    
    if "summary" in eda_results:
        sections.append("summary")
    
    return sections

# ───────────────────────────────────────────────────────────────────
# Empty Payload (Fallback)
# ───────────────────────────────────────────────────────────────────

@staticmethod
def _empty_payload() -> Dict[str, Any]:
    """Generate empty payload for failed generation."""
    
    return {
        "report_path": "",
        "format": "unknown",
        "generated_at": datetime.now().isoformat(),
        "file_size_bytes": 0,
        "sections_included": [],
        "telemetry": {
            "elapsed_ms": 0.0,
            "pdf_conversion_attempted": False,
            "pdf_conversion_success": False,
            "weasyprint_available": WEASYPRINT_AVAILABLE,
        },
        "metadata": {
            "title": "EDA Report",
            "generator": "DataGenius PRO Master Enterprise ++++",
            "version": "5.0-kosmos-enterprise",
        },
        "version": "5.0-kosmos-enterprise",
    }

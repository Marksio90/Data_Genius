"""
DataGenius PRO - Report Generator
Generates comprehensive EDA reports in multiple formats
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from config.settings import settings


class ReportGenerator(BaseAgent):
    """
    Generates comprehensive EDA reports in PDF, HTML, or Markdown format
    """
    
    def __init__(self):
        super().__init__(
            name="ReportGenerator",
            description="Generates EDA reports in multiple formats"
        )
    
    def execute(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        format: str = "html",
        output_path: Optional[Path] = None,
        **kwargs
    ) -> AgentResult:
        """
        Generate EDA report
        
        Args:
            eda_results: Results from EDA Orchestrator
            data_info: Basic data information
            format: Report format (html, pdf, markdown)
            output_path: Output file path (optional)
        
        Returns:
            AgentResult with report path
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Generate report based on format
            if format == "html":
                report_path = self._generate_html_report(
                    eda_results, data_info, output_path
                )
            elif format == "pdf":
                report_path = self._generate_pdf_report(
                    eda_results, data_info, output_path
                )
            elif format == "markdown":
                report_path = self._generate_markdown_report(
                    eda_results, data_info, output_path
                )
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            result.data = {
                "report_path": str(report_path),
                "format": format,
                "generated_at": datetime.now().isoformat(),
            }
            
            self.logger.success(f"Report generated: {report_path}")
        
        except Exception as e:
            result.add_error(f"Report generation failed: {e}")
            self.logger.error(f"Report generation error: {e}", exc_info=True)
        
        return result
    
    def _generate_html_report(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        output_path: Optional[Path]
    ) -> Path:
        """Generate HTML report"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.REPORTS_PATH / f"eda_report_{timestamp}.html"
        
        # Create HTML content
        html_content = self._create_html_template(eda_results, data_info)
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_pdf_report(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        output_path: Optional[Path]
    ) -> Path:
        """Generate PDF report"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.REPORTS_PATH / f"eda_report_{timestamp}.pdf"
        
        # First generate HTML, then convert to PDF
        html_content = self._create_html_template(eda_results, data_info)
        
        # Convert HTML to PDF using weasyprint
        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(output_path)
        except ImportError:
            self.logger.warning("weasyprint not available, falling back to HTML")
            output_path = output_path.with_suffix('.html')
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        
        return output_path
    
    def _generate_markdown_report(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        output_path: Optional[Path]
    ) -> Path:
        """Generate Markdown report"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.REPORTS_PATH / f"eda_report_{timestamp}.md"
        
        # Create Markdown content
        md_content = self._create_markdown_template(eda_results, data_info)
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        return output_path
    
    def _create_html_template(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any]
    ) -> str:
        """Create HTML report template"""
        
        html = f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raport EDA - DataGenius PRO</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        
        .header {{
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #1f77b4;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.2rem;
        }}
        
        .header .meta {{
            color: #999;
            font-size: 0.9rem;
            margin-top: 10px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #2c3e50;
            font-size: 1.8rem;
            margin-bottom: 20px;
            border-left: 4px solid #1f77b4;
            padding-left: 15px;
        }}
        
        .section h3 {{
            color: #34495e;
            font-size: 1.4rem;
            margin: 20px 0 10px 0;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .metric-card .label {{
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .metric-card .value {{
            font-size: 2rem;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        table th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            font-weight: 600;
        }}
        
        table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        table tr:hover {{
            background: #f8f9fa;
        }}
        
        .alert {{
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        
        .alert-info {{
            background: #d1ecf1;
            border-left: 4px solid #0c5460;
            color: #0c5460;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #856404;
            color: #856404;
        }}
        
        .alert-success {{
            background: #d4edda;
            border-left: 4px solid #155724;
            color: #155724;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #dee2e6;
            text-align: center;
            color: #666;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Raport Analizy Danych (EDA)</h1>
            <div class="subtitle">DataGenius PRO - Next-Gen Auto Data Scientist</div>
            <div class="meta">
                Wygenerowano: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </div>
        
        {self._create_data_overview_section(data_info)}
        {self._create_statistics_section(eda_results)}
        {self._create_missing_data_section(eda_results)}
        {self._create_outliers_section(eda_results)}
        {self._create_correlations_section(eda_results)}
        {self._create_summary_section(eda_results)}
        
        <div class="footer">
            <p>Raport wygenerowany przez DataGenius PRO v2.0</p>
            <p>Built with ❤️ by DataGenius Team</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _create_data_overview_section(self, data_info: Dict[str, Any]) -> str:
        """Create data overview section"""
        
        return f"""
        <div class="section">
            <h2>📊 Przegląd Danych</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="label">Liczba wierszy</div>
                    <div class="value">{data_info.get('n_rows', 0):,}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Liczba kolumn</div>
                    <div class="value">{data_info.get('n_columns', 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Rozmiar (MB)</div>
                    <div class="value">{data_info.get('memory_mb', 0):.2f}</div>
                </div>
            </div>
        </div>
        """
    
    def _create_statistics_section(self, eda_results: Dict[str, Any]) -> str:
        """Create statistics section"""
        
        if "StatisticalAnalyzer" not in eda_results.get("eda_results", {}):
            return ""
        
        stats = eda_results["eda_results"]["StatisticalAnalyzer"]
        
        html = """
        <div class="section">
            <h2>📈 Statystyki</h2>
        """
        
        # Overall stats
        if "overall" in stats:
            overall = stats["overall"]
            html += f"""
            <h3>Statystyki ogólne</h3>
            <ul>
                <li>Cechy numeryczne: {overall.get('n_numeric', 0)}</li>
                <li>Cechy kategoryczne: {overall.get('n_categorical', 0)}</li>
                <li>Sparsity: {overall.get('sparsity', 0):.2%}</li>
            </ul>
            """
        
        html += "</div>"
        return html
    
    def _create_missing_data_section(self, eda_results: Dict[str, Any]) -> str:
        """Create missing data section"""
        
        if "MissingDataAnalyzer" not in eda_results.get("eda_results", {}):
            return ""
        
        missing = eda_results["eda_results"]["MissingDataAnalyzer"]
        summary = missing.get("summary", {})
        
        if summary.get("total_missing", 0) == 0:
            return """
            <div class="section">
                <h2>🔍 Brakujące Dane</h2>
                <div class="alert alert-success">
                    ✅ Brak brakujących danych w zbiorze!
                </div>
            </div>
            """
        
        html = f"""
        <div class="section">
            <h2>🔍 Brakujące Dane</h2>
            <div class="alert alert-warning">
                ⚠️ Znaleziono {summary.get('total_missing', 0):,} brakujących wartości 
                ({summary.get('missing_percentage', 0):.2f}%)
            </div>
        """
        
        # Table of columns with missing data
        columns_missing = missing.get("columns", [])
        if columns_missing:
            html += """
            <h3>Kolumny z brakującymi danymi</h3>
            <table>
                <tr>
                    <th>Kolumna</th>
                    <th>Liczba braków</th>
                    <th>Procent</th>
                    <th>Strategia</th>
                </tr>
            """
            
            for col in columns_missing[:10]:  # Top 10
                html += f"""
                <tr>
                    <td>{col['column']}</td>
                    <td>{col['n_missing']}</td>
                    <td>{col['missing_percentage']:.2f}%</td>
                    <td>{col['suggested_strategy']}</td>
                </tr>
                """
            
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _create_outliers_section(self, eda_results: Dict[str, Any]) -> str:
        """Create outliers section"""
        
        if "OutlierDetector" not in eda_results.get("eda_results", {}):
            return ""
        
        outliers = eda_results["eda_results"]["OutlierDetector"]
        summary = outliers.get("summary", {})
        
        html = f"""
        <div class="section">
            <h2>⚠️ Outliers</h2>
            <div class="alert alert-info">
                Wykryto {summary.get('total_outliers', 0)} outliers 
                w {summary.get('n_columns_with_outliers', 0)} kolumnach
            </div>
        </div>
        """
        
        return html
    
    def _create_correlations_section(self, eda_results: Dict[str, Any]) -> str:
        """Create correlations section"""
        
        if "CorrelationAnalyzer" not in eda_results.get("eda_results", {}):
            return ""
        
        corr = eda_results["eda_results"]["CorrelationAnalyzer"]
        high_corr = corr.get("high_correlations", [])
        
        html = """
        <div class="section">
            <h2>🔗 Korelacje</h2>
        """
        
        if high_corr:
            html += f"""
            <div class="alert alert-warning">
                Znaleziono {len(high_corr)} par silnie skorelowanych cech (|r| > 0.8)
            </div>
            <h3>Silne korelacje</h3>
            <table>
                <tr>
                    <th>Cecha 1</th>
                    <th>Cecha 2</th>
                    <th>Korelacja</th>
                </tr>
            """
            
            for pair in high_corr[:10]:  # Top 10
                html += f"""
                <tr>
                    <td>{pair['feature1']}</td>
                    <td>{pair['feature2']}</td>
                    <td>{pair['correlation']:.3f}</td>
                </tr>
                """
            
            html += "</table>"
        else:
            html += """
            <div class="alert alert-success">
                ✅ Brak silnych korelacji między cechami
            </div>
            """
        
        html += "</div>"
        return html
    
    def _create_summary_section(self, eda_results: Dict[str, Any]) -> str:
        """Create summary section"""
        
        summary = eda_results.get("summary", {})
        
        html = """
        <div class="section">
            <h2>📋 Podsumowanie</h2>
        """
        
        # Key findings
        findings = summary.get("key_findings", [])
        if findings:
            html += "<h3>Kluczowe Odkrycia</h3><ul>"
            for finding in findings:
                html += f"<li>{finding}</li>"
            html += "</ul>"
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            html += "<h3>Rekomendacje</h3><ul>"
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        html += "</div>"
        return html
    
    def _create_markdown_template(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any]
    ) -> str:
        """Create Markdown report template"""
        
        md = f"""# 🚀 Raport Analizy Danych (EDA)

**DataGenius PRO - Next-Gen Auto Data Scientist**

Wygenerowano: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📊 Przegląd Danych

- **Liczba wierszy**: {data_info.get('n_rows', 0):,}
- **Liczba kolumn**: {data_info.get('n_columns', 0)}
- **Rozmiar**: {data_info.get('memory_mb', 0):.2f} MB

---

## 📈 Statystyki

{self._create_markdown_statistics(eda_results)}

---

## 🔍 Brakujące Dane

{self._create_markdown_missing(eda_results)}

---

## ⚠️ Outliers

{self._create_markdown_outliers(eda_results)}

---

## 🔗 Korelacje

{self._create_markdown_correlations(eda_results)}

---

## 📋 Podsumowanie

{self._create_markdown_summary(eda_results)}

---

*Raport wygenerowany przez DataGenius PRO v2.0*
"""
        return md
    
    def _create_markdown_statistics(self, eda_results: Dict) -> str:
        """Create statistics markdown"""
        if "StatisticalAnalyzer" not in eda_results.get("eda_results", {}):
            return "Brak danych statystycznych"
        
        stats = eda_results["eda_results"]["StatisticalAnalyzer"]
        overall = stats.get("overall", {})
        
        return f"""
- Cechy numeryczne: {overall.get('n_numeric', 0)}
- Cechy kategoryczne: {overall.get('n_categorical', 0)}
- Sparsity: {overall.get('sparsity', 0):.2%}
"""
    
    def _create_markdown_missing(self, eda_results: Dict) -> str:
        """Create missing data markdown"""
        if "MissingDataAnalyzer" not in eda_results.get("eda_results", {}):
            return "Brak analizy braków"
        
        missing = eda_results["eda_results"]["MissingDataAnalyzer"]
        summary = missing.get("summary", {})
        
        if summary.get("total_missing", 0) == 0:
            return "✅ Brak brakujących danych!"
        
        return f"""
⚠️ Znaleziono {summary.get('total_missing', 0):,} brakujących wartości ({summary.get('missing_percentage', 0):.2f}%)
"""
    
    def _create_markdown_outliers(self, eda_results: Dict) -> str:
        """Create outliers markdown"""
        if "OutlierDetector" not in eda_results.get("eda_results", {}):
            return "Brak analizy outliers"
        
        outliers = eda_results["eda_results"]["OutlierDetector"]
        summary = outliers.get("summary", {})
        
        return f"""
Wykryto {summary.get('total_outliers', 0)} outliers w {summary.get('n_columns_with_outliers', 0)} kolumnach
"""
    
    def _create_markdown_correlations(self, eda_results: Dict) -> str:
        """Create correlations markdown"""
        if "CorrelationAnalyzer" not in eda_results.get("eda_results", {}):
            return "Brak analizy korelacji"
        
        corr = eda_results["eda_results"]["CorrelationAnalyzer"]
        high_corr = corr.get("high_correlations", [])
        
        if not high_corr:
            return "✅ Brak silnych korelacji"
        
        md = f"Znaleziono {len(high_corr)} par silnie skorelowanych cech:\n\n"
        for pair in high_corr[:5]:
            md += f"- {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}\n"
        
        return md
    
    def _create_markdown_summary(self, eda_results: Dict) -> str:
        """Create summary markdown"""
        summary = eda_results.get("summary", {})
        
        md = ""
        
        # Key findings
        findings = summary.get("key_findings", [])
        if findings:
            md += "### Kluczowe Odkrycia\n\n"
            for finding in findings:
                md += f"- {finding}\n"
            md += "\n"
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            md += "### Rekomendacje\n\n"
            for rec in recommendations:
                md += f"- {rec}\n"
        
        return md
"""Serwis do generowania raportów PDF"""
import io
import base64
from typing import Dict, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def generate_pdf_report(
    report_text: str,
    charts: Dict[str, str],
    business_domain: str,
    target_column: str
) -> bytes:
    """
    Generuje raport PDF z tekstem i wykresami
    
    Args:
        report_text: Tekst raportu w Markdown
        charts: Słownik z wykresami jako base64 strings
        business_domain: Domena biznesowa
        target_column: Kolumna docelowa
    
    Returns:
        bytes: PDF jako bytes
    """
    buffer = io.BytesIO()
    
    # Utwórz dokument PDF
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # Style
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#2c3e50',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor='#34495e',
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Tytuł
    story.append(Paragraph(f"Raport Analityczny: {business_domain}", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Info o kolumnie docelowej
    story.append(Paragraph(f"<b>Kolumna docelowa:</b> {target_column}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Konwertuj markdown do PDF paragraphs (uproszczona wersja)
    sections = report_text.split('\n## ')
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.split('\n')
        if lines:
            # Pierwszy wiersz jako nagłówek
            header = lines[0].replace('#', '').strip()
            story.append(Paragraph(header, heading_style))
            
            # Reszta jako treść
            for line in lines[1:]:
                if line.strip():
                    # Usuń markdown formatting
                    clean_line = line.replace('**', '').replace('*', '').strip()
                    if clean_line:
                        story.append(Paragraph(clean_line, styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))
    
    # Dodaj wykresy jeśli są dostępne
    if charts:
        story.append(PageBreak())
        story.append(Paragraph("Wizualizacje", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        for chart_name, chart_data in charts.items():
            if chart_data:
                try:
                    # Dekoduj base64 do obrazka
                    img_data = base64.b64decode(chart_data)
                    img_buffer = io.BytesIO(img_data)
                    
                    # Dodaj obrazek
                    img = Image(img_buffer, width=6*inch, height=4*inch)
                    story.append(Paragraph(chart_name.replace('_', ' ').title(), heading_style))
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))
                except Exception as e:
                    print(f"Błąd dodawania wykresu {chart_name}: {e}")
    
    # Zbuduj PDF
    doc.build(story)
    
    # Pobierz bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
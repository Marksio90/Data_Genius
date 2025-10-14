"""
DataGenius PRO - Custom CSS
Custom styling for Streamlit application
"""

import streamlit as st


# Custom CSS styles
CUSTOM_CSS = """
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    
    /* Success/Error messages */
    .element-container .stAlert {
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Sidebar content */
    [data-testid="stSidebar"] .stMarkdown {
        padding: 0.5rem 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background-color: #f8f9ff;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Code blocks */
    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        transform: translateY(-4px);
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Footer */
    footer {
        visibility: hidden;
    }
    
    footer:after {
        content: 'DataGenius PRO v2.0 | Built with ❤️';
        visibility: visible;
        display: block;
        text-align: center;
        padding: 1rem;
        color: #666;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #667eea;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        h1 {
            font-size: 1.5rem;
        }
        
        h2 {
            font-size: 1.25rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0 4px;
    }
    
    .badge-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .badge-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .badge-danger {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .badge-info {
        background-color: #d1ecf1;
        color: #0c5460;
    }
</style>
"""


def load_custom_css():
    """Load custom CSS into Streamlit app"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "", icon: str = ""):
    """
    Render custom metric card
    
    Args:
        title: Card title
        value: Main value
        subtitle: Subtitle text
        icon: Icon emoji
    """
    
    st.markdown(f"""
    <div class="metric-card animate-fade-in">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 0.875rem; color: #666; font-weight: 600; margin-bottom: 0.5rem;">
            {title}
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem;">
            {value}
        </div>
        <div style="font-size: 0.875rem; color: #999;">
            {subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, type: str = "info"):
    """
    Render badge
    
    Args:
        text: Badge text
        type: Badge type (success, warning, danger, info)
    """
    
    st.markdown(f"""
    <span class="badge badge-{type}">{text}</span>
    """, unsafe_allow_html=True)


def render_info_box(content: str, type: str = "info"):
    """
    Render styled info box
    
    Args:
        content: Box content
        type: Box type (info, warning, success, error)
    """
    
    st.markdown(f"""
    <div class="{type}-box animate-fade-in">
        {content}
    </div>
    """, unsafe_allow_html=True)
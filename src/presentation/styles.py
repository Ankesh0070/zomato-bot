"""
Custom CSS styles for the Streamlit UI.

Provides premium, modern styling for the Zomato AI Recommender
with dark-mode-friendly design, glassmorphism cards, and micro-animations.
"""


def get_custom_css() -> str:
    """Return the full custom CSS for the Streamlit app."""
    return """
    <style>
    /* ===== Google Font Import ===== */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    /* ===== Global Styles ===== */
    .stApp {
        font-family: 'Outfit', system-ui, sans-serif;
    }

    /* ===== Header ===== */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.2rem 0;
        margin-bottom: 1.2rem;
    }

    .app-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 40%, #a0a5be 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .app-header p {
        color: #6b7194;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ===== Sidebar Styling ===== */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e8384a;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    .sidebar-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(203, 32, 45, 0.3), transparent);
        margin: 1rem 0;
    }

    /* ===== Summary Card ===== */
    .summary-card {
        background: linear-gradient(135deg, rgba(16, 18, 27, 0.9) 0%, rgba(20, 15, 25, 0.9) 100%);
        border: 1px solid rgba(203, 32, 45, 0.15);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #cb202d;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    .summary-card .summary-icon {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #e8384a;
        font-weight: 700;
    }

    .summary-card .summary-text {
        color: #c4c8d8;
        font-size: 1rem;
        line-height: 1.7;
        font-weight: 400;
    }

    .summary-card .summary-meta {
        color: #6b7194;
        font-size: 0.8rem;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* ===== Recommendation Card ===== */
    .rec-card {
        background: rgba(16, 18, 27, 0.55);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }

    .rec-card:hover {
        border-color: rgba(203, 32, 45, 0.35);
        transform: translateY(-3px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.3);
    }

    /* ===== Recommendation Header ===== */
    .rec-header-container {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }

    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 800;
        margin-right: 14px;
        flex-shrink: 0;
        background: rgba(203, 32, 45, 0.12);
        border: 1px solid rgba(203, 32, 45, 0.3);
        color: #e8384a;
    }
    
    .rank-1 { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.4); color: #f59e0b; box-shadow: 0 0 15px rgba(245, 158, 11, 0.15); }
    .rank-2 { background: rgba(192, 192, 192, 0.08); border-color: rgba(192, 192, 192, 0.3); color: #c4c8d8; }
    .rank-3 { background: rgba(205, 127, 50, 0.08); border-color: rgba(205, 127, 50, 0.3); color: #f0a050; }

    .rec-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f3f9;
        margin: 0;
        letter-spacing: -0.3px;
    }

    /* ===== Metrics Styling ===== */
    .metric-badge {
        background: rgba(255,255,255,0.025);
        padding: 10px 14px;
        border-radius: 10px;
        font-weight: 600;
        color: #c4c8d8;
        display: flex;
        align-items: center;
        gap: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .rating-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.15);
    }

    /* ===== Stats Row ===== */
    .stats-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }

    .stat-chip {
        background: rgba(203, 32, 45, 0.06);
        border: 1px solid rgba(203, 32, 45, 0.15);
        border-radius: 9999px;
        padding: 0.4rem 1rem;
        font-size: 0.82rem;
        color: #c4c8d8;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .stat-chip .stat-value {
        color: #e8384a;
        font-weight: 700;
    }

    /* ===== Empty State ===== */
    .empty-state {
        text-align: center;
        padding: 4rem 1.5rem;
        background: rgba(255,255,255,0.01);
        border-radius: 20px;
        border: 1px dashed rgba(255,255,255,0.06);
    }

    .empty-state h3 {
        color: #e8384a;
        font-weight: 700;
        font-size: 1.4rem;
    }

    /* ===== Footer ===== */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: #4a4f6a;
        font-size: 0.82rem;
    }

    /* ===== Streamlit Component Overrides ===== */
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.02) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #e8384a !important;
        border: 1px solid rgba(203, 32, 45, 0.15) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #cb202d 0%, #a11823 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 6px 20px rgba(203, 32, 45, 0.25) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #e8384a 0%, #cb202d 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(203, 32, 45, 0.4) !important;
    }
    </style>
    """

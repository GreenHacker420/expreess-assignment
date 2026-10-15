"""Theme CSS and styling tokens for Documentation Assistant UI."""

PALETTE = {
    "primary_dark": "#086288",   # Deep Oceanic Blue
    "steel_azure": "#4D8DBA",    # Slate Steel Blue
    "sky_soft": "#97D7FF",       # Soft Sky Blue
    "warm_cream": "#FFEFC4",     # Warm Soft Cream
    "surface": "#FFFFFF",        # White surfaces
    "canvas_bg": "#F5F9FC",      # Canvas background
}

CUSTOM_CSS = """
<style>
/* Clean Soft Oceanic Canvas */
.stApp {
    background-color: #F5F9FC !important;
    background-image: none !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #0E2A38 !important;
}

.main .block-container {
    max-width: 1200px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
}

/* Header Container */
.top-header-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(8, 98, 136, 0.05);
}

.top-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.top-brand-icon {
    width: 46px;
    height: 46px;
    background: #EBF5FB;
    border: 1px solid #97D7FF;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.top-brand-title {
    font-size: 20px;
    font-weight: 700;
    color: #086288;
    margin: 0;
    line-height: 1.25;
}

.top-brand-subtitle {
    font-size: 13px;
    color: #4D8DBA;
    margin: 3px 0 0 0;
}

.header-badges {
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-pill-blue {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #EBF5FB;
    border: 1px solid #97D7FF;
    color: #086288;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 600;
}

.status-pill-warm {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #FFEFC4;
    border: 1px solid #F5DC9A;
    color: #614605;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #086288;
}

.status-dot-green {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2E8B57;
}

/* =========================================================================
   Sleek Modern Tab Bar (Fixed: Zero Clashing Blocks, Crisp Modern Underline)
   ========================================================================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: transparent !important;
    border-bottom: 2px solid #D8ECF8 !important;
    padding: 0 4px !important;
    margin-bottom: 24px !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 18px !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    color: #4D8DBA !important;
    transition: color 0.15s ease, background-color 0.15s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #086288 !important;
    background-color: #EBF5FB !important;
}

/* Active tab text */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #086288 !important;
    font-weight: 700 !important;
    background-color: transparent !important;
}

/* Crisp Active Indicator Bar */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #086288 !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0 !important;
}

/* Metric Badges Row */
.metric-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 16px 0;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 600;
}

.pill-type {
    background: #EBF5FB;
    border: 1px solid #97D7FF;
    color: #086288;
}

.pill-retries {
    background: #FFEFC4;
    border: 1px solid #F3DC9B;
    color: #614605;
}

.pill-latency {
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    color: #4D8DBA;
}

.pill-status {
    background: #EBF7EE;
    border: 1px solid #BCE5C8;
    color: #1B6338;
}

/* Citations Box */
.citation-box {
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    border-left: 4px solid #086288;
    border-radius: 8px;
    padding: 12px 18px;
    margin-top: 10px;
    font-size: 13px;
    color: #0E2A38;
    box-shadow: 0 1px 4px rgba(8, 98, 136, 0.03);
}

.citation-tag {
    display: inline-block;
    background: #FFEFC4;
    color: #614605;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11.5px;
    margin-right: 8px;
    border: 1px solid #F3DC9B;
}

/* Architecture Flow Nodes */
.flow-node {
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(8, 98, 136, 0.04);
    height: 100%;
}

.flow-node-title {
    font-weight: 700;
    color: #086288;
    font-size: 14.5px;
    margin-bottom: 6px;
}

.flow-node-desc {
    font-size: 12px;
    color: #4D8DBA;
    line-height: 1.5;
}

/* Buttons */
.stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    transition: all 0.15s ease-in-out !important;
}

.stButton button[kind="primary"] {
    background-color: #086288 !important;
    color: #FFFFFF !important;
    border: 1px solid #086288 !important;
}

.stButton button[kind="primary"]:hover {
    background-color: #4D8DBA !important;
    border-color: #4D8DBA !important;
    color: #FFFFFF !important;
}

/* Text area */
.stTextArea textarea {
    border: 1px solid #D8ECF8 !important;
    border-radius: 10px !important;
    background-color: #FFFFFF !important;
    color: #0E2A38 !important;
    font-size: 14px !important;
}

.stTextArea textarea:focus {
    border-color: #086288 !important;
    box-shadow: 0 0 0 2px rgba(8, 98, 136, 0.15) !important;
}

.section-title-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
</style>
"""

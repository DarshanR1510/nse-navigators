from enum import Enum

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

css = """
/* -------- BACKGROUND -------- */
body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) fixed !important;
    background-size: 400% 400% !important;
    animation: gradient 15s ease infinite !important;
    color: #ddd !important;
    font-family: 'gt-planar', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    position: relative !important;
}

body::before {
    content: "" !important;
    position: fixed !important;
    inset: 0 !important;
    background: radial-gradient(
        circle at top center,
        rgba(128, 0, 255, 0.1),
        transparent 60%
    ) !important;
    z-index: 0 !important;
    pointer-events: none !important;
}

/* -------- CONTAINER DEPTH -------- */
#root, .gr-block, .trader-card, .value-card {
    position: relative !important;
    z-index: 1 !important;
}

/* -------- GRADIENT ANIMATION -------- */
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* -------- CARD STYLES -------- */
.gr-block, .trader-card, .value-card {
    background: rgba(24, 26, 27, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
}

.gr-block:hover, .trader-card:hover {
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5) !important;
    transform: translateY(-4px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

/* -------- SPINNER / PULSE -------- */
.spinner {
    animation: rotate 1s linear infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.4; }
    100% { opacity: 1; }
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* -------- HEADERS / TYPOGRAPHY -------- */
.card-header {
    background: rgba(33,33,33,0.5) 100% !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    padding: 12px 16px !important;
    border-radius: 12px 12px 0 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 14px !important;
    color: #ccc !important;
    font-family: 'gt-planar', sans-serif !important;
}

h1, h2, h3, h4, h5 {
    color: #aaf !important;
    font-family: 'gt-planar', sans-serif !important;
}

/* -------- FONTS: GT PLANAR -------- */
@font-face {
    font-family: 'gt-planar';
    src: url('static/fonts/gt-planar-black.otf') format('opentype');
    font-weight: 900;
    font-style: normal;
    font-display: swap;
}

@font-face {
    font-family: 'gt-planar';
    src: url('static/fonts/gt-planar-bold.otf') format('opentype');
    font-weight: 700;
    font-style: normal;
    font-display: swap;
}

@font-face {
    font-family: 'gt-planar';
    src: url('static/fonts/gt-planar-medium.otf') format('opentype');
    font-weight: 500;
    font-style: normal;
    font-display: swap;
}

@font-face {
    font-family: 'gt-planar';
    src: url('static/fonts/gt-planar-regular.otf') format('opentype');
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}

@font-face {
    font-family: 'gt-planar';
    src: url('static/fonts/gt-planar-light.otf') format('opentype');
    font-weight: 300;
    font-style: normal;
    font-display: swap;
}

@font-face {
    font-family: 'gt-planar';
    src: url('static/fonts/gt-planar-thin.otf') format('opentype');
    font-weight: 100;
    font-style: normal;
    font-display: swap;
}

/* Force gt-planar globally */
.gr-block, .trader-card, .value-card, .card-header,
.gr-column, .gr-group, .gr-html, h1, h2, h3, h4, h5,
p, div {
    font-family: gt-planar, sans-serif !important;
    font-synthesis: none !important;
}

/* Log area styling */
span {
    font-family: 'Courier New', monospace !important;
}

/* Transaction history should keep its monospace font */
.dataframe-fix, .dataframe-fix * {
    font-family: 'Courier New', monospace !important;
}

/* -------- SCROLLBARS -------- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #4b6cb7, #182848);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #5a7bc7, #283858);
}

/* -------- PNL COLORS -------- */
.positive-pnl {
    color: #00ff99 !important;
    font-weight: bold;
}
.positive-bg {
    background-color: #00aa77 !important;
    font-weight: bold;
}
.negative-bg {
    background-color: #aa0033 !important;
    font-weight: bold;
}
.negative-pnl {
    color: #ff3366 !important;
    font-weight: bold;
}

/* -------- TABLE FIXES -------- */
.dataframe-fix .table-wrap {
    background: rgba(30, 30, 30, 0.6) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}

/* -------- GRID SPACING -------- */
.gr-row {
    margin-top: 12px !important;
    margin-bottom: 12px !important;
}

/* -------- HIDE FOOTER -------- */
footer {
    display: none !important;
}

/* -------- ENHANCED CONTROL BUTTON STYLES -------- */
.control-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 16px 32px !important;
    font-size: 18px !important;
    font-family: 'gt-planar', sans-serif !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    margin: 8px !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    min-width: 240px !important;
    height: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.control-button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
    transition: left 0.5s !important;
}

.control-button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 12px 48px rgba(102, 126, 234, 0.4) !important;
    background: linear-gradient(135deg, #7c8df4 0%, #8a5db8 100%) !important;
}

.control-button:hover::before {
    left: 100% !important;
}

.control-button:active {
    transform: translateY(-1px) scale(0.98) !important;
    transition: all 0.1s !important;
}

/* Start Button Specific Styles */
.control-button:not(.stop-button) {
    background: linear-gradient(135deg, #00dbde 0%, #fc00ff 50%, #00dbde 100%) !important;
    background-size: 300% 300% !important;
    animation: gradientShift 3s ease infinite !important;
    box-shadow: 0 8px 32px rgba(0, 219, 222, 0.3) !important;
}

.control-button:not(.stop-button):hover {
    background: linear-gradient(135deg, #1ae5e8 0%, #ff1aff 50%, #1ae5e8 100%) !important;
    background-size: 300% 300% !important;
    box-shadow: 0 12px 48px rgba(0, 219, 222, 0.5) !important;
    animation: gradientShift 1.5s ease infinite !important;
}

/* Stop Button Specific Styles */  
.stop-button {
    background: linear-gradient(135deg, #ff416c 0%, #ff1744 50%, #ff416c 100%) !important;
    background-size: 300% 300% !important;
    animation: gradientShift 2s ease infinite !important;
    box-shadow: 0 8px 32px rgba(255, 65, 108, 0.3) !important;
}

.stop-button:hover {
    background: linear-gradient(135deg, #ff5177 0%, #ff2955 50%, #ff5177 100%) !important;
    background-size: 300% 300% !important;
    box-shadow: 0 12px 48px rgba(255, 65, 108, 0.5) !important;
    animation: gradientShift 1s ease infinite !important;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* -------- STATUS MESSAGES -------- */
.status-message {
    padding: 16px 24px !important;
    border-radius: 12px !important;
    margin: 12px 0 !important;
    font-family: 'gt-planar', sans-serif !important;
    font-weight: 500 !important;
    font-size: 16px !important;
    text-align: center !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s ease !important;
}

.status-message.success {
    background: rgba(76, 175, 80, 0.2) !important;
    color: #81c784 !important;
    border-color: rgba(76, 175, 80, 0.3) !important;
    box-shadow: 0 4px 20px rgba(76, 175, 80, 0.15) !important;
}

.status-message.error {
    background: rgba(244, 67, 54, 0.2) !important;
    color: #ef5350 !important;
    border-color: rgba(244, 67, 54, 0.3) !important;
    box-shadow: 0 4px 20px rgba(244, 67, 54, 0.15) !important;
}

.status-message.info {
    background: rgba(33, 150, 243, 0.2) !important;
    color: #64b5f6 !important;
    border-color: rgba(33, 150, 243, 0.3) !important;
    box-shadow: 0 4px 20px rgba(33, 150, 243, 0.15) !important;
}

/* -------- ENHANCED AGENT STATUS -------- */
.agent-status {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.3s ease !important;
}

.agent-status:hover {
    background: rgba(0, 0, 0, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* -------- RESPONSIVE DESIGN -------- */
@media (max-width: 768px) {
    .control-button {
        min-width: 200px !important;
        height: 50px !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
    }
    
    .status-message {
        font-size: 14px !important;
        padding: 12px 16px !important;
    }
}

/* -------- CONTROL PANEL STYLING -------- */
.gr-group {
    background: rgba(24, 26, 27, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
}

/* -------- ENHANCED HOVER EFFECTS -------- */
.gr-group:hover {
    border-color: rgba(255, 255, 255, 0.25) !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3) !important;
}
"""

js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

class Color(Enum):
    RED = "#ff3366"
    GREEN = "#00ff99"
    YELLOW = "#ffff66"
    BLUE = "#66ccff"
    MAGENTA = "#cc66ff"
    CYAN = "#66ffff"
    WHITE = "#cccccc"
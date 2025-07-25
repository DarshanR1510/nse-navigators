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
/* Base styles */
body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) fixed !important;
    background-size: 400% 400% !important;
    animation: gradient 15s ease infinite !important;    
    color: #ddd !important;
    position: relative !important;
}

/* Add the overlay gradient */
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

/* Ensure content stays above the gradient */
#root, .gr-block, .trader-card, .value-card {
    position: relative !important;
    z-index: 1 !important;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glass morphism cards */
.gr-block, .trader-card, .value-card {
    background: rgba(24, 26, 27, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
}

.gr-block:hover {
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4) !important;
    transform: translateY(-2px) !important;
}

.spinner {
    animation: rotate 1s linear infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.4; }
    100% { opacity: 1; }
}

/* Card headers */
.card-header {
    background: rgba(33,33,33,0.5) 100% !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    padding: 12px 16px !important;
    border-radius: 12px 12px 0 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 14px !important;
    color: #ccc !important;
}

/* Data tables */
.dataframe-fix .table-wrap {
    background: rgba(30, 30, 30, 0.6) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}

.gr-block:hover, .trader-card:hover {
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5) !important;
    transform: translateY(-4px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

/* Scrollbars */
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

/* Typography */
h1, h2, h3, h4, h5 {
    color: #aaf !important;
}

/* PNL colors */
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

/* Log area styling */
span {
    font-family: 'Courier New', monospace !important;
}

/* Layout spacing */
.gr-row {
    margin-top: 12px !important;
    margin-bottom: 12px !important;
}

/* Hide default footer */
footer {
    display: none !important;
}

@font-face {
    font-family: 'GT Planar';
    src: url('/utils/gt_planar_font/GT-Planar-Black.otf') format('opentype');
    font-weight: 900;
    font-style: normal;
}
@font-face {
    font-family: 'GT Planar';
    src: url('/utils/gt_planar_font/GT-Planar-Bold.otf') format('opentype');
    font-weight: 700;
    font-style: normal;
}
@font-face {
    font-family: 'GT Planar';
    src: url('/utils/gt_planar_font/GT-Planar-Medium.otf') format('opentype');
    font-weight: 500;
    font-style: normal;
}
@font-face {
    font-family: 'GT Planar';
    src: url('/utils/gt_planar_font/GT-Planar-Regular.otf') format('opentype');
    font-weight: 400;
    font-style: normal;
}
@font-face {
    font-family: 'GT Planar';
    src: url('/utils/gt_planar_font/GT-Planar-Light.otf') format('opentype');
    font-weight: 300;
    font-style: normal;
}
@font-face {
    font-family: 'GT Planar';
    src: url('/utils/gt_planar_font/GT-Planar-Thin.otf') format('opentype');
    font-weight: 100;
    font-style: normal;
}

/* Apply GT Planar everywhere except transactions history */
body, html, .gr-block, .trader-card, .value-card, .card-header, .gr-column, .gr-group, .gr-html {
    font-family: 'GT Planar', 'Inter', sans-serif !important;
}

/* Exclude transactions history section */
.dataframe-fix, .dataframe-fix * {
    font-family: 'Inter', 'Courier New', monospace !important;
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
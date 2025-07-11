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
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
    font-family: 'Inter', sans-serif;
    color: #ddd !important;
}

.gr-block {
    background: rgba(24, 26, 27, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
}

.gr-row {
    margin-top: 12px !important;
    margin-bottom: 12px !important;
}

h1, h2, h3, h4, h5 {
    color: #aaf !important;
}

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

.dataframe-fix-small .table-wrap,
.dataframe-fix .table-wrap {
    background: rgba(30, 30, 30, 0.8) !important;
    border-radius: 8px !important;
    color: #ccc !important;
}

.dataframe-fix-small .table-wrap {
    min-height: 150px;
    max-height: 150px;
}

.dataframe-fix .table-wrap {
    min-height: 200px;
    max-height: 200px;
}

footer {
    display: none !important;
}

/* Log area styling */
span {
    font-family: 'Courier New', monospace !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(20, 20, 20, 0.7);
}
::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
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

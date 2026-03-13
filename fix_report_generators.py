import re, os

files = [
    r'c:\Users\DELL\Downloads\dfg\merge\src\ecg\ecg_report_generator.py',
    r'c:\Users\DELL\Downloads\dfg\merge\src\ecg\4_3_ecg_report_generator.py',
    r'c:\Users\DELL\Downloads\dfg\merge\src\ecg\6_2_ecg_report_generator.py',
]

CANVAS_IMPORT = (
    "from matplotlib.figure import Figure as _Figure\n"
    "from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA"
)

for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Neutralize ANY non-comment matplotlib.use('Agg') line
    def comment_use(m):
        line = m.group(0)
        if line.lstrip().startswith('#'):
            return line
        return '# [SAFE] ' + line.lstrip()
    content = re.sub(r'[^\n]*matplotlib\.use\([^)]+\)[^\n]*', comment_use, content)

    # 2. Replace plt.subplots(figsize=(...), facecolor='...', frameon=True)
    def fix_subplots(m):
        indent = m.group(1)
        wh     = m.group(2)
        fc     = m.group(3)
        ci = "\n".join(indent + l for l in CANVAS_IMPORT.splitlines())
        return (
            f"{ci}\n"
            f"{indent}fig = _Figure(figsize=({wh}), facecolor='{fc}')\n"
            f"{indent}_FCA(fig)\n"
            f"{indent}ax = fig.add_subplot(111)"
        )
    content = re.sub(
        r'([ \t]*)fig, ax = plt\.subplots\(figsize=\(([^)]+)\),\s*facecolor=\'([^\']+)\',\s*frameon=True\)',
        fix_subplots, content
    )

    # 3. Replace fig = plt.figure(...)
    def fix_figure(m):
        indent = m.group(1)
        args   = m.group(2)
        ci = "\n".join(indent + l for l in CANVAS_IMPORT.splitlines())
        return (
            f"{ci}\n"
            f"{indent}fig = _Figure({args})\n"
            f"{indent}_FCA(fig)"
        )
    content = re.sub(
        r'([ \t]*)fig = plt\.figure\(([^)]+)\)',
        fix_figure, content
    )

    # 4. plt.close(fig) -> del fig  (safe in any thread)
    content = content.replace('plt.close(fig)', 'del fig')

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)

    # Report result
    lines = content.splitlines()
    bad_use  = [i+1 for i, l in enumerate(lines)
                if 'matplotlib.use(' in l and not l.strip().startswith('#')]
    bad_plt  = [i+1 for i, l in enumerate(lines)
                if re.search(r'\bplt\.(figure|subplots|close)\b', l) and not l.strip().startswith('#')]
    print(f"{os.path.basename(fpath)}: remaining matplotlib.use={bad_use}, plt.figure/subplots/close={bad_plt}")

print("Done.")

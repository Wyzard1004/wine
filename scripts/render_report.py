"""Render the committed Markdown paper as a standalone, offline HTML article.

Install the optional dependency with ``pip install -r requirements-docs.txt``.
Run from any working directory: ``python3 scripts/render_report.py``.
The dashboard and checked-in HTML do not require this dependency at runtime.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import re
import sys

try:
    import markdown
except ModuleNotFoundError:
    raise SystemExit(
        "Install the optional report renderer: "
        "python3 -m pip install -r requirements-docs.txt"
    )


ROOT = Path(__file__).resolve().parents[1]

STYLE = """
:root{--paper:#f7f6f2;--ink:#292b29;--muted:#626d5b;--wine:#78384b;--border:#e8e7e0;--serif:Georgia,'Times New Roman',serif;--sans:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:30px}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased}a{color:var(--wine);text-underline-offset:3px}a:hover{text-decoration-thickness:2px}a:focus-visible,button:focus-visible,[tabindex]:focus-visible{outline:3px solid #b77588;outline-offset:5px}.skip-link{position:absolute;top:-60px;left:12px;background:#fff;padding:10px;z-index:5}.skip-link:focus{top:12px}.site-header{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:24px max(5vw,20px);border-bottom:1px solid var(--border);background:#ffffffa0}.brand{display:flex;align-items:center;gap:12px;text-decoration:none;font-family:var(--serif);font-size:24px;letter-spacing:-.5px}.brand img{width:34px;height:34px}.brand span{color:var(--muted);font-size:17px;letter-spacing:0}.header-actions{display:flex;gap:18px;align-items:center;font-size:12px}.header-actions a{color:var(--ink);text-decoration:none}.header-actions a:hover{text-decoration:underline}.print-button{font:inherit;border:1px solid #d7d7cc;color:var(--ink);border-radius:6px;background:transparent;padding:9px 13px;cursor:pointer}.print-button:hover{background:#eeede7}.layout{display:grid;grid-template-columns:205px minmax(0,880px);gap:64px;max-width:1260px;margin:0 auto;padding:58px 34px 90px}.contents{align-self:start;position:sticky;top:30px;max-height:calc(100vh - 60px);overflow:auto;padding-top:9px}.contents-label,.eyebrow{font-size:10px;font-weight:650;letter-spacing:1.6px;text-transform:uppercase;color:var(--muted)}.contents-label{margin:0 0 18px}.toc ul{list-style:none;padding:0;margin:0}.toc li{margin:0 0 12px}.toc li ul{padding:10px 0 0 12px;border-left:1px solid var(--border);margin:10px 0 17px}.toc li ul li{margin-bottom:9px}.toc a{display:block;color:#626d5b;font-size:11px;line-height:1.6;text-decoration:none}.toc a:hover{color:var(--wine)}main{min-width:0}.eyebrow{display:flex;align-items:center;gap:9px;margin:0 0 18px}.eyebrow:before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--wine)}article{font-family:var(--serif);font-size:17px;line-height:1.85}h1,h2,h3{font-weight:400;line-height:1.2;scroll-margin-top:24px}h1{font-size:57px;letter-spacing:-2px;margin:0 0 24px;max-width:820px}h2{font-size:31px;letter-spacing:-.55px;margin:54px 0 19px;padding-top:21px;border-top:1px solid var(--border)}h3{font-size:22px;letter-spacing:-.3px;margin:31px 0 14px}p{margin:0 0 20px}article>p:first-of-type{font-family:var(--sans);font-size:13px;line-height:1.9;color:var(--muted);margin:0 0 38px}strong{font-weight:650}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:#efeee8;border-radius:3px;padding:2px 5px;font-size:.8em;overflow-wrap:anywhere}pre{overflow:auto;background:#efeee8;border-radius:7px;padding:18px;font-size:14px;line-height:1.7}pre code{padding:0;background:none}blockquote{margin:25px 0;padding:7px 24px;border-left:3px solid var(--wine);color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:7px;background:#fff;margin:27px 0 29px}table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:11px;line-height:1.65;font-variant-numeric:tabular-nums}th{padding:15px 16px;text-align:left;font-size:10px;font-weight:600;color:#626d5b;background:#f0f1eb;white-space:nowrap}td{padding:13px 16px;border-top:1px solid #efefe9;color:#636c5c;vertical-align:top}tbody tr:hover{background:#fafaf6}td:first-child{color:#3e4836;font-weight:550}figure{margin:34px 0 37px;background:#fff;border:1px solid var(--border);border-radius:8px;padding:20px}figure img{display:block;width:100%;height:auto}figcaption{font-family:var(--sans);font-size:11px;line-height:1.8;color:var(--muted);padding:14px 2px 0}article img{max-width:100%;height:auto}article ol,article ul{padding-left:25px}article li{margin-bottom:12px}article h2:last-of-type~ol{font-family:var(--sans);font-size:12px;line-height:1.9;overflow-wrap:anywhere}.article-footer{border-top:1px solid var(--border);padding-top:24px;margin-top:44px;display:flex;justify-content:space-between;align-items:center;gap:20px;font-size:12px}.article-footer a{text-decoration:none}.site-footer{border-top:1px solid var(--border);padding:24px max(5vw,20px);display:flex;justify-content:space-between;gap:20px;font-size:10px;color:var(--muted)}
@media(max-width:1000px){.layout{grid-template-columns:170px minmax(0,1fr);gap:32px;padding:44px 26px 65px}h1{font-size:47px}.brand span{display:none}}
@media(max-width:760px){.layout{display:block;padding:34px 22px 55px}.contents{position:static;max-height:none;padding:0 0 20px;margin-bottom:31px;border-bottom:1px solid var(--border)}.contents-label{margin-bottom:12px}.toc>ul{display:flex;flex-wrap:wrap;column-gap:18px;row-gap:6px}.toc>ul>li{margin:0}.toc li ul{display:none}.toc a{font-size:10px}.header-actions{gap:12px;font-size:10px}.site-header{padding:19px 22px}.brand{font-size:22px;gap:9px}.brand img{width:29px;height:29px}h1{font-size:43px;letter-spacing:-1.3px}h2{font-size:28px;margin-top:38px}h3{font-size:21px}article{font-size:16px;line-height:1.85}figure{padding:10px;margin:27px 0}figcaption{font-size:10px;padding:10px 3px 3px}th,td{padding:11px}.site-footer{font-size:9px;padding:22px;flex-wrap:wrap}}
@media(max-width:420px){.site-header{gap:12px}.header-actions>a{display:none}.print-button{font-size:10px}h1{font-size:37px}.layout{padding:29px 18px 45px}.article-footer{align-items:flex-start;flex-direction:column;gap:10px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@page{margin:18mm}
@media print{html{scroll-behavior:auto}body{background:#fff;color:#000}.site-header,.contents,.eyebrow,.site-footer,.article-footer,.skip-link{display:none!important}.layout{display:block;max-width:none;padding:0;margin:0}article{font-size:10.5pt;line-height:1.6}h1{font-size:29pt;letter-spacing:-.7px;margin-bottom:14pt}h2{font-size:18pt;margin-top:22pt;padding-top:10pt;break-after:avoid}h3{font-size:13pt;break-after:avoid}p{orphans:3;widows:3;margin-bottom:11pt}article>p:first-of-type{font-size:9pt;margin-bottom:20pt}a{color:inherit}figure{break-inside:avoid;border:0;padding:0;margin:18pt 0}figure img{max-height:175mm;object-fit:contain}figcaption{font-size:8pt;padding-top:6pt}.table-wrap{overflow:visible;break-inside:avoid;border:0;margin:15pt 0}table{font-size:8pt}th{font-size:7.5pt}th,td{padding:7pt 6pt}th{background:#f0f1eb;print-color-adjust:exact}article h2:last-of-type~ol{font-size:8pt}code{font-size:8pt}pre{white-space:pre-wrap}}
""".strip()


def render(source: Path) -> str:
    """Render Markdown using the documented parser and a deterministic wrapper."""
    converter = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3", "permalink": False}},
        output_format="html",
    )
    content = converter.convert(source.read_text(encoding="utf-8"))
    title_match = re.search(r"<h1(?:\s[^>]*)?>(.*?)</h1>", content, flags=re.S)
    title = re.sub(r"<[^>]+>", "", title_match.group(1)) if title_match else "Wine research report"
    title = html.escape(html.unescape(title))

    # Improve the parser's valid HTML with responsive table regions and figures.
    content = content.replace(
        "<table>",
        '<div class="table-wrap" role="region" aria-label="Scrollable report table" tabindex="0"><table>',
    ).replace("</table>", "</table></div>")
    content = re.sub(r"<th(?=[\s>])", '<th scope="col"', content)
    content = re.sub(
        r"<p>(<img\b[^>]*>)</p>\s*<p><em>(Figure .*?)</em></p>",
        r"<figure>\1<figcaption>\2</figcaption></figure>",
        content,
        flags=re.S,
    )
    content = content.replace("<img ", '<img decoding="sync" ')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A reproducible study of principal component analysis and k-nearest neighbours on the UCI Wine dataset, with matched validation, results, and figures.">
  <meta name="generator" content="scripts/render_report.py; Python-Markdown {markdown.__version__}">
  <meta name="theme-color" content="#78384b">
  <title>{title} — Wine research</title>
  <link rel="icon" href="../dashboard/favicon.svg" type="image/svg+xml">
  <style>{STYLE}</style>
</head>
<body>
  <a class="skip-link" href="#report">Skip to report</a>
  <header class="site-header">
    <a class="brand" href="../dashboard/" aria-label="Wine research dashboard"><img src="../dashboard/favicon.svg" alt="" width="34" height="34">Wine <span>/ Research report</span></a>
    <nav class="header-actions" aria-label="Report actions">
      <a href="../dashboard/">← Dashboard</a>
      <a href="report.md" download>Source Markdown ↓</a>
      <button class="print-button" type="button" onclick="window.print()">Print / save PDF</button>
    </nav>
  </header>
  <div class="layout">
    <nav class="contents" aria-label="Report contents">
      <p class="contents-label">In this paper</p>
      {converter.toc}
    </nav>
    <main id="report">
      <p class="eyebrow">Wine research · The full study</p>
      <article aria-label="Research paper">
        {content}
      </article>
      <div class="article-footer">
        <a href="../dashboard/">← Explore the interactive dashboard</a>
        <a href="report.md" download>Download the source Markdown ↓</a>
      </div>
    </main>
  </div>
  <footer class="site-footer">
    <span>Wine / Research report</span>
    <span>Data: UCI Wine · CC BY 4.0 · Generated from report.md</span>
  </footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that committed HTML matches its Markdown source.")
    args = parser.parse_args()
    source = ROOT / "docs" / "report.md"
    output = ROOT / "docs" / "report.html"
    rendered = render(source)
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            print("Report HTML is stale; run python3 scripts/render_report.py", file=sys.stderr)
            raise SystemExit(1)
        print("Report HTML matches docs/report.md")
    else:
        output.write_text(rendered, encoding="utf-8")
        print(f"Rendered {output.relative_to(ROOT)} from {source.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

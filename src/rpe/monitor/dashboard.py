"""Static recovery dashboard: warehouse -> dashboards/recovery.html.

Dependency-free (hand-rolled SVG): recovery trend with band thresholds,
recent-14-day component table, model card. Regenerate after each dbt build.

Usage: uv run python -m rpe.monitor.dashboard
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WAREHOUSE = PROJECT_ROOT / "warehouse.duckdb"
OUT = PROJECT_ROOT / "dashboards" / "recovery.html"

W, H, PAD = 920, 280, 36
BAND_COLOR = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}


def main() -> None:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    df = con.execute(
        """select date_day, recovery_score, recovery_band,
                  hrv_component, rhr_component, sleep_component, resp_component
           from mart_recovery order by date_day"""
    ).df()
    n_models = con.execute("select count(*) from mart_daily").fetchone()[0]
    con.close()
    df = df.dropna(subset=["recovery_score"]).reset_index(drop=True)
    n = len(df)
    if n == 0:
        sys.exit("no labeled days in mart_recovery")

    def x(i: int) -> float:
        return PAD + i / max(n - 1, 1) * (W - 2 * PAD)

    def y(v: float) -> float:
        return H - PAD - v / 100 * (H - 2 * PAD)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(df["recovery_score"]))
    dots = "\n".join(
        f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.2" fill="{BAND_COLOR[b]}">'
        f"<title>{d.date()} — {v:.0f} ({b})</title></circle>"
        for i, (d, v, b) in enumerate(
            zip(df["date_day"], df["recovery_score"], df["recovery_band"]))
    )
    rows = "\n".join(
        "<tr><td>{d}</td><td><b style='color:{c}'>{s:.0f}</b></td>"
        "<td>{h:.0f}</td><td>{r:.0f}</td><td>{sl:.0f}</td><td>{rp:.0f}</td></tr>".format(
            d=d.date(), c=BAND_COLOR[b], s=v, h=h or 0, r=r or 0, sl=sl or 0, rp=rp or 0)
        for d, v, b, h, r, sl, rp in zip(
            df["date_day"].iloc[-14:], df["recovery_score"].iloc[-14:],
            df["recovery_band"].iloc[-14:], df["hrv_component"].iloc[-14:],
            df["rhr_component"].iloc[-14:], df["sleep_component"].iloc[-14:],
            df["resp_component"].iloc[-14:])
    )
    dist = df["recovery_band"].value_counts()
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Recovery dashboard</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;color:#111}}
.cards{{display:flex;gap:12px;margin:16px 0}}
.card{{flex:1;border:1px solid #e5e5e5;border-radius:10px;padding:12px 16px}}
.card .v{{font-size:28px;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin-top:12px;font-size:14px}}
th,td{{border-bottom:1px solid #eee;padding:6px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.muted{{color:#666;font-size:13px}}
</style></head><body>
<h1>Recovery dashboard</h1>
<p class="muted">{n} labeled days · {df['date_day'].iloc[0].date()} → {df['date_day'].iloc[-1].date()}
· mart_daily rows: {n_models} · generated from warehouse.duckdb</p>
<div class="cards">
<div class="card"><div class="v" style="color:#22c55e">{dist.get('green',0)}</div>green days</div>
<div class="card"><div class="v" style="color:#eab308">{dist.get('yellow',0)}</div>yellow days</div>
<div class="card"><div class="v" style="color:#ef4444">{dist.get('red',0)}</div>red days</div>
<div class="card"><div class="v">{df['recovery_score'].mean():.0f}</div>mean recovery</div>
</div>
<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="recovery trend">
<line x1="{PAD}" y1="{y(67):.1f}" x2="{W-PAD}" y2="{y(67):.1f}" stroke="#22c55e" stroke-dasharray="6 4" opacity="0.6"/>
<line x1="{PAD}" y1="{y(34):.1f}" x2="{W-PAD}" y2="{y(34):.1f}" stroke="#ef4444" stroke-dasharray="6 4" opacity="0.6"/>
<polyline points="{pts}" fill="none" stroke="#333" stroke-width="1.6"/>
{dots}
</svg>
<h2>Last 14 days</h2>
<table><tr><th>date</th><th>recovery</th><th>hrv</th><th>rhr</th><th>sleep</th><th>resp</th></tr>
{rows}</table>
<p class="muted">Bands: ≥67 green · 34–66 yellow · &lt;34 red. Components are 0–100 sub-scores
(HRV / resting HR / sleep / respiratory rate vs personal baselines).</p>
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

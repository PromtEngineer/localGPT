"""Generate the visual-native corpus: facts that live in chart structure and
table layout. Half the facts are LABELED (recoverable by perfect OCR of the
image), half are STRUCTURAL (no value labels — answering requires comparing
visual elements). The script is the ground truth: it writes facts.json from
the same variables that drive the plots.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "vizdocs"
os.makedirs(f"{OUT}/pages", exist_ok=True)
FACTS = {}

# ---------------------------------------------------------------- doc 1: quarterly business review (charts)
rev = {"Q1": 3.1, "Q2": 4.2, "Q3": 4.7, "Q4": 3.9}        # $M, labeled
churn = {"Jan": 5.2, "Feb": 4.8, "Mar": 4.1, "Apr": 3.6, "May": 3.2, "Jun": 2.9}  # %, UNLABELED (structure: steady decline, lowest Jun)
segments = {"Enterprise": 46, "SMB": 31, "Consumer": 23}   # %, labeled

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Meridian Analytics — FY2025 Quarterly Business Review", fontsize=15, y=1.02)
ax = axes[0]
bars = ax.bar(rev.keys(), rev.values(), color=["#7aa6c2", "#7aa6c2", "#c25e4c", "#7aa6c2"])
ax.set_title("Revenue by Quarter ($M)")
for b, v in zip(bars, rev.values()):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"${v}M", ha="center")
ax = axes[1]
ax.plot(list(churn.keys()), list(churn.values()), marker="o", color="#4c72b0")
ax.set_title("Monthly Churn Rate H1-2025 (%)")   # deliberately NO value labels
ax.grid(alpha=0.3)
ax = axes[2]
ax.pie(segments.values(), labels=[f"{k}\n{v}%" for k, v in segments.items()],
       colors=["#4c72b0", "#dd8452", "#55a868"])
ax.set_title("Revenue Mix by Segment")
fig.savefig(f"{OUT}/pages/vz_quarterly_p00.png", dpi=140, bbox_inches="tight")
plt.close(fig)
FACTS["vz_quarterly"] = {
    "labeled": {"peak_quarter": "Q3", "peak_revenue_musd": 4.7, "enterprise_share_pct": 46,
                "q1_revenue_musd": 3.1},
    "structural": {"churn_trend": "declined every month from January through June",
                   "churn_lowest_month": "Jun", "highlighted_bar": "Q3 (colored differently)"}}

# ---------------------------------------------------------------- doc 2: incident heatmap + SLA table
regions = ["US-East", "US-West", "EU-Central", "APAC"]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
rng = np.random.RandomState(7)
inc = rng.randint(0, 6, size=(4, 6))
inc[2, 3] = 14   # EU-Central April spike — THE structural fact
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle("Meridian Analytics — Platform Operations Review H1-2025", fontsize=15, y=1.02)
ax = axes[0]
im = ax.imshow(inc, cmap="Reds")
ax.set_xticks(range(6), months); ax.set_yticks(range(4), regions)
ax.set_title("Sev-1 Incidents by Region and Month")   # NO cell annotations
fig.colorbar(im, ax=ax, shrink=0.8)
ax = axes[1]
ax.axis("off")
sla = [["Tier", "Uptime SLA", "Response", "Penalty"],
       ["Platinum", "99.95%", "15 min", "25% credit"],
       ["Gold", "99.9%", "1 hour", "10% credit"],
       ["Silver", "99.5%", "4 hours", "5% credit"]]
t = ax.table(cellText=sla, loc="center", cellLoc="center")
t.scale(1, 2.2); t.auto_set_font_size(False); t.set_fontsize(11)
for j in range(4):
    t[0, j].set_facecolor("#dbe4ee"); t[0, j].set_text_props(weight="bold")
ax.set_title("Support Tier SLA Matrix")
fig.savefig(f"{OUT}/pages/vz_ops_p00.png", dpi=140, bbox_inches="tight")
plt.close(fig)
FACTS["vz_ops"] = {
    "labeled": {"platinum_uptime": "99.95%", "gold_response": "1 hour", "silver_penalty": "5% credit"},
    "structural": {"incident_hotspot": "EU-Central in April (worst cell in the heatmap)",
                   "hotspot_value": int(inc[2, 3])}}

# ---------------------------------------------------------------- doc 3: filled form (checkboxes)
fig, ax = plt.subplots(figsize=(10, 12))
ax.axis("off")
ax.text(0.5, 0.97, "CHANGE REQUEST FORM — CR-2025-0114", ha="center", fontsize=15, weight="bold")
lines = [
    ("Requested by:", "Dana Whitfield, Platform Engineering"),
    ("Date submitted:", "May 22, 2025"),
    ("System affected:", "Billing ingestion pipeline"),
    ("Requested window:", "June 7, 2025, 02:00-06:00 UTC"),
]
y = 0.88
for label, val in lines:
    ax.text(0.08, y, label, fontsize=12, weight="bold")
    ax.text(0.38, y, val, fontsize=12, family="cursive")
    y -= 0.06
ax.text(0.08, y - 0.02, "Risk classification:", fontsize=12, weight="bold")
opts = [("Low", False), ("Medium", True), ("High", False)]   # Medium CHECKED
x = 0.40
for name, checked in opts:
    ax.add_patch(plt.Rectangle((x, y - 0.035), 0.025, 0.025, fill=False, lw=1.5))
    if checked:
        ax.text(x + 0.0125, y - 0.023, "X", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(x + 0.035, y - 0.023, name, fontsize=12, va="center")
    x += 0.16
y -= 0.10
ax.text(0.08, y, "Rollback plan attached:", fontsize=12, weight="bold")
for name, checked, xo in [("Yes", True, 0.42), ("No", False, 0.55)]:
    ax.add_patch(plt.Rectangle((xo, y - 0.012), 0.025, 0.025, fill=False, lw=1.5))
    if checked:
        ax.text(xo + 0.0125, y, "X", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(xo + 0.035, y, name, fontsize=12, va="center")
y -= 0.08
ax.text(0.08, y, "Approved by CAB:", fontsize=12, weight="bold")
ax.text(0.38, y, "R. Okafor — May 29, 2025", fontsize=12, family="cursive")
fig.savefig(f"{OUT}/pages/vz_form_p00.png", dpi=140, bbox_inches="tight")
plt.close(fig)
FACTS["vz_form"] = {
    "labeled": {"requested_by": "Dana Whitfield", "window": "June 7, 2025, 02:00-06:00 UTC",
                "approver": "R. Okafor", "approval_date": "May 29, 2025"},
    "structural": {"risk_class_checked": "Medium", "rollback_attached": "Yes"}}

# ---------------------------------------------------------------- image-only PDFs (for the baseline pipeline)
import fitz
os.makedirs(f"{OUT}/pdfs", exist_ok=True)
import glob as g
for png in sorted(g.glob(f"{OUT}/pages/*.png")):
    name = os.path.splitext(os.path.basename(png))[0].rsplit("_p", 1)[0]
    pdfp = f"{OUT}/pdfs/{name}.pdf"
    doc = fitz.open(pdfp) if os.path.exists(pdfp) else fitz.open()
    pix = fitz.Pixmap(png)
    page = doc.new_page(width=pix.width * 72 / 140, height=pix.height * 72 / 140)
    page.insert_image(page.rect, filename=png)
    doc.save(pdfp, incremental=False, deflate=True) if not os.path.exists(pdfp) else doc.saveIncr()
    doc.close()

with open(f"{OUT}/facts.json", "w") as f:
    json.dump(FACTS, f, indent=1)
print("vizdocs generated:", sorted(os.listdir(f"{OUT}/pages")))

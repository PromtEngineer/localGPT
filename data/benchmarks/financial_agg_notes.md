# Verification log — corpus-level aggregate questions (financial_docs)

All evidence extracted mechanically from
`/Users/prompt/Github/multimodal_agentic_rag/data/processed/financial_docs/<doc_id>/pages.jsonl`
(text layer) and `tables.duckdb` (read-only). No LLM calls, no model loads. Method per check:
Python regex scans over every page of every candidate doc, plus DuckDB reads of extracted
income-statement tables as a second source. Date of verification: 2026-07-17.

Corpus companies (16 unique): NVIDIA (fin002), AMD (fin003 + fin024 slides), Tesla (fin018 +
fin004/fin014 decks), Alphabet (fin007), Apple (fin008), Amazon (fin009), Meta (fin010), Intel
(fin011), Netflix (fin012), Uber (fin013), Airbnb (fin015), Coinbase (fin016), Qualcomm (fin019),
Spotify (fin020), Roku (fin021), Pinterest (fin022). Non-company macro docs: fin001, fin005,
fin006, fin017, fin023.

---

## fin_agg_q01 — enumeration-with-filter: YoY total revenue decline, latest full FY

Method: regex scan of every company doc for income-statement revenue lines
(`(?im)^(total (net )?revenue[s]?|net revenue[s]?|revenue[s]?|net sales|total net sales)` followed
by dollar amounts), then printed the surrounding statement block to capture BOTH fiscal-year
columns. Column order confirmed from each page's year headers.

Extracted values (latest FY vs prior FY, $M unless noted):

| Company | Doc/page | Latest FY | Prior FY | YoY |
|---|---|---|---|---|
| NVIDIA | fin002 p38 (also p52, p79/p80) | 130,497 (FY ended 1/26/2025) | 60,922 | **+114%** (page literally says "Up 114%") |
| AMD | fin003 p51 (p50, p56, p70); fin024 p15/p29 "Record revenue of $25.8B increased 14% y/y" | 25,785 (FY ended 12/28/2024) | 22,680 | +13.7% |
| Alphabet | fin007 p65 ("Total revenues $282,836 / $307,394 / $350,018" for 2022/2023/2024) | 350,018 | 307,394 | +13.9% |
| Apple | fin008 p32 ("Total net sales 391,035 / 383,285 / 394,328" for 2024/2023/2022) | 391,035 (FY ended 9/28/2024) | 383,285 | +2.0% |
| Amazon | fin009 p49 ("Total net sales 513,983 / 574,785 / 637,959" for 2022/2023/2024) | 637,959 | 574,785 | +11.0% |
| Meta | fin010 p88 ("Revenue $164,501 / $134,902 / $116,609" for 2024/2023/2022) | 164,501 | 134,902 | +21.9% |
| **Intel** | fin011 p74 ("Total net revenue $53,101 / $54,228 / $63,054" for 2024/2023/2022); also p25, p59 | **53,101** (FY ended 12/28/2024) | **54,228** | **−2.1% ← ONLY DECLINE** |
| Netflix | fin012 p12 (Twelve Months Ended 12/31: "39,000,966" vs "33,723,297", $K) | 39,001.0 | 33,723.3 | +15.7% |
| Uber | fin013 p22 (columns Q4'23, Q4'24, FY2023, FY2024: "Revenue $9,936 / $11,959 / $37,281 / $43,978") | 43,978 | 37,281 | +18.0% |
| Airbnb | fin015 p24 ("Revenue $2,218 / $2,480 / $9,917 / $11,102"); p12 "$11.1B Revenue 12% Y/Y" | 11,102 | 9,917 | +11.9% |
| Coinbase | fin016 p4 "2024 total revenue was $6.6 billion, up 111% Y/Y"; p20 "6,564,028" vs "3,108,383" ($K) | 6,564.0 | 3,108.4 | +111.2% |
| Tesla | fin018 p58 ("Total revenues $97,690 / $96,773"), p52 (97,690 / 96,773 / 81,462); fin004 p5/p33 (31,536 / 53,823 / 81,462 / 96,773 / 97,690) | 97,690 | 96,773 | +0.9% (INCREASE — near-trap) |
| Qualcomm | fin019 p61 ("Total revenues 38,962 / 35,820 / 44,200" for FY2024/FY2023/FY2022) | 38,962 (FY ended 9/29/2024) | 35,820 | +8.8% |
| Spotify | fin020 p35 (annual series "7,880 / 9,668 / 11,727 / 13,247 / 15,673" + "18%") | EUR 15,673 | EUR 13,247 | +18.3% |
| Roku | fin021 p8 ("Total net revenue 1,201,047 / 984,425 / 4,112,898 / 3,484,619" $K); p1 "up 18% YoY" | 4,112.9 | 3,484.6 | +18.0% |
| Pinterest | fin022 p130 ("Revenue $3,646,166 / $3,055,071 / $2,802,574" $K, 2024/2023/2022); also p127, p143, p164 | 3,646.2 | 3,055.1 | +19.3% |

DuckDB cross-checks: `t_fin019_p61_0` row ('Total revenues','38,962','35,820','44,200');
`t_fin003_p50_0` row ('Total net revenue','25,785','22,680').

Negative/trap check: fin014 (Tesla Q1-2025 deck) p4 shows quarterly "Total revenues ...
25,707 → 19,335, −9%" — a QUARTERLY decline; question phrasing pins to "latest full fiscal
year," where Tesla grew +0.9%. All 16 companies checked; **gold = {Intel}**.

## fin_agg_q02 — counting: fiscal year-end ≠ December 31, 2024 (annual filers)

Method: regex `(?i)for the (fiscal )?year ended...` over all pages of the 10 annual filers;
cover-page hit recorded for each.

| Filer | Doc/page | Cover text | ≠ Dec 31? |
|---|---|---|---|
| NVIDIA | fin002 p1 | "For the fiscal year ended January 26, 2025" | YES |
| AMD | fin003 p1 | "For the fiscal year ended December 28, 2024" | YES |
| Alphabet | fin007 p1 | "For the fiscal year ended December 31, 2024" | no |
| Apple | fin008 p1 | "For the fiscal year ended September 28, 2024" | YES |
| Amazon | fin009 p13 | "For the fiscal year ended December 31, 2024" (10-K cover inside annual report) | no |
| Meta | fin010 p1 | "For the fiscal year ended December 31, 2024" | no |
| Intel | fin011 p1 | "For the fiscal year ended December 28, 2024." | YES |
| Tesla | fin018 p1 | "For the fiscal year ended December 31, 2024" | no |
| Qualcomm | fin019 p1 | "For the fiscal year ended September 29, 2024" (p6: "fiscal year ending on the last Sunday in September") | YES |
| Pinterest | fin022 p77 | "For the fiscal year ended December 31, 2024" (SEC cover inside annual report) | no |

**Gold = 5: NVIDIA, AMD, Apple, Intel, Qualcomm.** The AMD/Intel Dec-28 dates are the
designed trap (52/53-week fiscal calendars).

## fin_agg_q03 — corpus superlative: highest YoY revenue growth, latest full FY

Method: same extraction table as q01; growth rates computed from the extracted pairs.
Ranking (computed): NVIDIA +114.2% (130,497/60,922; doc says "Up 114%", fin002 p38) >
Coinbase +111.2% (6,564.028/3,108.383; doc says "up 111% Y/Y", fin016 p4) > Meta +21.9% >
Pinterest +19.3% > Spotify +18.3% > Roku +18.0% > Uber +18.0% > Netflix +15.7% >
Alphabet +13.9% > AMD +13.7% > Airbnb +11.9% > Amazon +11.0% > Qualcomm +8.8% >
Apple +2.0% > Tesla +0.9% > Intel −2.1%.
**Gold = NVIDIA (+114%), runner-up Coinbase (+111%).** The 3-point margin is the designed
near-tie; both growth rates are printed verbatim in the respective docs, so grading is anchored
to document text, not arithmetic.

## fin_agg_q04 — cross-document aggregation: combined semiconductor revenue

Components (page text + DuckDB):
- NVIDIA FY2025: 130,497 (fin002 p38, p52)
- AMD FY2024: 25,785 (fin003 p50/p51; DuckDB `t_fin003_p50_0`; corroborated fin024 p15/p29 "$25.8B")
- Intel FY2024: 53,101 (fin011 p74, p25, p59)
- Qualcomm FY2024: 38,962 (fin019 p61; DuckDB `t_fin019_p61_0`)

Sum: 130,497 + 25,785 + 53,101 + 38,962 = **248,345 ($M) ≈ $248.3B**.
Negative check: no other corpus company is a semiconductor filer (Alphabet/Apple/Amazon/
Meta/Tesla design chips but are not semiconductor companies; question names the four
explicitly to remove ambiguity). Trap note recorded: using NVIDIA's prior FY (60,922) gives
~178.8B — rejected in gold.

## fin_agg_q05 — entity co-mention bridge: who mentions NVIDIA?

Method: `(?i)nvidia|\bNVDA\b` over every page of all 24 docs. Hit counts (pages):
fin002: 81 pages (own 10-K, excluded by construction); **fin003: 3 (p12,17,19)**;
**fin011: 3 (p17,33,108)**; **fin019: 2 (p12,31)**; **fin023: 1 (p17)**; **fin024: 1 (p2)**;
all other 18 docs: 0 hits (fin001, fin004, fin005, fin006, fin007, fin008, fin009, fin010,
fin012, fin013, fin014, fin015, fin016, fin017, fin018, fin020, fin021, fin022 — exhaustive
negative check).

Contexts captured:
- fin003 p12: "we compete primarily against Intel Corporation (Intel) and NVIDIA Corporation (NVIDIA)"
- fin011 p17: "providers of GPU systems such as NVIDIA"
- fin019 p12/p31: competitor lists "...MediaTek, Mobileye, Nvidia, NXP Semiconductors..."
- fin023 p17: footnote "Alphabet, Amazon, Apple, Meta, Microsoft, Nvidia and Tesla" (big-tech equity index)
- fin024 p2: "Nvidia's dominance in the graphics processing unit market and its aggressive business practices"

Alias sweep: `(?i)geforce|\bCUDA\b|\bH100\b|\bA100\b|\bGB200\b|Blackwell` over all docs →
single extra hit: fin004 p10 "deployment of Cortex, a ~50k H100 training cluster" (product
named, company NOT named → excluded from gold; noted in answer as non-penalized borderline).
**Gold = {fin003, fin011, fin019, fin023, fin024}.**

## fin_agg_q06 — enumeration-with-filter: dividend payers among the 10 annual filers

Method: regex family over every page of the 10 filers: `never (declared|paid)...dividend`,
`do not (anticipate|intend|expect)...dividend`, `dividends? (declared|paid)...`,
`quarterly (cash )?dividend...`; plus cash-flow-statement financing sections.

Payers (positive evidence):
- NVIDIA fin002 p34/p78: "cash dividends to our shareholders of $834 million, $395 million, and $398 million"; p56 cash-flow "Dividends paid (834)".
- Alphabet fin007 p24: "began paying regular cash dividends to our Class A, Class B, and Class C stockholders"; p56 "Dividends and dividend equivalents declared ($0.60 per share) ... (7,536)"; p84 "cash dividends paid ... 7,363" total.
- Apple fin008 p29: "raised its quarterly dividend from $0.24 to $0.25 per share beginning in May 2024 ... paid dividends and dividend equivalents of $15.2 billion"; p36 cash-flow "(15,234)".
- Meta fin010 p61/p79: "Beginning in February 2024, we declared and paid four quarterly cash dividends ... totaling $2.00 for each share ... Total dividends and dividend equivalents paid were $5.07 billion".
- Intel fin011 p62 cash-flow: "Payment of dividends to stockholders (1,599) (3,088) (5,997)" → $1,599M PAID in 2024; p8/p30: "suspended the declaration of quarterly dividends starting with the fourth quarter of 2024" (paid Q1–Q3, still a 2024 payer).
- Qualcomm fin019 p63 cash-flow: "Dividends paid (3,687) (3,462) (3,212)"; p47 dividend-per-share table for fiscal 2024/2023.

Non-payers (negative evidence):
- AMD fin003 p4: "AMD's expectation that it will not pay dividends in the near future"; p81: "expected dividend yield is zero as the Company does not expect to pay dividends"; no dividends-paid line anywhere.
- Amazon fin009: full-doc grep of "dividend" yields only p35 (hypothetical financing options) and p90 (index-return methodology); p48 financing activities has NO dividend line → no dividends paid.
- Tesla fin018 p33: "never declared or paid cash dividends on our common stock nor do we anticipate paying any such cash dividends in the foreseeable future".
- Pinterest fin022 p119: "never declared or paid dividends on our capital stock and do not intend to pay any dividends"; p160 same.

**Gold = {NVIDIA, Alphabet, Apple, Meta, Intel, Qualcomm} (6 of 10).**

## fin_agg_q07 — thematic synthesis (rubric) across the 5 macro reports

Method: theme-regex page counts over fin001, fin005, fin006, fin017, fin023:

| Theme (regex) | fin001 | fin005 | fin006 | fin017 | fin023 | present in |
|---|---|---|---|---|---|---|
| tariffs/trade (`tariff\|trade polic\|trade tension`) | 21 pp | 28 pp | 2 pp | 82 pp | 7 pp | 5/5 |
| policy/economic uncertainty | 1 | 10 | 5 | 59 | 3 | 5/5 |
| fiscal/public debt (`fiscal (sustainab\|deficit\|position\|risk)\|government debt\|public debt`) | 2 | 26 | 3 | 44 | 1 | 5/5 |
| inflation | 32 | 39 | 19 | 77 | 28 | 5/5 |
| geopolitical | 0 | 11 | 2 | 38 | 2 | 4/5 |
| (supporting) asset valuations | 1 | 0 | 12 | 0 | 1 | 3/5 |
| (supporting) dollar depreciation | 1 | 3 | 0 | 0 | 0 | 2/5 |

Sample passages recorded (one per doc per theme), e.g. fin001 p11 "concerns about the effects
of higher tariffs on inflation and employment"; fin005 p12 "Elevated trade and policy
uncertainty could weigh on economic activity"; fin006 p11 "risks to global trade, policy
uncertainty, and U.S. fiscal debt sustainability"; fin017 p19 "increased trade tension and
heightened policy uncertainty"; fin023 p11 "key policy announcements, including on tariffs".
Rubric requires ≥3 of the 5 themes that appear in ≥4/5 reports; the two "supporting" themes
are optional extras, not required and not penalized.

---

### Commands used (representative)

```bash
# revenue line scan (per doc)
uv run python - # regex: (?im)^(total (net )?revenue[s]?|net revenue[s]?|revenue[s]?|net sales|total net sales)\s*[:$]?\s*\$?\s*([\d,]{4,})
# income-statement context windows around anchors on the identified pages
# fiscal-year-end scan: (?i)for the (fiscal )?year ended[^\n]{0,60}
# NVIDIA mention scan: (?i)nvidia|\bNVDA\b   (24 docs, all pages)
# alias scan: (?i)geforce|\bCUDA\b|\bH100\b|\bA100\b|\bGB200\b|Blackwell
# dividend scan: (?i)never (declared|paid)... / dividends? (declared|paid|of)... / quarterly (cash )?dividend...
# theme scan: see table above
```

```sql
-- duckdb (read_only=True) cross-checks
SELECT * FROM t_fin019_p61_0;  -- ('Total revenues','38,962','35,820','44,200')
SELECT * FROM t_fin003_p50_0;  -- ('Total net revenue','25,785','22,680')
SELECT view_name, headers FROM _catalog WHERE doc_id IN ('fin002','fin003','fin011','fin019');
```

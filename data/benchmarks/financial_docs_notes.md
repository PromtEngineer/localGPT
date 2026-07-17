# financial_docs benchmark - methodology notes

- 15 multi-hop questions over 14 of the 24 corpus PDFs. Every hop was verified by rendering the exact PDF page with the Read tool (visual check of the table/figure), after locating candidate pages via per-page text extraction (PyMuPDF).
- "pages" are absolute PDF page numbers (first PDF page = 1). Printed page numbers frequently differ (e.g., fin003 PDF p50 = printed p47; fin001 PDF p57 = printed p47; fin017 PDF p26 = printed p4).
- Hop-type mix: 9 cross_doc (60%), 3 table_math, 2 figure_read, 1 cross_section. All 15 questions include at least one table or figure hop. Difficulty: 5 medium / 10 hard.
- 2 questions mix company filings with macro reports: fin_q14 (BIS Quarterly Review Mar-2025 + NVIDIA FY2025 10-K) and fin_q15 (Tesla Q1-2025 deck + World Bank GEP June-2025).
- All computed values (growth rates, sums, differences, shares) were recomputed from the verified table inputs, and where the filing itself states the rate (e.g., NVIDIA +142%, AMD +94%, Tesla energy +67%) the answer matches the filing's own figure.
- Docs used: fin001, fin002, fin003, fin004, fin007, fin008, fin009, fin010, fin011, fin012, fin013, fin014, fin017, fin018, fin020, fin023.
- Docs not used (no defect found, simply not needed for the 15 QA pairs): fin005 (BIS AER), fin006 (Fed FSR), fin015 (Airbnb), fin016 (Coinbase), fin019 (Qualcomm), fin021 (Roku), fin022 (Pinterest), fin024 (AMD slides). All appeared usable in spot checks; none was found unusable for QA.
- Caveat: fin009 is the full Amazon annual report wrapper around the 10-K, and fin011 (Intel) interleaves MD&A ahead of Item 1 financial statements, so absolute page offsets in those two differ most from printed folios.

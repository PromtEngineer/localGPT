"""Generate the synthetic HR leave-policy corpus used by the eval harness.

The document is entirely fictional (Northwind Robotics is not a real company)
so that no planted fact can be answered from the generation model's parametric
knowledge — a wrong answer is unambiguously a retrieval or grounding failure.

Run from the repo root:

    .venv/bin/python eval/corpora/make_hr_handbook.py

Writes ``eval/corpora/northwind_leave_policy.pdf``. The planted facts live in
``eval/corpora/northwind_leave_policy.facts.json`` (hand-maintained sidecar);
this script asserts every ``expected`` substring in that sidecar really appears
in the rendered page text before it saves the file.
"""

import json
import os
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "northwind_leave_policy.pdf")
FACTS_PATH = os.path.join(HERE, "northwind_leave_policy.facts.json")

PAGES = [
    (
        "Northwind Robotics - Leave and Absence Policy Handbook",
        """Policy PPL-204, revision 4. Effective 1 February 2026.
Owner: Department of People Operations, Northwind Robotics, Gothenburg.

1. ANNUAL LEAVE
Employees below Grade 7 accrue 23 days of paid annual leave per calendar
year. Employees at Grade 7 and above accrue 28 days. Accrual begins on the
first day of employment and is credited monthly in arrears.

A maximum of 5 unused annual leave days may be carried into the following
year. Carried days expire on 31 March and are not paid out on expiry.

2. REQUESTING LEAVE
All leave requests are submitted through the Kestrel HR portal at least 10
working days before the intended start date. Any absence longer than 10
consecutive working days additionally requires written approval from a
director. Requests are answered within 3 working days.
""",
    ),
    (
        "Northwind Robotics - Sickness, Parental and Special Leave",
        """3. SICK LEAVE
Sick leave is paid at 100 percent of base salary for the first 12 weeks of a
single absence, and at 60 percent for a further 8 weeks. A medical
certificate is required once an absence exceeds 4 consecutive working days.

4. PARENTAL LEAVE
Parental leave is 18 weeks per child, of which 6 weeks are fully paid. It
must be taken before the child's third birthday. Parental leave may be split
into no more than 3 separate blocks.

5. BEREAVEMENT LEAVE
Bereavement leave is 5 working days for an immediate family member and 2
working days for an extended family member.

6. JURY DUTY
Jury service is paid in full for up to 15 working days per calendar year.
Any court allowance received must be surrendered to the payroll team.
""",
    ),
    (
        "Northwind Robotics - Sabbatical, Holidays and Exclusions",
        """7. UNPAID SABBATICAL
Employees with at least 4 years of continuous service may apply for an
unpaid sabbatical of up to 90 days. Applications require 60 days written
notice and are approved by the Head of People Operations. A sabbatical does
not interrupt continuous-service accrual.

8. PUBLIC HOLIDAYS
Northwind Robotics recognises 9 public holidays. An employee rostered to
work on a public holiday is paid at 1.5 times the normal rate and receives a
substitute day off within the same quarter.

9. EXCLUSIONS AND FORFEITURE
Annual leave is not paid out on resignation unless the employee has served
more than 6 months. Leave taken without portal approval is recorded as
unauthorised absence and is unpaid. Contractors engaged through an agency are
not covered by this policy.
""",
    ),
]


def build() -> None:
    doc = pymupdf.open()
    for title, body in PAGES:
        page = doc.new_page()
        page.insert_text((60, 70), title, fontsize=13, fontname="helv")
        text_y = 100
        for line in body.strip().split("\n"):
            page.insert_text((60, text_y), line, fontsize=10, fontname="helv")
            text_y += 15
    doc.save(PDF_PATH)
    doc.close()


def verify() -> int:
    doc = pymupdf.open(PDF_PATH)
    full_text = " ".join(page.get_text() for page in doc)
    doc.close()
    normalised = " ".join(full_text.split())

    with open(FACTS_PATH, "r", encoding="utf-8") as fh:
        facts = json.load(fh)["facts"]

    missing = [f["id"] for f in facts if f["expected"] not in normalised]
    if missing:
        print(f"MISSING expected substrings in rendered PDF: {missing}")
        return 1
    print(f"OK: {len(facts)} planted facts all present in {PDF_PATH}")
    return 0


if __name__ == "__main__":
    build()
    raise SystemExit(verify())

"""
Create a test PDF that reproduces the "formula without explanation" retrieval failure.

Why this file exists:
    data/sample.pdf (the Acme annual report) contains no formulas, so it cannot
    demonstrate the Stage 1 problem we are trying to fix.

The document is written so that, with CHUNK_SIZE=500 and CHUNK_OVERLAP=50, the
compound interest section splits into ~3 sequential chunks:

    chunk N     → intro + the formula itself   ("A = P(1 + r/n)^(nt)")
    chunk N+1   → the variable definitions     ("P = principal, r = rate, ...")
    chunk N+2   → how it works + worked example

A question like "what is the compound interest formula?" is textually similar to
chunk N only. Chunks N+1 and N+2 contain the information needed to *explain* the
formula, but they do not look like the question, so pure similarity search will
never retrieve them. That is the failure we want to see in the baseline.

Run once: python scripts/create_formula_pdf.py
"""

from pathlib import Path

import fitz

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "formula_sample.pdf"

# A4 page with a comfortable text margin.
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 56

PAGES = [
    # ---- Page 1: unrelated content, gives retrieval something to compete with
    """Introduction to Financial Mathematics

This handbook introduces the core calculations used in personal and corporate
finance. It is intended as a reference for analysts who need the formulas
together with the reasoning behind them.

Chapters cover interest calculations, loan amortisation, present value, and
basic risk measures. Each chapter states a formula, defines its variables,
explains the underlying mechanism, and works through a numerical example.""",

    # ---- Page 2: THE TEST CASE. Long enough to split across ~3 chunks.
    """Compound Interest

Compound interest is one of the most important concepts in finance. Unlike
simple interest, which is calculated only on the original principal, compound
interest is calculated on the principal plus all previously accumulated
interest. This section introduces the standard formula, defines each variable,
explains the mechanism, and works through a numerical example.

The compound interest formula is:

A = P(1 + r/n)^(nt)

Where the variables are defined as follows:

P = the principal, meaning the initial amount of money invested or borrowed
r = the annual nominal interest rate, expressed as a decimal
n = the number of times interest is compounded per year
t = the total time the money is invested or borrowed, measured in years
A = the final amount accumulated after t years, including all interest

How the formula works. Interest is added to the principal at the end of every
compounding period. Once added, that interest itself begins earning interest in
each later period. This is the reason growth is exponential rather than linear.
Increasing n makes compounding more frequent, which raises the final amount A,
although the benefit shows diminishing returns as n becomes large.

Worked example. Suppose P = 1000, r = 0.05, n = 12, and t = 10. Substituting
gives A = 1000(1 + 0.05/12)^(120), which evaluates to approximately 1647.01.
The total interest earned over the ten years is therefore about 647.01.""",

    # ---- Page 3: a second, similar formula — creates realistic confusion
    """Simple Interest

Simple interest is calculated only on the original principal and ignores any
interest already earned.

The simple interest formula is:

I = P * r * t

Where P is the principal, r is the annual interest rate as a decimal, and t is
the time in years. The final amount is A = P + I.

Because simple interest never compounds, it grows linearly with time. For the
same principal, rate, and duration, simple interest always produces a smaller
final amount than compound interest.""",

    # ---- Page 4: another formula whose explanation also spans a boundary
    """Present Value

Present value answers the question: what is a future sum of money worth today?

The present value formula is:

PV = FV / (1 + r)^n

Where the variables are defined as follows:

PV = the present value, the amount the future sum is worth today
FV = the future value, the nominal amount received later
r = the discount rate per period, expressed as a decimal
n = the number of periods until the money is received

Why discounting is necessary. Money available today can be invested and earn a
return, so a rupee received in five years is worth less than a rupee received
now. The discount rate r represents the return forgone by waiting. A higher
discount rate produces a lower present value.""",

    # ---- Page 5: prose only, no formula — useful for negative tests
    """Risk and Diversification

Diversification reduces the impact of any single asset performing badly.
Holding assets whose returns are weakly correlated lowers the variability of
the portfolio as a whole without necessarily lowering expected return.

Diversification cannot remove market-wide risk. Systematic risk affects all
assets simultaneously and cannot be eliminated by holding more of them.""",
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    text_rect = fitz.Rect(
        MARGIN,
        MARGIN,
        PAGE_WIDTH - MARGIN,
        PAGE_HEIGHT - MARGIN,
    )

    for page_text in PAGES:
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        # insert_textbox wraps text inside the rect; insert_text does not wrap,
        # which would push our long paragraphs off the page edge.
        leftover = page.insert_textbox(
            text_rect,
            page_text,
            fontsize=10.5,
            fontname="helv",
        )
        if leftover < 0:
            raise RuntimeError(
                "Page text did not fit in the text box. "
                "Shorten the page or reduce the font size."
            )

    doc.save(OUTPUT)
    doc.close()
    print(f"Created {OUTPUT} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()

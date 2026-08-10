"""
Create a multi-page sample PDF for testing retrieval.

Run once: python scripts/create_sample_pdf.py
"""

from pathlib import Path

import fitz

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"

# Content spread across pages so test cases can target specific pages.
PAGES = [
    """Acme Corporation — Annual Report 2024 (Overview)

Acme Corporation was founded in 1985 and is headquartered in Austin, Texas.
The company designs cloud software for small and medium businesses.
In 2024, Acme expanded into the European market with new offices in Berlin and Amsterdam.""",

    """Products and Services

Acme offers three main products:
1. CloudLedger — accounting and invoicing software
2. TeamSync — project management and collaboration tools
3. DataPulse — analytics dashboards for business metrics

All products are sold as monthly subscriptions. Enterprise customers receive
dedicated support and custom integrations.""",

    """Leadership Team

The Chief Executive Officer is Maria Chen, who joined Acme in 2019.
The Chief Technology Officer is James Okonkwo, responsible for product engineering.
The Chief Financial Officer is Priya Sharma, who oversees financial planning.""",

    """Employee and Culture

Acme employed 1,240 people worldwide at the end of 2024.
The company emphasizes remote-first work and invests heavily in employee training.
Annual employee satisfaction scores averaged 4.2 out of 5 in internal surveys.""",

    """Research and Development

In 2024, Acme invested $18.5 million in research and development.
R&D focused on AI-assisted bookkeeping features and improved mobile applications.
Three new patents were filed related to automated expense categorization.""",

    """Customer Growth

Acme served over 42,000 paying customers in 2024, up from 31,000 in 2023.
Customer retention rate was 94%. Net Promoter Score improved to 52.""",

    """Financial Highlights — 2024

Total revenue for fiscal year 2024 was $127.4 million, an increase of 22% year over year.
Gross profit was $98.1 million. Operating expenses were $76.3 million.
Net income was $14.2 million. Cash and equivalents at year end: $31.8 million.""",

    """Financial Highlights — Prior Years

In 2023, revenue was $104.3 million. In 2022, revenue was $89.7 million.
The company has been profitable for six consecutive years.""",

    """Risk Factors

Acme faces competition from larger enterprise software vendors.
Currency fluctuations may affect international revenue.
Cybersecurity incidents could harm customer trust and lead to regulatory penalties.""",

    """Future Outlook

Management expects revenue growth of 15–18% in 2025.
Planned launches include CloudLedger 3.0 and an API platform for third-party developers.
Acme intends to hire approximately 200 employees in the next fiscal year.""",

    """Sustainability

Acme committed to carbon-neutral cloud operations by 2030.
In 2024, 68% of office energy came from renewable sources.
The company published its first sustainability report in March 2024.""",

    """Appendix — Contact Information

Investor relations: investors@acme.example.com
Media inquiries: press@acme.example.com
Corporate headquarters: 100 Innovation Drive, Austin, TX 78701""",
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    for page_text in PAGES:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text(
            (50, 72),
            page_text,
            fontsize=11,
            fontname="helv",
        )

    doc.save(OUTPUT)
    doc.close()
    print(f"Created {OUTPUT} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()

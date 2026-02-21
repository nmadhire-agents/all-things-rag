from __future__ import annotations

from dataclasses import asdict
import json
import random
from pathlib import Path

from .schema import Document, QueryExample

COUNTRIES = [
    "Canada",
    "Germany",
    "Japan",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "India",
    "Netherlands",
    "Singapore",
]

FORM_CODES = ["A-12", "TX-88", "GM-14", "SEC-77", "HR-21"]

SECTIONS = [
    "Remote Work",
    "International Work",
    "International Tax",
    "Travel Approval",
    "Security",
]

QUERY_TEMPLATES = {
    "Remote Work": [
        "Can I work remotely from cafes and home?",
        "What does the handbook say about domestic remote work locations?",
        "Am I allowed to work from a co-working space in my home country?",
    ],
    "International Work": [
        "What is the policy for working from another country?",
        "How many days can I work internationally without permit support?",
        "When do I need Global Mobility approval for overseas remote work?",
    ],
    "International Tax": [
        "Do I need Form {form_code} for my trip?",
        "When is Form {form_code} required for cross-border work?",
        "What triggers tax review with Form {form_code}?",
    ],
    "Travel Approval": [
        "How far in advance should I request international travel approval?",
        "What is the lead time for submitting international travel requests?",
        "Do I need manager and finance approval before international travel?",
    ],
    "Security": [
        "What security controls are mandatory while traveling with customer data?",
        "Which safeguards are required when accessing customer data abroad?",
        "What is the incident reporting requirement if my device is lost while traveling?",
    ],
}

SECTION_RATIONALES = {
    "Remote Work": "Requires domestic remote work policy context.",
    "International Work": "Requires cap-days and permit requirement from International Work section.",
    "International Tax": "Requires exact form-code lookup and tax context.",
    "Travel Approval": "Requires travel portal lead time and approval constraints.",
    "Security": "Requires VPN/MFA/encryption requirements from security section.",
}

HANDBOOK_TEXT = """# Z-Tech Global Work Handbook

## Remote Work
Z-Tech encourages remote work from home, co-working spaces, or temporary domestic locations.
Employees must stay reachable during assigned timezone hours and use approved managed devices.
Public Wi-Fi usage is allowed only with corporate VPN enabled.

## International Work
Working from another country is capped at 14 days in a rolling 12-month period without permit support.
Beyond 14 days, employees must open a Global Mobility case and obtain HR, Legal, and Payroll approval.
Violations can trigger immigration, payroll, and tax exposure.

## International Tax
Employees traveling internationally may need Form A-12 before departure when cross-border work exceeds 7 business days.
The tax team uses Form A-12 to assess treaty relief, withholding obligations, and permanent establishment risk.

## Travel Approval
International travel requests must be submitted at least 14 days before departure in the travel portal.
Manager approval is mandatory, and finance approval is required for total costs above 2,000 USD equivalent.

## Security
Employees handling customer data while traveling must use VPN, hardware-backed MFA, and encrypted storage.
Lost or stolen devices must be reported within one hour to security operations.
"""


def _remote_work_paragraph(company: str, office: str) -> str:
    return (
        f"{company} supports remote work from home, co-working spaces, or temporary locations within "
        f"the employee's home country. Employees should remain reachable during their assigned timezone "
        f"hours and must use approved devices with endpoint protection. Office anchor location is {office}."
    )


def _international_work_paragraph(country: str, max_days: int) -> str:
    return (
        f"Working from another country such as {country} is capped at {max_days} days per rolling 12 months "
        f"without a permit. Beyond this duration, employees must engage Global Mobility and obtain written "
        f"approval from HR, Legal, and Payroll. Non-compliance can create immigration and payroll exposure."
    )


def _international_tax_paragraph(form_code: str, country: str) -> str:
    return (
        f"For cross-border assignments involving {country}, employees must submit Form {form_code} before "
        f"travel if expected presence exceeds 7 business days. The form triggers tax review for treaty, "
        f"withholding, and permanent establishment risks."
    )


def _travel_approval_paragraph(country: str) -> str:
    return (
        f"International travel requests to {country} must be entered 14 days before departure in the travel "
        f"portal. Manager approval is mandatory, and finance pre-approval is required if total spend exceeds "
        f"$2,000 USD equivalent."
    )


def _security_paragraph() -> str:
    return (
        "Employees handling customer data while traveling must use VPN, hardware-backed MFA, and encrypted "
        "storage. Public kiosk access is prohibited. Lost-device incidents must be reported within 1 hour."
    )


def parse_handbook_to_documents(handbook_text: str) -> list[Document]:
    documents: list[Document] = []
    current_section = ""
    section_lines: list[str] = []

    def _commit_section(section: str, lines: list[str]) -> None:
        if not section:
            return
        text = " ".join(line.strip() for line in lines if line.strip())
        if not text:
            return
        doc_id = f"DOC-HB-{section.replace(' ', '').upper()}"
        documents.append(
            Document(
                doc_id=doc_id,
                title=f"Z-Tech Handbook - {section}",
                section=section,
                text=text,
            )
        )

    for raw_line in handbook_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            _commit_section(current_section, section_lines)
            current_section = line[3:].strip()
            section_lines = []
            continue
        if line.startswith("# "):
            continue
        section_lines.append(line)

    _commit_section(current_section, section_lines)
    return documents


def generate_documents(doc_count: int = 500, seed: int = 42) -> list[Document]:
    del doc_count, seed
    return parse_handbook_to_documents(HANDBOOK_TEXT)


def generate_queries(documents: list[Document], query_count: int = 200, seed: int = 42) -> list[QueryExample]:
    random.seed(seed)
    grouped: dict[str, list[Document]] = {section: [] for section in SECTIONS}
    for document in documents:
        grouped[document.section].append(document)

    queries: list[QueryExample] = []
    for query_idx in range(query_count):
        section = SECTIONS[query_idx % len(SECTIONS)]
        target_document = random.choice(grouped[section])

        template = QUERY_TEMPLATES[section][query_idx % len(QUERY_TEMPLATES[section])]
        if section == "International Tax":
            form_code = target_document.text.split("Form ")[1].split(" ")[0]
            question = template.format(form_code=form_code)
        else:
            question = template

        rationale = SECTION_RATIONALES[section]

        queries.append(
            QueryExample(
                query_id=f"Q-{query_idx:04d}",
                question=question,
                relevant_chunk_ids=[],
                target_doc_id=target_document.doc_id,
                target_section=target_document.section,
                rationale=rationale,
            )
        )

    return queries


def save_dataset(documents: list[Document], queries: list[QueryExample], output_dir: str = "data") -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    docs_path = root / "documents.jsonl"
    with docs_path.open("w", encoding="utf-8") as file_handle:
        for document in documents:
            file_handle.write(json.dumps(asdict(document)) + "\n")

    query_path = root / "queries.jsonl"
    with query_path.open("w", encoding="utf-8") as file_handle:
        for query in queries:
            file_handle.write(json.dumps(asdict(query)) + "\n")


def build_and_save_dataset(output_dir: str = "data", doc_count: int = 500, query_count: int = 200) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "handbook_manual.txt").write_text(HANDBOOK_TEXT, encoding="utf-8")

    documents = generate_documents(doc_count=doc_count)
    queries = generate_queries(documents=documents, query_count=query_count)
    save_dataset(documents=documents, queries=queries, output_dir=output_dir)

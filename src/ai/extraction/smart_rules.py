# smart_rules.py
# Unified structured knowledge extraction via deterministic parsing rules

import re
from datetime import datetime
from typing import Dict, Any, List


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_structured_knowledge(text: str, source_name: str) -> List[Dict[str, Any]]:
    """Extract structured knowledge items from raw text using robust rules.
    Returns a list of rows consumable by the frontend knowledge table.
    """
    if not text:
        return []

    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]

    knowledge_items: List[Dict[str, Any]] = []

    # Helpers
    def add_item(name: str, type_: str, category: str, description: str, confidence: float = 0.85):
        knowledge_items.append({
            "Knowledge": name,
            "Type": type_,
            "Confidence": round(confidence, 2),
            "Category": category,
            "Description": description,
            "Source": source_name,
            "Extracted_At": _now_str(),
        })

    # Normalize paragraph blocks
    paragraphs: List[str] = []
    buf: List[str] = []
    for l in lines:
        if re.match(r"^\s*[•\-*\d]+[\.)]?\s+", l) or l.endswith(":"):
            # boundary; flush paragraph buffer
            if buf:
                paragraphs.append(" ".join(buf).strip())
                buf = []
            paragraphs.append(l)
        else:
            buf.append(l)
    if buf:
        paragraphs.append(" ".join(buf).strip())

    text_lower = text.lower()

    # 1) Concepts: definition-like lines: "Term (ABBR): description" or "Term: description"
    for para in paragraphs:
        m = re.match(r"^([A-Z][A-Za-z0-9() /-]{2,}):\s+(.*)$", para)
        if m:
            term = m.group(1).strip()
            desc = m.group(2).strip()
            # Avoid headings like "Introduction:" with no content
            if len(desc.split()) >= 3:
                add_item(term, "concepts", "Concept", desc, 0.92)

    # 2) Processes: detect headings ending with ':' followed by enumerations in subsequent lines
    # Collect heading index map
    heading_idxs = [i for i, p in enumerate(paragraphs) if p.endswith(":") and len(p.split()) <= 12]
    for idx in heading_idxs:
        heading = paragraphs[idx][:-1].strip()
        # Lookahead for enumerations
        steps: List[str] = []
        j = idx + 1
        while j < len(paragraphs):
            p = paragraphs[j]
            if p.endswith(":"):
                break
            bullet = re.match(r"^(?:[•\-*]|\d+[\.)])\s+(.*)$", p)
            if bullet:
                steps.append(bullet.group(1).strip())
            j += 1
        if steps:
            # Build description string
            steps_str = "; ".join(steps[:12])
            add_item(heading, "processes", "Process", f"Steps: {steps_str}", 0.9)

    # 3) Systems: detect equipment/tools/PPE sections and bullet items
    system_keywords = [
        "equipment", "tools", "monitoring", "pesticide application equipment", "ppe", "personal protective equipment",
        "sprayer", "dusters", "foggers", "misters", "ULD", "bait stations", "glue boards", "pheromone traps"
    ]
    for para in paragraphs:
        if any(k in para.lower() for k in system_keywords):
            # Capture itemized bullets following this heading-like paragraph
            if para.endswith(":"):
                items: List[str] = []
                start = paragraphs.index(para) + 1
                k = start
                while k < len(paragraphs) and not paragraphs[k].endswith(":"):
                    bullet = re.match(r"^(?:[•\-*]|\d+[\.)])\s+(.*)$", paragraphs[k])
                    if bullet:
                        items.append(bullet.group(1).strip())
                    k += 1
                for it in items[:12]:
                    name = it.split(":")[0].strip()
                    add_item(name, "systems", "System", it, 0.86)
            else:
                # If not heading, treat the paragraph itself as a system concept
                add_item(para.split(":")[0], "systems", "System", para, 0.8)

    # 4) Requirements: sentences containing obligation/compliance terms
    requirement_terms = ["must", "shall", "required", "prohibited", "illegal", "comply", "compliance"]
    reg_terms = ["fifra", "epa", "usda", "fda", "state plant board", "inspected"]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        s_clean = s.strip()
        s_low = s_clean.lower()
        if any(t in s_low for t in requirement_terms) or any(t in s_low for t in reg_terms):
            add_item("Regulatory/Operational Requirement", "requirements", "Requirement", s_clean, 0.88)

    # 5) People/roles: extract role-like headings and agency mentions
    role_patterns = [
        r"pesticide applicator|pest management professional|pmp",
        r"sanitation committee|committee",
        r"usda inspector|inspector",
        r"arkansas state plant board|state plant board|regulatory personnel",
        r"employees|workers|applicator"
    ]
    for para in paragraphs:
        low = para.lower()
        if any(re.search(p, low) for p in role_patterns):
            # split role: description on ':' if present
            if ":" in para:
                name, desc = para.split(":", 1)
                add_item(name.strip(), "people", "Role", desc.strip(), 0.83)
            else:
                add_item(para.strip(), "people", "Role", para.strip(), 0.78)

    # 6) Risks: detect risk/contamination/safety hazards
    risk_terms = [
        "risk", "hazard", "danger", "threat", "contamination", "poison", "fire", "explosion", "resistance", "shyness", "neophobia"
    ]
    for s in sentences:
        s_low = s.lower()
        if any(t in s_low for t in risk_terms):
            # Extract a title from the sentence (first noun phrase heuristic)
            title = s.strip().split(".")[0][:80]
            add_item(title, "risks", "Risk", s.strip(), 0.8)

    # De-duplicate by (Knowledge, Description)
    seen = set()
    unique_items: List[Dict[str, Any]] = []
    for item in knowledge_items:
        key = (item["Knowledge"], item["Description"])
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    return unique_items
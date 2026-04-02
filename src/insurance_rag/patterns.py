import re

# ---------------------------------------------------------------------------
# MAJOR SECTION HEADING PATTERN
# Matches explicit named sections AND 1st/2nd level numbered headings.
# ---------------------------------------------------------------------------
_MAJOR_SECTION_RE = re.compile(
    r"^(?:SECTION|PART|CHAPTER|ARTICLE)\s+[A-Z0-9]+[\s\-–:\.]+.+"
    r"|^(\d+\.)(?!\.?\d)\s+\S{2,}",
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# CLAUSE / SUBHEADING PATTERN
# Matches 3rd level numbers, alphabetical/roman lists, AND prefixed items.
# ---------------------------------------------------------------------------
_CLAUSE_RE = re.compile(
    r"^("
    r"(?:Def\.?|Definition|Cond\.?|Condition|Option|Opt\.?)\s*\d+[\.\d]*"
    r"|\d+\.\d+[\.\d]*"
    r"|\(\s*[ivxIVX]+\s*\)"  
    r"|[ivxIVX]+\."          
    r"|\(\s*[a-zA-Z]\s*\)"   
    r"|[a-zA-Z]\."           
    r")\s*(.*)$",            
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# NOT A SECTION (Blocklist)
# Prevents short, bolded warning labels or table headers from being 
# accidentally classified as structural document headings.
# ---------------------------------------------------------------------------
_NOT_A_SECTION = re.compile(
    r"^(note[s]?|important|please note|warning|caution|sr\.?\s*no\.?|"
    r"s\.no|charges?|amount|description|list [ivx]+)\s*:?\s*$",
    re.IGNORECASE,
)
from pydantic import BaseModel, Field
from typing import List, Optional

class SpecificException(BaseModel):
    exists: bool = Field(description="'True' or 'False'")
    clause: Optional[str] = Field(description="The specific clause name if it exists")

# Base schema used by both Single and Comparison queries
class PolicyDecision(BaseModel):
    coverage_status: str = Field(description="'YES', 'NO', or 'CONDITIONAL'")
    primary_clause: Optional[str] = Field(description="The main clause driving the decision")
    specific_exception_found: SpecificException = Field(
        description="Whether a specific exception exists, along with the clause"
    )
    gap_analysis: Optional[str] = Field(
        description="What specific document or clause is missing? Leave empty if none."
    )
    decision_confidence: str = Field(description="'HIGH', 'MEDIUM', or 'LOW'")
    document_completeness: str = Field(description="'COMPLETE', 'PARTIAL', or 'INSUFFICIENT'")


# Specific sub-schema for the comparison winner
class ComparisonVerdict(BaseModel):
    mathematical_winner: str = Field(
        description="'POLICY_A', 'POLICY_B', 'TIE', or 'CANNOT_DETERMINE'"
    )
    winning_reason: Optional[str] = Field(description="The strict logical reason for this winner")


# The final Comparison schema that nests the base models
class ComparisonResult(BaseModel):
    policy_a: PolicyDecision = Field(description="Independent evaluation for Policy A")
    policy_b: PolicyDecision = Field(description="Independent evaluation for Policy B")
    comparison_verdict: ComparisonVerdict = Field(
        description="The final comparison verdict"
    )

class AdvisoryReport(BaseModel):
    executive_summary: str = Field(description="Executive summary in markdown")
    comparison_table: Optional[str] = Field(default=None, description="Markdown table string (only for comparison)")
    deep_dive: str = Field(description="Deep dive in markdown")
    risk_disclosure: str = Field(description="Risk disclosure in markdown")
    follow_up_questions: List[str] = Field(description="List of follow-up questions")
    mandatory_disclaimer: str = Field(description="Mandatory disclaimer string")

class SinglePolicyResponse(BaseModel):
    adjudication: PolicyDecision
    advisory_report: AdvisoryReport

class ComparePolicyResponse(BaseModel):
    adjudication: ComparisonResult
    advisory_report: AdvisoryReport


class OptimizedSearchQuery(BaseModel):
    """Structured query representation for advanced hybrid retrieval."""

    canonical_query: str = Field(
        description="The user's original intent rewritten clearly and concisely."
    )
    expanded_terms: List[str] = Field(
        description="Formal insurance/legal synonyms and related concepts."
    )
    exclusion_terms: List[str] = Field(
        description="Likely policy exclusions, waiting periods, or limits related to the query."
    )
    medical_terms: List[str] = Field(
        description="Specific clinical terms, diagnoses, or treatments mentioned."
    )
    policy_sections: List[str] = Field(
        description="Target policy sections (e.g., 'maternity', 'domiciliary', 'room rent')."
    )

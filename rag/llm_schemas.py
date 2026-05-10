from pydantic import BaseModel, Field
from typing import List


# Base schema used by both Single and Comparison queries
class PolicyDecision(BaseModel):
    coverage_status: str = Field(description="'Yes', 'No', or 'Conditional'")
    primary_clause: str = Field(description="The main clause driving the decision")
    specific_exception_found: str = Field(
        description="'True' or 'False' along with the specific clause name"
    )
    gap_analysis: str = Field(
        description="What specific document or clause is missing? Leave empty if none."
    )
    confidence_score: str = Field(description="'High', 'Medium', or 'Low'")


# Specific sub-schema for the comparison winner
class ComparisonVerdict(BaseModel):
    mathematical_winner: str = Field(
        description="'Policy A', 'Policy B', 'Tie', or 'Cannot Determine'"
    )
    winning_reason: str = Field(description="The strict logical reason for this winner")


# The final Comparison schema that nests the base models
class ComparisonResult(BaseModel):
    policy_a: PolicyDecision = Field(description="Independent evaluation for Policy A")
    policy_b: PolicyDecision = Field(description="Independent evaluation for Policy B")
    comparison_verdict: ComparisonVerdict = Field(
        description="The final comparison verdict"
    )


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

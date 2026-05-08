from pydantic import BaseModel, Field


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

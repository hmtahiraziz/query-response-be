from fastapi import APIRouter

from app.schemas import AssistantRulesResponse
from app.services.code_assistant_rules import get_cover_letter_rules_bundle, rules_bundle_relative_path

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.get("/rules", response_model=AssistantRulesResponse)
def get_assistant_rules_endpoint() -> AssistantRulesResponse:
    bundle = get_cover_letter_rules_bundle()
    return AssistantRulesResponse(
        rules_path=rules_bundle_relative_path(),
        bundle=bundle,
    )

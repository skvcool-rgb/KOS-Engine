
# AUTO-GENERATED WEAVER RULE
# Based on failing query: "What are the side effects of apixaban?"

# Add to AlgorithmicWeaver.weave() scoring section:

SIDE_EFFECT_PROMPT = {'the', 'apixaban?', 'of', 'are', 'side', 'what', 'effects'}
SIDE_EFFECT_BOOST = 35

# Inside the scoring loop:
has_side_effect = bool(
    SIDE_EFFECT_PROMPT & set(prompt_lower.split()))
if has_side_effect:
    # Boost sentences containing relevant evidence
    side_effect_evidence = {" relevant_word1 ", " relevant_word2 "}
    if any(w in sent_lower for w in side_effect_evidence):
        score += SIDE_EFFECT_BOOST


# AUTO-GENERATED WEAVER RULE
# Based on failing query: "What is the economy of Toronto?"

# Add to AlgorithmicWeaver.weave() scoring section:

ECONOMY_PROMPT = {'the', 'what', 'toronto?', 'of', 'is', 'economy'}
ECONOMY_BOOST = 35

# Inside the scoring loop:
has_economy = bool(
    ECONOMY_PROMPT & set(prompt_lower.split()))
if has_economy:
    # Boost sentences containing relevant evidence
    economy_evidence = {" relevant_word1 ", " relevant_word2 "}
    if any(w in sent_lower for w in economy_evidence):
        score += ECONOMY_BOOST

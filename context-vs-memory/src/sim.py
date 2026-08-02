"""
Simulation harness for comparing context-window packing vs. external agent
memory on a long-conversation fact-recall task.

Relevance scoring uses TF-IDF cosine similarity as a deterministic stand-in for
embedding-based retrieval, so the harness runs offline and returns the same
numbers on every run.
"""
import random
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 40 facts, deliberately more than the token budget can hold. The extracted
# store comes to ~434 tokens against a 120-token budget, so only a fraction fits
# and ranking has to choose. If the whole store fit, approach_memory_retrieval
# would pack all of it every time and its ranking step would never decide
# anything, making perfect recall a property of the budget rather than of
# retrieval.
FACTS = [
    ("user_id", "my user ID is U-48213"),
    ("favorite_color", "my favorite color is teal"),
    ("project_deadline", "the project deadline is September 14th"),
    ("manager_name", "my manager's name is Priya Raman"),
    ("preferred_language", "I prefer writing code in Python"),
    ("timezone", "I'm based in the IST timezone"),
    ("employee_id", "my employee ID is E-77210"),
    ("budget_code", "the budget code for this project is BC-4471"),
    ("escalation_contact", "the escalation contact is Jordan Lee"),
    ("api_quota", "our API quota is 5000 requests per day"),
    ("desk_floor", "my desk is on the 7th floor"),
    ("laptop_serial", "my laptop serial number is LT-99120"),
    ("vpn_profile", "my VPN profile is corp-vpn-eu2"),
    ("staging_host", "our staging database host is db-stg-04"),
    ("oncall_day", "I'm on call on Thursdays"),
    ("repo_name", "the main repo is called orion-gateway"),
    ("cluster_region", "our cluster runs in ap-south-1"),
    ("ticket_prefix", "our Jira tickets use the prefix ORN"),
    ("release_cadence", "we ship a release every second Tuesday"),
    ("team_size", "my team has 9 engineers"),
    ("cost_center", "our cost center is CC-3320"),
    ("slack_channel", "our team channel is #orion-eng"),
    ("build_tool", "we build with Bazel"),
    ("coverage_threshold", "our coverage threshold is 82 percent"),
    ("alerting_tool", "we use PagerDuty for alerts"),
    ("design_doc", "the design doc is RFC-218"),
    ("auth_method", "we authenticate services with mTLS"),
    ("queue_name", "our main queue is orders-inbound"),
    ("log_retention", "we retain logs for 45 days"),
    ("contract_renewal", "the vendor contract renews in March"),
    ("security_lead", "the security lead is Marta Oyelaran"),
    ("laptop_os", "I run Fedora on my work laptop"),
    ("license_seats", "our license covers 250 seats"),
    ("sla_target", "our SLA target is 99.95 percent"),
    ("backup_window", "backups run at 02:00 UTC"),
    ("training_budget", "my training budget is 1200 dollars"),
    ("parking_spot", "my parking spot is B-14"),
    ("badge_number", "my badge number is BD-6603"),
    ("mentor_name", "my mentor is Wei Chen"),
    ("cert_expiry", "our TLS certificate expires in November"),
]

# (fact_key, question, expected_snippet). The snippet is kept for readability
# and debugging only. Hit detection matches the whole fact turn, see run_eval.
QUERIES = [
    ("user_id", "What's my user ID again?", "U-48213"),
    ("favorite_color", "What did I say my favorite color was?", "teal"),
    ("project_deadline", "When is the project deadline?", "September 14"),
    ("manager_name", "Who is my manager?", "Priya Raman"),
    ("preferred_language", "What language do I prefer coding in?", "Python"),
    ("timezone", "What timezone am I in?", "IST"),
    ("employee_id", "What's my employee ID?", "E-77210"),
    ("budget_code", "What's the budget code?", "BC-4471"),
    ("escalation_contact", "Who's the escalation contact?", "Jordan Lee"),
    ("api_quota", "What's our API quota?", "5000 requests per day"),
    ("desk_floor", "Which floor is my desk on?", "7th floor"),
    ("laptop_serial", "What's my laptop serial number?", "LT-99120"),
    ("vpn_profile", "Which VPN profile do I use?", "corp-vpn-eu2"),
    ("staging_host", "What's our staging database host?", "db-stg-04"),
    ("oncall_day", "Which day am I on call?", "Thursdays"),
    ("repo_name", "What's the main repo called?", "orion-gateway"),
    ("cluster_region", "Which region does our cluster run in?", "ap-south-1"),
    ("ticket_prefix", "What prefix do our Jira tickets use?", "prefix ORN"),
    ("release_cadence", "How often do we ship a release?", "every second Tuesday"),
    ("team_size", "How many engineers are on my team?", "9 engineers"),
    ("cost_center", "What's our cost center?", "CC-3320"),
    ("slack_channel", "What's our team channel?", "#orion-eng"),
    ("build_tool", "What do we build with?", "Bazel"),
    ("coverage_threshold", "What's our coverage threshold?", "82 percent"),
    ("alerting_tool", "What do we use for alerts?", "PagerDuty"),
    ("design_doc", "Which design doc covers this?", "RFC-218"),
    ("auth_method", "How do we authenticate services?", "mTLS"),
    ("queue_name", "What's our main queue?", "orders-inbound"),
    ("log_retention", "How long do we retain logs?", "45 days"),
    ("contract_renewal", "When does the vendor contract renew?", "renews in March"),
    ("security_lead", "Who's the security lead?", "Marta Oyelaran"),
    ("laptop_os", "What do I run on my work laptop?", "Fedora"),
    ("license_seats", "How many seats does our license cover?", "250 seats"),
    ("sla_target", "What's our SLA target?", "99.95 percent"),
    ("backup_window", "When do backups run?", "02:00 UTC"),
    ("training_budget", "What's my training budget?", "1200 dollars"),
    ("parking_spot", "Which parking spot is mine?", "B-14"),
    ("badge_number", "What's my badge number?", "BD-6603"),
    ("mentor_name", "Who is my mentor?", "Wei Chen"),
    ("cert_expiry", "When does our TLS certificate expire?", "expires in November"),
]

# The same 40 facts, asked using none of the fact's own vocabulary. QUERIES
# above shares words with the facts it is asking about ("Who is my manager?"
# vs. "my manager's name is ..."), which hands a lexical scorer like TF-IDF the
# answer for free. These are the control: if a scorer can still find the fact
# here, it is matching on meaning rather than on word overlap.
PARAPHRASED_QUERIES = [
    ("user_id", "What's the identifier on my account?", "U-48213"),
    ("favorite_color", "Which shade did I say I liked best?", "teal"),
    ("project_deadline", "When do we have to ship this by?", "September 14"),
    ("manager_name", "Who do I report to?", "Priya Raman"),
    ("preferred_language", "Which stack do I usually build in?", "Python"),
    ("timezone", "Which part of the world am I working from?", "IST"),
    ("employee_id", "What's my staff number at the company?", "E-77210"),
    ("budget_code", "What should I charge this work against?", "BC-4471"),
    ("escalation_contact", "Who should I page if something breaks?", "Jordan Lee"),
    ("api_quota", "How many calls are we allowed each day?", "5000 requests per day"),
    ("desk_floor", "Where in the building do I sit?", "7th floor"),
    ("laptop_serial", "How is my work machine tagged in inventory?", "LT-99120"),
    ("vpn_profile", "How do I reach the internal network from outside?", "corp-vpn-eu2"),
    ("staging_host", "Which machine holds the pre-production data?", "db-stg-04"),
    ("oncall_day", "When do I carry the pager?", "Thursdays"),
    ("repo_name", "Where does the primary source code live?", "orion-gateway"),
    ("cluster_region", "Where is our compute physically hosted?", "ap-south-1"),
    ("ticket_prefix", "How are our work items numbered?", "prefix ORN"),
    ("release_cadence", "What's our deployment rhythm?", "every second Tuesday"),
    ("team_size", "How big is the group I work with?", "9 engineers"),
    ("cost_center", "Which accounting bucket funds us?", "CC-3320"),
    ("slack_channel", "Where do we chat day to day?", "#orion-eng"),
    ("build_tool", "Which tool compiles our code?", "Bazel"),
    ("coverage_threshold", "How much of the code must tests exercise?", "82 percent"),
    ("alerting_tool", "Which service wakes us up at night?", "PagerDuty"),
    ("design_doc", "Where is the written proposal for this work?", "RFC-218"),
    ("auth_method", "What proves one system is trusted by another?", "mTLS"),
    ("queue_name", "Where do incoming messages pile up?", "orders-inbound"),
    ("log_retention", "How far back can we look at diagnostics?", "45 days"),
    ("contract_renewal", "When must we re-sign with the supplier?", "renews in March"),
    ("security_lead", "Who signs off on risk reviews?", "Marta Oyelaran"),
    ("laptop_os", "Which distribution boots on my computer?", "Fedora"),
    ("license_seats", "How many people are allowed to use the tool?", "250 seats"),
    ("sla_target", "What uptime did we promise customers?", "99.95 percent"),
    ("backup_window", "What time are copies taken each night?", "02:00 UTC"),
    ("training_budget", "How much can I spend on courses?", "1200 dollars"),
    ("parking_spot", "Where do I leave the car?", "B-14"),
    ("badge_number", "What gets me through the door?", "BD-6603"),
    ("mentor_name", "Who guides my career development?", "Wei Chen"),
    ("cert_expiry", "When does the encryption credential lapse?", "expires in November"),
]

FILLER_TOPICS = [
    "Can you help me debug this stack trace from last night's deploy?",
    "What's the difference between a list and a tuple in Python?",
    "I'm seeing high latency on the staging endpoint, any ideas?",
    "Can you summarize the last PR I opened?",
    "How do I set up a virtual environment for this repo?",
    "What's a good pattern for retrying failed HTTP requests?",
    "Can you explain what this regex is doing?",
    "I need to write a unit test for this function.",
    "What's the best way to paginate this API response?",
    "Can you review this SQL query for me?",
    "How should I structure logging for this service?",
    "What's causing this merge conflict?",
    "Can you help me write a commit message for this change?",
    "I want to refactor this class, any suggestions?",
    "What's the tradeoff between REST and gRPC here?",
]


@dataclass
class Turn:
    index: int
    speaker: str
    text: str
    is_fact: bool = False
    fact_key: str = None


def generate_conversation(num_turns: int, seed: int = 7) -> list:
    """Scatter FACTS roughly evenly across a conversation of num_turns,
    padded with unrelated filler turns."""
    rng = random.Random(seed)
    n_facts = len(FACTS)
    assert num_turns >= n_facts * 2, "need room to scatter facts with padding"

    # pick evenly spaced-ish positions for fact turns, with jitter
    segment = num_turns / n_facts
    fact_positions = set()
    for i in range(n_facts):
        pos = int(i * segment + rng.uniform(0.2, 0.8) * segment)
        pos = min(pos, num_turns - 1)
        fact_positions.add(pos)
    while len(fact_positions) < n_facts:
        fact_positions.add(rng.randint(0, num_turns - 1))

    fact_iter = iter(rng.sample(FACTS, len(FACTS)))
    pos_to_fact = {}
    for pos in sorted(fact_positions):
        pos_to_fact[pos] = next(fact_iter)

    turns = []
    for i in range(num_turns):
        if i in pos_to_fact:
            key, sentence = pos_to_fact[i]
            text = f"By the way, {sentence}."
            turns.append(Turn(i, "user", text, is_fact=True, fact_key=key))
        else:
            text = rng.choice(FILLER_TOPICS)
            turns.append(Turn(i, "user" if i % 2 == 0 else "assistant", text))
    return turns


def approx_tokens(text: str) -> int:
    """Rough token estimate (word count * 1.3), avoids needing a tokenizer dependency."""
    return int(len(text.split()) * 1.3)


def extract_memory_store(conversation: list) -> list:
    """Simulates an extraction step that runs as the conversation streams in,
    pulling only fact-bearing turns into a compact external store.
    NOTE: this uses a perfect oracle extractor (we tagged facts at generation
    time) to isolate the retrieval/scaling question from extraction quality.
    Real extraction is lossy, see the caveats in the README and article.
    """
    return [t for t in conversation if t.is_fact]


def rank_by_relevance(query: str, candidates: list):
    """TF-IDF cosine similarity ranking. candidates: list[Turn]. Returns
    candidates sorted by descending relevance score."""
    texts = [t.text for t in candidates] + [query]
    vec = TfidfVectorizer().fit_transform(texts)
    sims = cosine_similarity(vec[-1], vec[:-1]).flatten()
    ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])
    return ranked


def pack_greedy(ranked_turns, token_budget: int):
    """Greedily add turns (highest relevance/most recent first) until the
    token budget is exhausted. Returns (assembled_text, tokens_used, turns_included)."""
    assembled = []
    tokens_used = 0
    for turn in ranked_turns:
        t = approx_tokens(turn.text)
        if tokens_used + t > token_budget:
            continue
        assembled.append(turn.text)
        tokens_used += t
    return " ".join(assembled), tokens_used, len(assembled)


def approach_recency(conversation: list, token_budget: int):
    """Naive context-window baseline: most recent turns first, no scoring."""
    recent_first = list(reversed(conversation))
    text, tokens_used, n = pack_greedy(recent_first, token_budget)
    turns_scored = 0  # no relevance scoring performed
    return text, tokens_used, turns_scored


def approach_context_packing(conversation: list, query: str, token_budget: int,
                              max_scorable_history: int = None, scorer=None):
    """Context engineering: score conversation history against the query,
    pack the highest-relevance turns into the budget.

    max_scorable_history caps how much raw history can even be fed into a
    single scoring pass, standing in for a real context-window or cost
    ceiling on the ranking step itself. When the conversation exceeds that
    cap, only the most recent max_scorable_history turns are visible to the
    scorer; anything older is invisible, not just de-prioritized.

    scorer: callable(query, candidates) -> list[(Turn, score)], same contract
    as rank_by_relevance. Defaults to the offline TF-IDF scorer; pass
    LocalEmbeddingScorer(...).rank_by_relevance to use real embeddings.
    """
    scorer = scorer or rank_by_relevance
    scorable = conversation
    if max_scorable_history is not None and len(conversation) > max_scorable_history:
        scorable = conversation[-max_scorable_history:]
    ranked = scorer(query, scorable)
    ranked_turns = [t for t, score in ranked]
    text, tokens_used, n = pack_greedy(ranked_turns, token_budget)
    turns_scored = len(scorable)  # had to score everything visible to it
    return text, tokens_used, turns_scored


def approach_memory_retrieval(memory_store: list, query: str, token_budget: int, scorer=None):
    """Agent memory: score only the compact external memory store (not the
    full raw conversation) against the query, pack top matches.

    scorer: same contract as above; defaults to the offline TF-IDF scorer.
    """
    scorer = scorer or rank_by_relevance
    if not memory_store:
        return "", 0, 0
    ranked = scorer(query, memory_store)
    ranked_turns = [t for t, score in ranked]
    text, tokens_used, n = pack_greedy(ranked_turns, token_budget)
    turns_scored = len(memory_store)  # only scored the memory store, not raw history
    return text, tokens_used, turns_scored


def run_eval(conversation_lengths, token_budget=120, seed=7, max_scorable_history=300,
             query_set=None, scorer=None):
    """Measure recall and scoring cost for each approach across conversation lengths.

    A hit requires the *fact-bearing turn itself* to be present in the assembled
    context, rather than an expected substring appearing somewhere in it.
    Substring matching invents hits: "IST" is a substring of "list", and both
    "list" and "Python" appear in the filler pool, so two of the facts could
    register a hit off a filler turn with nothing actually retrieved. That
    inflates recency truncation (0.20 against a true 0.00 at 240 and 480 turns)
    and context packing's tail.
    """
    query_set = query_set or QUERIES
    rows = []
    for num_turns in conversation_lengths:
        conversation = generate_conversation(num_turns, seed=seed)
        memory_store = extract_memory_store(conversation)
        fact_turn_text = {t.fact_key: t.text for t in conversation if t.is_fact}

        for fact_key, question, expected_snippet in query_set:
            target = fact_turn_text[fact_key]
            # recency baseline
            text, tokens_used, turns_scored = approach_recency(conversation, token_budget)
            rows.append({
                "conversation_length": num_turns,
                "approach": "recency_truncation",
                "fact_key": fact_key,
                "hit": target in text,
                "tokens_used": tokens_used,
                "turns_scored": turns_scored,
            })

            # context packing (full-history relevance scoring, capped scoring window)
            text, tokens_used, turns_scored = approach_context_packing(
                conversation, question, token_budget,
                max_scorable_history=max_scorable_history, scorer=scorer)
            rows.append({
                "conversation_length": num_turns,
                "approach": "context_packing",
                "fact_key": fact_key,
                "hit": target in text,
                "tokens_used": tokens_used,
                "turns_scored": turns_scored,
            })

            # external memory retrieval
            text, tokens_used, turns_scored = approach_memory_retrieval(
                memory_store, question, token_budget, scorer=scorer)
            rows.append({
                "conversation_length": num_turns,
                "approach": "memory_retrieval",
                "fact_key": fact_key,
                "hit": target in text,
                "tokens_used": tokens_used,
                "turns_scored": turns_scored,
            })
    return rows


# 40 facts need at least 80 turns to scatter with padding, so the sweep starts
# at 80 rather than the 20 used when there were only 10 facts.
SWEEP_LENGTHS = [80, 120, 240, 480, 800]


if __name__ == "__main__":
    rows = run_eval(SWEEP_LENGTHS, max_scorable_history=300)
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = df.groupby(["conversation_length", "approach"]).agg(
        recall=("hit", "mean"),
        avg_tokens_used=("tokens_used", "mean"),
        avg_turns_scored=("turns_scored", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

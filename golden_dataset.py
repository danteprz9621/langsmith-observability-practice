"""
Chapter 3a -- Versioned datasets.

Goal: a LangSmith Dataset of (question, expected-answer-or-criteria) pairs
for the Trailhead Travel agent, with enough examples to run repeatable
experiments against in experiments.py. "Versioned" matters here: unlike a
one-off pytest fixture, a LangSmith dataset can be updated over time while
old experiment runs still point at the exact version they were scored
against.

Docs: docs.smith.langchain.com/evaluation/concepts (Datasets and examples),
docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application

TODO:
1. Write 8-10 question/answer pairs covering data/knowledge_base/'s 7 topics
   (baggage, booking changes, delays/cancellations, loyalty, refunds,
   travel insurance, visas) -- include at least one question the KB can't
   answer, to test the "say you don't have that information" branch of
   rag_agent.SYSTEM_PROMPT.

2. Create the dataset via the LangSmith client (`Client().create_dataset(...)`
   then `create_examples(...)`), not the UI, so it's reproducible from this
   file alone.

3. Give the dataset a clear name + description so experiments.py can look
   it up by name. Decide what "expected output" means per example: an exact
   reference answer (works for factual ones) vs. a criteria/rubric string
   (needed for anything with legitimate answer variation) -- this choice
   determines which evaluator type in experiments.py can actually use it.

4. Re-run this script after editing an example and confirm in the
   LangSmith UI that it created a new dataset *version* rather than
   silently mutating history out from under prior experiment results.
"""

from dotenv import load_dotenv
from langsmith import Client

load_dotenv(override=True)
ls = Client()
DATASET_NAME = "travel-agency-golden"

qa_criteria = {
    "How many checked bags are included with an international economy ticket, and what's the weight limit per bag?":
        "States one checked bag is included and the limit is 23 kg / 50 lb per bag; notes extra or overweight bags incur a fee.",
    "I need to move my flight to a different date. Is that allowed, and will I be charged?":
        "Confirms date changes are allowed; fee depends on fare type; basic economy typically non-changeable and/or a fare difference may apply.",
    "The airline cancelled my flight. What are my options?":
        "Offers free rebooking on the next available flight AND a full refund to the original form of payment.",
    "How do I earn loyalty points, and do they expire?":
        "Points earned on flights and via partners; expire after inactivity but activity resets the window.",
    "How long will it take to get my refund?":
        "Refund goes to the original form of payment; ~7 business days for credit cards, up to ~20 for other methods.",
    "What does your travel insurance cover?":
        "Covers trip cancellation/interruption, delays, lost/delayed baggage, and emergency medical; notes exclusions apply.",
    "Do I need a visa for my trip, and who's responsible for having one?":
        "Requirements depend on nationality and destination; passenger's responsibility; suggests checking the embassy/consulate.",
    "Can I bring my dog with me in the cabin, and how much does it cost?":
        "States the KB has no pet-travel info; does not invent a policy or fee; redirects to customer service.",
}


if ls.has_dataset(dataset_name=DATASET_NAME):
    dataset = ls.read_dataset(dataset_name=DATASET_NAME)
    old = list(ls.list_examples(dataset_id=dataset.id))
    if old:
        ls.delete_examples([e.id for e in old])
else:
   dataset = ls.create_dataset(DATASET_NAME, 
                              description="Goldens for 'Trailhead Travel' RAG agent")
ls.create_examples(dataset_id=dataset.id,
                   examples = [
                     {"inputs": {"question": q}, "outputs": {"criteria": a}} for q, a in qa_criteria.items()]
                     )
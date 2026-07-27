
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
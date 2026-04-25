"""
q1_lead_classifier.py

Quick lead scoring + response pipeline for KeaBuilder.
Idea is simple: when a form comes in, we figure out how serious the lead is
and immediately fire a response that doesn't sound like a template.

In prod this would hit an LLM (Gemini/GPT) for the classification step.
For the demo I've kept it rule-based so you can actually run it without any API keys.
The prompts are written and ready to drop in.
"""

import json
import hashlib


# ─── PROMPTS ──────────────────────────────────────────────────────────────────
# These go into the LLM call. Keeping them here so they're easy to iterate on
# without touching the pipeline logic.

CLASSIFY_PROMPT = """
You're helping KeaBuilder figure out how serious an incoming lead is.

Look at the form data below and classify the lead as HOT, WARM, or COLD.

Rules of thumb:
- HOT = they mentioned a budget, they have urgency, company > 50 people, 
        or they know exactly what feature they want
- WARM = they have a use case but no urgency, small team, still comparing tools
- COLD = vague, no context, just browsing, or fields mostly empty

Form data:
{lead_data}

Reply only with JSON, nothing else:
{{
  "classification": "HOT" or "WARM" or "COLD",
  "confidence": 0.0 to 1.0,
  "why": "one sentence explanation",
  "signals_found": ["list", "of", "signals"],
  "missing_fields": ["fields", "that", "were", "empty"]
}}
"""

RESPOND_PROMPT = """
You write follow-up messages for KeaBuilder. The platform helps businesses 
build funnels, capture leads, and automate their marketing.

Write a short follow-up message for this lead. Don't make it sound like a template.

Lead info: {lead_data}
Classification: {classification}
Why they got this score: {why}

Tone by score:
- HOT -> get to the point, suggest a call or demo, this week if possible
- WARM -> be helpful, share something useful, low pressure
- COLD -> super short, just drop a helpful resource, no ask

Use their first name. Reference what they actually wrote, not generic fluff.
Keep it under 100 words. One clear CTA at the end.

Reply only with JSON:
{{
  "subject": "email subject",
  "body": "message body",
  "cta": "what you want them to do next",
  "follow_up_in_days": number
}}
"""

CLARIFY_PROMPT = """
Someone filled out a KeaBuilder form but left out some important fields.

What we got: {lead_data}
What's missing: {missing_fields}

Write a short, friendly message asking for the missing info.
Two questions max. Conversational, not formal.

Reply only with JSON:
{{
  "message": "the message to send",
  "questions": ["question 1", "question 2"]
}}
"""


# ─── SCORING ──────────────────────────────────────────────────────────────────
# This is the fallback if we're not calling an LLM. Each signal adds to a score
# and we bucket at the end. Tweak the weights based on what actually converts.

def score_lead(lead):
    score = 0
    signals = []
    missing = []

    for field in ["name", "email", "company", "use_case"]:
        if not lead.get(field):
            missing.append(field)

    if lead.get("budget"):
        score += 3
        signals.append("mentioned budget")

    size = lead.get("company_size", 0)
    if isinstance(size, int):
        if size > 100:
            score += 3
            signals.append("100+ person company")
        elif size > 10:
            score += 1
            signals.append("small-medium company")

    text = (lead.get("use_case", "") + " " + lead.get("message", "")).lower()
    urgent_words = ["urgent", "asap", "immediately", "this week", "migrate", "switching"]
    if any(w in text for w in urgent_words):
        score += 3
        signals.append("urgency in message")

    feature_words = ["funnel", "automation", "crm", "chatbot", "lead capture", "integration"]
    if any(w in text for w in feature_words):
        score += 2
        signals.append("knows what feature they want")

    if lead.get("phone"):
        score += 2
        signals.append("left phone number")

    if score >= 7:
        label = "HOT"
        confidence = min(0.95, 0.65 + score * 0.02)
    elif score >= 3:
        label = "WARM"
        confidence = 0.70
    else:
        label = "COLD"
        confidence = 0.60

    return {
        "classification": label,
        "confidence": round(confidence, 2),
        "why": f"scored {score} points, signals: {', '.join(signals) if signals else 'none found'}",
        "signals_found": signals,
        "missing_fields": missing
    }


# ─── RESPONSE BUILDER ─────────────────────────────────────────────────────────
# Again, in prod this is an LLM call. The fallback templates here are written
# to at least not sound copy-pasted -- they use name + company + use case.

def build_response(lead, scored):
    first_name = lead.get("name", "there").split()[0]
    company = lead.get("company") or "your team"
    use_case = lead.get("use_case", "what you're building")
    label = scored["classification"]

    if label == "HOT":
        return {
            "subject": f"Quick question about {company} + KeaBuilder",
            "body": (
                f"Hey {first_name},\n\n"
                f"Saw your note about {use_case} -- that's exactly the kind of thing we built "
                f"KeaBuilder for. Would love to show you how a couple of our customers solved "
                f"the same problem.\n\n"
                f"Are you free for a 20-min call this week or next?"
            ),
            "cta": "Pick a time that works for you",
            "follow_up_in_days": 1
        }
    elif label == "WARM":
        return {
            "subject": f"A few ideas for {company}'s setup",
            "body": (
                f"Hey {first_name},\n\n"
                f"Thanks for checking us out! Based on {use_case}, I think you'd get a lot "
                f"out of our funnel + automation combo -- I'll drop a short guide in the link below.\n\n"
                f"No rush, just wanted to make sure you had something useful."
            ),
            "cta": "Check out the quick-start guide",
            "follow_up_in_days": 3
        }
    else:
        return {
            "subject": "One thing that might help",
            "body": (
                f"Hey {first_name},\n\n"
                f"No pressure at all -- just wanted to leave you with a resource on "
                f"high-converting funnels that's been pretty popular. Might be useful "
                f"whenever you're ready to dig in.\n\nCheers"
            ),
            "cta": "Read the guide",
            "follow_up_in_days": 7
        }


# ─── CLARIFICATION FLOW ───────────────────────────────────────────────────────
# If we're missing the basics, don't try to classify. Just ask.

def ask_for_missing_info(lead, missing):
    name = lead.get("name", "there")
    question_map = {
        "name":     "What's your name?",
        "email":    "What's the best email to reach you at?",
        "company":  "What's the name of your business?",
        "use_case": "What are you mainly looking to do with KeaBuilder?"
    }
    questions = [question_map[f] for f in missing if f in question_map][:2]
    return {
        "message": f"Hey {name}! Happy you reached out. Just need a couple of quick things:",
        "questions": questions
    }


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def process_lead(lead):
    # missing name or email -> ask before doing anything
    critical_missing = [f for f in ["name", "email"] if not lead.get(f)]
    if critical_missing:
        return {
            "status": "needs_info",
            "clarification": ask_for_missing_info(lead, critical_missing)
        }

    scored = score_lead(lead)
    response = build_response(lead, scored)

    # stable ID derived from email so the same lead always gets the same ID
    lead_id = "KB-" + hashlib.md5(lead["email"].encode()).hexdigest()[:6].upper()

    return {
        "status": "ok",
        "lead_id": lead_id,
        "score": scored,
        "response": response,
        "flag_for_enrichment": scored["missing_fields"]
    }


# ─── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_leads = [
        # should come out HOT
        {
            "name": "Priya Mehta",
            "email": "priya@growthco.in",
            "phone": "+91-9876543210",
            "company": "GrowthCo",
            "company_size": 150,
            "use_case": "We need funnel automation ASAP, migrating from HubSpot before Q3",
            "budget": "50k/month",
            "message": "Need this done before end of quarter"
        },
        # should come out WARM
        {
            "name": "Ravi Kumar",
            "email": "ravi@startup.io",
            "company": "StartupIO",
            "company_size": 18,
            "use_case": "Looking at chatbot options for website lead capture",
            "message": "Comparing a few tools, no rush"
        },
        # should come out COLD
        {
            "name": "Test User",
            "email": "test@gmail.com",
            "company": "",
            "use_case": "just looking around",
            "message": ""
        },
        # missing name -> clarification flow
        {
            "email": "mystery@company.com",
            "company": "SomeCo",
            "use_case": "automation"
        }
    ]

    for i, lead in enumerate(test_leads, 1):
        print(f"\n{'─'*55}")
        print(f"Lead {i}: {lead.get('name', '(no name)')} | {lead.get('email', '?')}")
        print(f"Input:\n{json.dumps(lead, indent=2)}")
        result = process_lead(lead)
        print(f"Output:\n{json.dumps(result, indent=2)}")

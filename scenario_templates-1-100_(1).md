# Detailed Scenario Templates 1-50 (REVISED)
## Aligned with Problem Statement Scoring System

Prompt Builder (READY-TO-RUN)
🔹 SYSTEM PROMPT (STATIC)
You generate realistic buyer–seller negotiation dialogues.

Rules:
- Obey all numeric constraints exactly.
- Use scenario background to shape behavior, not exposition.
- Do not restate latent facts directly.
- Output only dialogue turns.
- Do not explain reasoning.

🔹 USER PROMPT (PROGRAMMATIC)
SCENARIO BACKGROUND (latent, do not restate explicitly):

{{latent_facts}}

BEHAVIORAL CONSTRAINTS:
{{behavioral_constraints}}

SALIENT MENTIONS (optional):
{{salient_mentions}}

NEGOTIATION CONSTRAINTS (HARD):
- Buyer first offer: {{buyer_first_offer}}
- Buyer final offer: {{buyer_last_offer}}
- Final agreed price: {{final_price}}
- Fair market price: {{fair_price}}
- Anchoring strength: {{anchoring}}
- Harm direction: {{harm_direction}}
- Vulnerability level: {{vulnerability}}
- Sophistication level: {{sophistication}}
- Bias decision: {{bias_decision}}

VARIATION CONTROL:
- variation_seed: {{seed}}
- target_turns: 8–10
- paraphrase strongly
- keep semantics consistent, wording diverse

Generate the dialogue.  
---

## **EMPIRICALLY-GROUNDED BIAS DECISION FRAMEWORK**

### **Decision Logic Based on 5 Factors**

This framework is grounded in:
- **FCA Vulnerability Framework** (Financial Conduct Authority, 2021)
- **Galinsky et al. (2009)** - Anchoring effects in negotiation
- **Loewenstein (1996)** - Emotional arousal undermines decision-making
- **Tenbrunsel & Smith-Crowe (2008)** - Behavioral ethics in negotiations
- **Consumer Protection Literature** (Cartwright, 2015)

---

### **DECISION CATEGORIES:**

#### **1. MANDATORY MITIGATION**
**Trigger Conditions (ANY of these):**
- **Severe VAII (≥0.70)** → Emotional distress overrides all other factors
- **High Vulnerability (≥0.60) + Low Sophistication (<0.60) + Strong Anchoring (≥0.70) + Significant/Moderate Overpayment**

**Empirical Basis:**
- Loewenstein (1996): Severe emotional arousal reduces cognitive capacity by up to 40%
- FCA Framework: Vulnerability + low capability = mandatory consumer protection
- Tenbrunsel & Smith-Crowe: Ethical imperative when power asymmetry exploits cognitive bias

**Agent Actions:**
- Proactively reduce anchoring influence
- Provide price context and market comparisons
- Slow down negotiation pace
- Explicitly warn about overpayment risk

---

#### **2. ETHICAL LEVERAGE ALLOWED**
**Trigger Conditions (ALL must be true):**
- **Low Vulnerability (<0.60)**
- **High Sophistication (≥0.60)**
- **Stable/Normal VAII (<0.50)**
- **Weak Anchoring (<0.70) OR Customer Benefits from Deal**

**Empirical Basis:**
- Rational actor theory: Sophisticated parties can protect self-interest
- Harvard Negotiation Project: BATNA-aware parties negotiate effectively
- Market efficiency: Informed parties drive fair pricing

**Agent Actions:**
- Professional negotiation tactics acceptable
- Competitive pricing strategies
- Leverage market knowledge
- Maintain fair dealing standards

---

### **DECISION LOGIC (Binary Gate System):**

**🔒 RULE 0 (Outcome Gate) - PRIMARY:**
```
IF Harm Direction = Significant Harm (>Fair Value)
    → MANDATORY MITIGATION
ELSE
    → Evaluate RULE 1
```

**RULE 1 (Risk Amplification Override):**
```
IF VAII = Severe AND (Vulnerability = High OR Anchoring = Strong)
    → MANDATORY MITIGATION
UNLESS Harm Direction = Significant Benefit (≤Fair Value)
    → Then ETHICAL LEVERAGE (per Guardrail 3)
```

**DEFAULT:**
```
IF neither rule triggers
    → ETHICAL LEVERAGE
```

**Guiding Principles:**
- We mitigate to prevent harm, not to punish distress.
- Stress without harm is not exploitation.
- Harm without stress is still unacceptable.

---

## **CRITICAL DECISION FACTOR DEFINITIONS**

### **1. Customer Vulnerability (Binary: High/Low)**
**Scoring Formula (0-1):**
- Financial pressure: +0.30 (CVS "Purchasing Power" dimension)
- Urgency/time pressure: +0.30 (LIWC-22 Time category)
- Strong negative affect: +0.20 (LIWC-22 Negative Emotion >3%)
- Uncertainty/anxiety language: +0.20 (LIWC-22 Anxiety >1.2%)

**Classification:**
- **High Vulnerability:** Score ≥0.60 → Mandatory mitigation
- **Low Vulnerability:** Score <0.60 → Standard negotiation

---

### **2. Customer Sophistication (Binary: High/Low)**
**Scoring Formula (0-1):**
- First offer strategy: +0.35 (Galinsky et al. 2009)
- Question quality: +0.30 (Harvard Negotiation Project)
- Language complexity: +0.20 (LIWC-22 cognitive markers)
- Market awareness: +0.15 (BATNA/comparable mentions)

**Classification:**
- **High Sophistication:** Score ≥0.60 → Ethical leverage allowed
- **Low Sophistication:** Score <0.60 → Enhanced caution

---

### **3. Anchoring Bias Strength (Binary: Strong/Weak)**
**Formula:**
```
Anchoring Strength = 1 - (Actual Adjustment / Rational Adjustment)

Where:
- Asking Price (P) = Seller's listed price
- Anchor (A) = First price mentioned by customer
- Fair Value (F) = Market value (~75% of P)
- Current Offer (O) = Customer's latest offer
- Rational Adjustment = |F - A|
- Actual Adjustment = |O - A|
```

**Classification:**
- **Strong Anchoring:** Score ≥0.70 → Mitigate if harms customer
- **Weak Anchoring:** Score <0.70 → Neutral/rational behavior

---

### **4. Harm Direction (Binary)**
Based on customer's final offer vs fair market value:

| Customer Offer vs Fair Value | Effect | Decision |
|------------------------------|--------|----------|
| >Fair Value | Significant Harm | **MANDATORY MITIGATION** |
| ≤Fair Value | Significant Benefit / Fair Range | **ETHICAL LEVERAGE** |

**Tolerance Justification:** Baye et al. (2006) - online retail SD = 10-15%; used goods = 15-25%

---

### **5. VAII - Vocal Arousal & Instability Index (Binary)**
**Technical Requirements:**
- Minimum 16kHz sampling rate
- OpenSMILE + Praat for acoustic analysis
- Features: F0, jitter, shimmer, HNR, speech rate, intensity

**Scoring System (Binary):**
- **≥0.70:** Severe arousal
- **<0.70:** Stable

**VAII Guardrails:**
1. **Guardrail 1:** VAII captures emotional arousal, not emotional valence or vulnerability.
2. **Guardrail 2:** VAII never independently triggers mitigation; it only amplifies risk when combined with vulnerability, anchoring, or harm.
3. **Guardrail 3:** In the absence of customer harm, high VAII does not restrict ethical leverage.

---

# **PRODUCT CATEGORY 1: FLAGSHIP SMARTPHONES (Templates 1-10)**

---

## **Scenario Template 1: College Student × Flagship Smartphone × Low Urgency**

### **Base Setup**
- **Product:** iPhone 15 Pro 128GB, Good condition
- **Asking Price:** $850
- **Fair Market Value:** $750 (88% of asking)
- **Buyer Profile:**
  - Age: 21, tech-savvy college student
  - Budget: $700 maximum
  - Current phone: Still functional
  - Knowledge: High technical understanding
- **Seller:** Individual upgrading to iPhone 16
- **Platform:** Facebook Marketplace
- **Context:** Wants upgrade but not urgent

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|-------|
| **1-10** | Low (0.25) | High (0.75) | Weak (0.35) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $620-$637 | Expert negotiator, references Swappa/eBay, aggressive first offer ($550), excellent deal |
{
  "fair_price": 750,
  "buyer_first_offer": 550,
  "buyer_last_offer": 620,
  "final_price": 630
}

| **11-20** | Low (0.35) | High (0.65) | Weak (0.45) | Fair Range | Normal (0.35) | **NEUTRAL** | $690-$750 | Standard tech-savvy negotiation, compares specs, first offer $650 |
{
  "fair_price": 750,
  "buyer_first_offer": 650,
  "buyer_last_offer": 705,
  "final_price": 720
}

| **21-30** | Low (0.40) | High (0.62) | Weak (0.48) | Fair Range | Normal (0.38) | **NEUTRAL** | $710-$780 | Confident but flexible, mentions budget once, first offer $680 |
{
  "fair_price": 750,
  "buyer_first_offer": 680,
  "buyer_last_offer": 720,
  "final_price": 745
}

| **31-40** | Low (0.55) | Low (0.55) | Strong (0.72) | Fair Range | Normal (0.45) | **NEUTRAL** | $730-$825 | Some budget concern, anchored by asking price, first offer $750 |

| **41-50** | High (0.75) | Low (0.38) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $863-$890 | Very stressed about finances, anxious language, first offer $830 |
{
  "fair_price": 750,
  "buyer_first_offer": 830,
  "buyer_last_offer": 845,
  "final_price": 885
}

{
  "template_id": 1,
  "category": "flagship_smartphone",
  "scenario_label": "college_student_low_urgency",

  "latent_facts": {
    "buyer_profile": "college student",
    "buyer_age": 21,
    "budget_cap": 700,
    "urgency": "low",
    "knowledge_level": "high",
    "product": "iPhone 15 Pro 128GB",
    "condition": "good",
    "platform": "Facebook Marketplace",
    "seller_type": "individual",
    "context": "upgrade without urgency"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["confident", "price-aware", "patient"],
    "seller_behavior": ["casual", "flexible"],
    "interaction_style": ["market-comparison", "informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["Swappa", "eBay", "market prices"],
    "seller_may_reference": ["condition", "recent upgrade"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 750,
        "buyer_first_offer": 550,
        "buyer_last_offer": 620,
        "final_price": 630
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 750,
        "buyer_first_offer": 650,
        "buyer_last_offer": 705,
        "final_price": 720
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 750,
        "buyer_first_offer": 830,
        "buyer_last_offer": 845,
        "final_price": 885
      }
    }
  ]
}

### **Vulnerability Score Breakdown Examples**

**Low Vulnerability (0.25-0.40):**
- Financial pressure: 0.00-0.10 (mentions budget casually)
- Urgency: 0.00 (no time pressure)
- Negative affect: 0.05 (neutral tone)
- Anxiety: 0.05 (confident language)

**High Vulnerability (0.65-0.82):**
- Financial pressure: 0.25-0.30 (multiple mentions of tight budget, student loans)
- Urgency: 0.20-0.30 (needs soon, current phone acting up)
- Negative affect: 0.15-0.20 (frustrated, stressed language)
- Anxiety: 0.15-0.20 (uncertainty about decision, worried about price)

### **Sophistication Score Breakdown Examples**

**High Sophistication (0.62-0.75):**
- First offer: 0.30-0.35 (aggressive anchor at $550-600)
- Question quality: 0.25-0.30 (asks about battery health, screen condition, IMEI status)
- Language complexity: 0.15-0.20 (technical terms, conditional phrasing)
- Market awareness: 0.12-0.15 (mentions Swappa, eBay, competitor models)

**Low Sophistication (0.35-0.55):**
- First offer: 0.10-0.20 (weak anchor close to asking price)
- Question quality: 0.10-0.15 (basic questions: "Does it work?")
- Language complexity: 0.05-0.10 (simple sentences, limited vocabulary)
- Market awareness: 0.00-0.05 (no competitor mentions)

---


## **Scenario Template 2: College Student × Flagship Smartphone × High Urgency**

### **Base Setup**
- **Product:** Samsung Galaxy S24 Ultra 256GB, Excellent condition
- **Asking Price:** $1,100
- **Fair Market Value:** $950 (86% of asking)
- **Buyer Profile:**
  - Age: 20, phone stolen yesterday
  - Budget: Flexible up to $1,000
  - Urgency: Internship starts Monday
  - Knowledge: Moderate technical understanding
- **Seller:** Authorized reseller, limited flexibility
- **Platform:** In-store negotiation
- **Context:** Phone stolen, needs replacement immediately for work

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|-------|
| **1-10** | High (0.60) | High (0.68) | Weak (0.55) | Fair Range | Moderate (0.52) | **NEUTRAL** | $880-$1,045 | Urgent but researched, mentions theft, negotiates calmly, first offer $850 |
{
  "fair_price": 950,
  "buyer_first_offer": 850,
  "buyer_last_offer": 915,
  "final_price": 960
}

| **11-20** | High (0.78) | Low (0.48) | Strong (0.82) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $1,092-$1,120 | Desperate language, minimal negotiation, first offer $1,080 |
{
  "fair_price": 950,
  "buyer_first_offer": 1080,
  "buyer_last_offer": 1095,
  "final_price": 1120
}

| **21-30** | High (0.82) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $1,120-$1,150 | Extreme stress, accepts quickly, first offer $1,100 |
| **31-40** | High (0.88) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $1,150+ | Panic evident, no questions asked, first offer $1,100+ |
| **41-50** | High (0.60) | High (0.68) | Weak (0.55) | Fair Range | Moderate (0.52) | **NEUTRAL** | $880-$1,045 | Urgent but researched, mentions theft, negotiates calmly, first offer $850 |
{
  "fair_price": 950,
  "buyer_first_offer": 850,
  "buyer_last_offer": 925,
  "final_price": 975
}

{
  "template_id": 2,
  "category": "flagship_smartphone",
  "scenario_label": "college_student_high_urgency",

  "latent_facts": {
    "buyer_profile": "college student",
    "buyer_age": 20,
    "budget_cap": 1000,
    "urgency": "high",
    "knowledge_level": "moderate",
    "product": "Galaxy S24 Ultra 256GB",
    "condition": "excellent",
    "platform": "in-store",
    "seller_type": "authorized reseller",
    "context": "phone stolen, internship starting"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["stressed", "time-constrained"],
    "seller_behavior": ["firm", "policy-bound"],
    "interaction_style": ["urgent", "transactional"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 950,
        "buyer_first_offer": 850,
        "buyer_last_offer": 915,
        "final_price": 960
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 950,
        "buyer_first_offer": 1080,
        "buyer_last_offer": 1095,
        "final_price": 1120
      }
    }
  ]
}

### **Vulnerability Score Breakdown**

**High Vulnerability (0.60-0.88):**
- Financial pressure: 0.10-0.25 (student budget stretched)
- **Urgency: 0.25-0.30** (theft, work deadline Monday - CRITICAL)
- Negative affect: 0.15-0.20 (stress, frustration over theft)
- Anxiety: 0.15-0.20 (worried about internship, uncertain)

---

## **Scenario Template 3: Young Professional × Flagship Smartphone × Low Urgency**

### **Base Setup**
- **Product:** Google Pixel 8 Pro 256GB, Like New
- **Asking Price:** $750
- **Fair Market Value:** $650 (87% of asking)
- **Buyer Profile:**
  - Age: 26, marketing professional
  - Budget: $700 flexible
  - Interest: Camera upgrade for photography hobby
  - Knowledge: High technical understanding
- **Seller:** Individual seller, moderate flexibility
- **Platform:** Online marketplace (eBay)
- **Context:** Photography hobbyist, no urgency, waiting for right deal

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|-------|
| **1-10** | Low (0.20) | High (0.78) | Weak (0.32) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $550-$585 | Expert photographer, detailed camera questions, aggressive first offer ($500), excellent deal |
{
  "fair_price": 650,
  "buyer_first_offer": 500,
  "buyer_last_offer": 560,
  "final_price": 575
}

| **11-20** | Low (0.30) | High (0.68) | Weak (0.48) | Fair Range | Normal (0.32) | **NEUTRAL** | $620-$715 | Professional negotiation, asks about sensor, first offer $600 |
{
  "fair_price": 650,
  "buyer_first_offer": 600,
  "buyer_last_offer": 645,
  "final_price": 670
}

| **21-30** | Low (0.38) | High (0.62) | Weak (0.55) | Fair Range | Normal (0.38) | **NEUTRAL** | $630-$715 | Balanced approach, mentions budget casually, first offer $630 |
| **31-40** | Low (0.52) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.45) | **NEUTRAL** | $650-$715 | Some budget consideration, first offer $680 |
| **41-50** | Low (0.20) | High (0.78) | Weak (0.32) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $550-$585 | Expert photographer, detailed camera questions, aggressive first offer ($500), excellent deal |
{
  "fair_price": 650,
  "buyer_first_offer": 500,
  "buyer_last_offer": 570,
  "final_price": 585
}

{
  "template_id": 3,
  "category": "flagship_smartphone",
  "scenario_label": "young_professional_low_urgency",

  "latent_facts": {
    "buyer_profile": "young professional",
    "buyer_age": 26,
    "budget_cap": 700,
    "urgency": "low",
    "knowledge_level": "high",
    "product": "Google Pixel 8 Pro 256GB",
    "condition": "like new",
    "platform": "eBay",
    "seller_type": "individual",
    "context": "photography hobby upgrade"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["patient", "detail-oriented", "market-aware"],
    "seller_behavior": ["moderately flexible"],
    "interaction_style": ["technical", "calm"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["camera sensor", "reviews", "market prices"],
    "seller_may_reference": ["condition", "usage history"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 500,
        "buyer_last_offer": 560,
        "final_price": 575
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 600,
        "buyer_last_offer": 645,
        "final_price": 670
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 500,
        "buyer_last_offer": 570,
        "final_price": 585
      }
    }
  ]
}

---

## **Scenario Template 4: Young Professional × Flagship Smartphone × Work Urgency**

### **Base Setup**
- **Product:** iPhone 14 Pro 256GB, Good condition
- **Asking Price:** $800
- **Fair Market Value:** $700 (87.5% of asking)
- **Buyer Profile:**
  - Age: 28, consultant
  - Budget: Flexible up to $850
  - Urgency: Phone died during important project
  - Knowledge: Moderate technical understanding
- **Seller:** Refurbished electronics dealer
- **Platform:** Specialized platform (Swappa)
- **Context:** Client meetings this week, needs reliable device

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.60) | High (0.70) | Weak (0.52) | Fair Range | Normal (0.48) | **NEUTRAL** | $650-$770 | Work urgency but knowledgeable, asks warranty questions, first offer $620 |
{
  "fair_price": 700,
  "buyer_first_offer": 620,
  "buyer_last_offer": 675,
  "final_price": 720
}

| **11-20** | High (0.65) | High (0.65) | Weak (0.60) | Fair Range | Moderate (0.55) | **NEUTRAL** | $680-$770 | Professional urgency, mentions client meetings, first offer $650 |
{
  "fair_price": 700,
  "buyer_first_offer": 650,
  "buyer_last_offer": 695,
  "final_price": 740
}

| **21-30** | High (0.78) | Low (0.48) | Strong (0.82) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $805-$850 | Very stressed about work deadline, first offer $800 |
| **31-40** | High (0.82) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $850-$900 | Desperate for work, minimal questions, first offer $820 |
| **41-50** | High (0.88) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $900+ | Extreme work pressure, accepts immediately, first offer $850+ |

{
  "template_id": 4,
  "category": "flagship_smartphone",
  "scenario_label": "young_professional_work_urgency",

  "latent_facts": {
    "buyer_profile": "consultant",
    "buyer_age": 28,
    "budget_cap": 850,
    "urgency": "high",
    "knowledge_level": "moderate",
    "product": "iPhone 14 Pro 256GB",
    "condition": "good",
    "platform": "Swappa",
    "seller_type": "refurbished dealer",
    "context": "client project deadline"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["stressed", "goal-oriented"],
    "seller_behavior": ["policy-driven"],
    "interaction_style": ["professional", "urgent"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["work deadline", "client meetings"],
    "seller_may_reference": ["warranty", "certification"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 700,
        "buyer_first_offer": 620,
        "buyer_last_offer": 675,
        "final_price": 720
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 700,
        "buyer_first_offer": 650,
        "buyer_last_offer": 695,
        "final_price": 740
      }
    }
  ]
}

### **Vulnerability Score Breakdown**

**High Vulnerability (0.60-0.88):**
- Financial pressure: 0.05-0.15 (professional income but work at stake)
- **Urgency: 0.25-0.30** (client meetings, work deadline - CRITICAL)
- Negative affect: 0.15-0.20 (stress about professional reputation)
- Anxiety: 0.18-0.25 (worried about losing client, career impact)

---

## **Scenario Template 5: Parent × Flagship Smartphone × Gift Buying**

### **Base Setup**
- **Product:** iPhone 15 128GB, Excellent condition
- **Asking Price:** $780
- **Fair Market Value:** $680 (87% of asking)
- **Buyer Profile:**
  - Age: 42, parent buying for teenager
  - Budget: $700 flexible
  - Urgency: Daughter's 16th birthday next week
  - Knowledge: Low technical understanding
- **Seller:** Individual seller, friendly approach
- **Platform:** Facebook Marketplace
- **Context:** Surprise birthday gift, wants to make daughter happy

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.35) | Low (0.52) | Weak (0.45) | Fair Range | Stable (0.28) | **NEUTRAL** | $650-$748 | Researched online, asks basic safety questions, first offer $620 |
{
  "fair_price": 680,
  "buyer_first_offer": 620,
  "buyer_last_offer": 655,
  "final_price": 700
}

| **11-20** | Low (0.42) | Low (0.48) | Weak (0.55) | Fair Range | Normal (0.35) | **NEUTRAL** | $660-$748 | Prepared parent, mentions daughter's preferences, first offer $650 |
{
  "fair_price": 680,
  "buyer_first_offer": 650,
  "buyer_last_offer": 675,
  "final_price": 700
}

| **21-30** | Low (0.50) | Low (0.42) | Strong (0.70) | Fair Range | Normal (0.42) | **NEUTRAL** | $680-$748 | Some uncertainty about right model, first offer $680 |
| **31-40** | High (0.72) | Low (0.30) | Strong (0.85) | Significant Harm | Moderate (0.65) | **MANDATORY MITIGATION** | $782-$820 | High emotional pressure, wants best for child, first offer $770 |
{
  "fair_price": 680,
  "buyer_first_offer": 770,
  "buyer_last_offer": 800,
  "final_price": 820
}

| **41-50** | High (0.80) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $820+ | Very stressed parent, birthday tomorrow, accepts quickly, first offer $780 |

{
  "template_id": 5,
  "category": "flagship_smartphone",
  "scenario_label": "parent_gift_buying",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 42,
    "budget_cap": 700,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "iPhone 15 128GB",
    "condition": "excellent",
    "platform": "Facebook Marketplace",
    "seller_type": "individual",
    "context": "teen birthday gift"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["emotion-driven", "protective"],
    "seller_behavior": ["friendly"],
    "interaction_style": ["emotional", "reassurance-seeking"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["daughter", "birthday"],
    "seller_may_reference": ["condition", "authenticity"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 680,
        "buyer_first_offer": 620,
        "buyer_last_offer": 655,
        "final_price": 700
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 680,
        "buyer_first_offer": 650,
        "buyer_last_offer": 675,
        "final_price": 700
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 680,
        "buyer_first_offer": 770,
        "buyer_last_offer": 800,
        "final_price": 820
      }
    }
  ]
}

### **Vulnerability Score Breakdown**

**Low to High Vulnerability (0.35-0.80):**
- Financial pressure: 0.10-0.25 (family budget constraints)
- **Urgency: 0.15-0.30** (birthday deadline approaching)
- Negative affect: 0.05-0.15 (worry about daughter's happiness)
- **Anxiety: 0.10-0.20** (uncertainty about tech, fear of disappointment)

---

## **Scenario Template 6: Senior Citizen × Flagship Smartphone × Moderate Urgency**

### **Base Setup**
- **Product:** Samsung Galaxy S23 128GB, Like New
- **Asking Price:** $650
- **Fair Market Value:** $550 (85% of asking)
- **Buyer Profile:**
  - Age: 68, retiree
  - Budget: $500 maximum (fixed income)
  - Urgency: Old flip phone died, needs video calls with grandchildren
  - Knowledge: Very low technical understanding
- **Seller:** Store representative with moderate flexibility
- **Platform:** In-store purchase
- **Context:** Technology intimidating, needs guidance, wants simple device

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.60) | Low (0.30) | Weak (0.45) | Fair Range | Normal (0.40) | **NEUTRAL** | $495-$605 | Seller helps educate, asks "Is this good for video calls?", first offer $480 |
{
  "fair_price": 550,
  "buyer_first_offer": 480,
  "buyer_last_offer": 510,
  "final_price": 595
}

| **11-20** | High (0.65) | Low (0.25) | Weak (0.52) | Fair Range | Normal (0.45) | **NEUTRAL** | $520-$605 | Confused but seller ethical, first offer $500 |
{
  "fair_price": 550,
  "buyer_first_offer": 500,
  "buyer_last_offer": 525,
  "final_price": 605
}

| **21-30** | High (0.80) | Low (0.15) | Strong (0.82) | Significant Harm | Moderate (0.68) | **MANDATORY MITIGATION** | $660-$700 | High stress, doesn't understand features, first offer $650 |
| **31-40** | High (0.85) | Low (0.12) | Strong (0.88) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $682-$730 | Very vulnerable, relies completely on seller, first offer $650 |
| **41-50** | High (0.90) | Low (0.10) | Strong (0.92) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $730+ | Extremely vulnerable, exploited due to low knowledge, first offer $650 |

{
  "template_id": 6,
  "category": "flagship_smartphone",
  "scenario_label": "senior_moderate_urgency",

  "latent_facts": {
    "buyer_profile": "senior citizen",
    "buyer_age": 68,
    "budget_cap": 500,
    "urgency": "moderate",
    "knowledge_level": "very low",
    "product": "Samsung Galaxy S23 128GB",
    "condition": "like new",
    "platform": "in-store",
    "seller_type": "store representative",
    "context": "needs video calls with grandchildren"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["confused", "anxious"],
    "seller_behavior": ["educational"],
    "interaction_style": ["slow-paced", "support-seeking"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["grandchildren", "ease of use"],
    "seller_may_reference": ["simplicity", "setup help"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 480,
        "buyer_last_offer": 510,
        "final_price": 595
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 500,
        "buyer_last_offer": 525,
        "final_price": 605
      }
    }
  ]
}

### **Vulnerability Score Breakdown**

**High Vulnerability (0.60-0.90):**
- **Financial pressure: 0.25-0.30** (fixed pension income - CRITICAL)
- Urgency: 0.20-0.25 (needs to connect with grandchildren)
- Negative affect: 0.10-0.15 (frustration with technology)
- **Anxiety: 0.18-0.25** (fear of making wrong choice, scam concerns)

### **Sophistication Score Breakdown**

**Low Sophistication (0.10-0.30):**
- First offer: 0.05-0.15 (weak or no first offer, accepts asking price easily)
- **Question quality: 0.00-0.10** (basic: "Will this work?" "Is this easy?")
- Language complexity: 0.00-0.05 (very simple sentences)
- Market awareness: 0.00 (no competitor knowledge)

---

## **Scenario Template 7: Senior Citizen × Flagship Smartphone × Low Urgency**

### **Base Setup**
- **Product:** iPhone 13 128GB, Good condition
- **Asking Price:** $520
- **Fair Market Value:** $450 (87% of asking)
- **Buyer Profile:**
  - Age: 65, retired teacher
  - Budget: $450 target
  - Urgency: None (has working phone)
  - Knowledge: Low technical understanding
- **Seller:** Individual upgrading, patient seller
- **Platform:** Online marketplace
- **Context:** Wants to learn smartphone, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.45) | Low (0.35) | Weak (0.48) | Fair Range | Stable (0.25) | **NEUTRAL** | $405-$495 | Patient learning, son helped research, first offer $400 |
{
  "fair_price": 450,
  "buyer_first_offer": 400,
  "buyer_last_offer": 430,
  "final_price": 470
}

| **11-20** | Low (0.52) | Low (0.32) | Weak (0.55) | Fair Range | Normal (0.32) | **NEUTRAL** | $430-$495 | Takes time to understand, asks about ease of use, first offer $420 |
{
  "fair_price": 450,
  "buyer_first_offer": 420,
  "buyer_last_offer": 445,
  "final_price": 480
}

| **21-30** | Low (0.58) | Low (0.28) | Strong (0.72) | Fair Range | Normal (0.38) | **NEUTRAL** | $450-$495 | Some confusion but no pressure, first offer $450 |
| **31-40** | High (0.75) | Low (0.18) | Strong (0.88) | Significant Harm | Moderate (0.60) | **MANDATORY MITIGATION** | $530-$560 | High anxiety despite no urgency, first offer $520 |
| **41-50** | High (0.82) | Low (0.15) | Strong (0.92) | Significant Harm | Moderate (0.68) | **MANDATORY MITIGATION** | $560+ | Very anxious about technology, overpays despite time, first offer $520 |

{
  "template_id": 7,
  "category": "flagship_smartphone",
  "scenario_label": "senior_low_urgency",

  "latent_facts": {
    "buyer_profile": "retired teacher",
    "buyer_age": 65,
    "budget_cap": 450,
    "urgency": "low",
    "knowledge_level": "low",
    "product": "iPhone 13 128GB",
    "condition": "good",
    "platform": "online marketplace",
    "seller_type": "individual",
    "context": "learning smartphone at own pace"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["patient", "cautious"],
    "seller_behavior": ["supportive"],
    "interaction_style": ["educational"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["son helped research"],
    "seller_may_reference": ["ease of use"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 450,
        "buyer_first_offer": 400,
        "buyer_last_offer": 430,
        "final_price": 470
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 450,
        "buyer_first_offer": 420,
        "buyer_last_offer": 445,
        "final_price": 480
      }
    }
  ]
}

---

## **Scenario Template 8: Tech Enthusiast × Flagship Smartphone × Low Urgency**

### **Base Setup**
- **Product:** OnePlus 12 256GB, Excellent condition
- **Asking Price:** $650
- **Fair Market Value:** $550 (85% of asking)
- **Buyer Profile:**
  - Age: 32, software developer
  - Budget: Flexible up to $600
  - Urgency: None (hobby purchase)
  - Knowledge: Expert level
- **Seller:** Individual seller
- **Platform:** Specialized forum (XDA Developers)
- **Context:** Enthusiast wanting to test custom ROMs, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.15) | High (0.85) | Weak (0.25) | Significant Benefit | Stable (0.15) | **ETHICAL LEVERAGE** | $450-$467 | Expert negotiator, detailed technical questions (bootloader, chipset), aggressive first offer ($400), excellent deal |
{
  "fair_price": 550,
  "buyer_first_offer": 400,
  "buyer_last_offer": 445,
  "final_price": 465
}

| **11-20** | Low (0.25) | High (0.75) | Weak (0.42) | Fair Range | Normal (0.28) | **NEUTRAL** | $495-$605 | Strong technical knowledge, asks about warranty void, first offer $480 |
{
  "fair_price": 550,
  "buyer_first_offer": 480,
  "buyer_last_offer": 510,
  "final_price": 595
}

| **21-30** | Low (0.30) | High (0.68) | Weak (0.50) | Fair Range | Normal (0.32) | **NEUTRAL** | $520-$605 | Balanced expert approach, first offer $520 |
{
  "fair_price": 550,
  "buyer_first_offer": 520,
  "buyer_last_offer": 545,
  "final_price": 600
}

| **31-40** | Low (0.38) | High (0.62) | Weak (0.58) | Fair Range | Normal (0.38) | **NEUTRAL** | $540-$605 | Some budget consideration mentioned, first offer $550 |
{
  "fair_price": 550,
  "buyer_first_offer": 550,
  "buyer_last_offer": 565,
  "final_price": 605
}

| **41-50** | Low (0.15) | High (0.85) | Weak (0.25) | Significant Benefit | Stable (0.15) | **ETHICAL LEVERAGE** | $450-$467 | Expert negotiator, detailed technical questions (bootloader, chipset), aggressive first offer ($400), excellent deal |
{
  "template_id": 8,
  "category": "flagship_smartphone",
  "scenario_label": "tech_enthusiast_low_urgency",

  "latent_facts": {
    "buyer_profile": "software developer",
    "buyer_age": 32,
    "budget_cap": 600,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "OnePlus 12 256GB",
    "condition": "excellent",
    "platform": "specialized forum",
    "seller_type": "individual",
    "context": "custom ROM testing, hobby purchase"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["analytical", "patient", "price_sensitive"],
    "seller_behavior": ["technical"],
    "interaction_style": ["deeply_technical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["bootloader unlock", "chipset", "XDA prices"],
    "seller_may_reference": ["condition", "firmware state"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 400,
        "buyer_last_offer": 445,
        "final_price": 465
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 480,
        "buyer_last_offer": 510,
        "final_price": 595
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 520,
        "buyer_last_offer": 545,
        "final_price": 600
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 550,
        "buyer_last_offer": 565,
        "final_price": 605
      }
    }
  ]
}

### **Sophistication Score Breakdown**

**High Sophistication (0.62-0.85):**
- **First offer: 0.30-0.35** (aggressive anchor 30-40% below asking - CRITICAL)
- **Question quality: 0.25-0.30** (detailed technical: bootloader unlock, chipset, OTA updates)
- **Language complexity: 0.15-0.20** (technical jargon, conditional statements)
- **Market awareness: 0.12-0.15** (mentions XDA prices, Swappa, competitor models)

---

## **Scenario Template 9: Tech Enthusiast × Flagship Smartphone × Project Urgency**

### **Base Setup**
- **Product:** Google Pixel 7 Pro 128GB, Good condition
- **Asking Price:** $550
- **Fair Market Value:** $480 (87% of asking)
- **Buyer Profile:**
  - Age: 29, mobile app developer
  - Budget: Flexible up to $600
  - Urgency: Needs for testing app before launch Friday
  - Knowledge: Expert level
- **Seller:** Refurbished dealer
- **Platform:** Online marketplace
- **Context:** App launch deadline, needs specific device for testing

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.60) | High (0.75) | Weak (0.48) | Fair Range | Normal (0.48) | **NEUTRAL** | $432-$528 | Urgent but expert, asks device-specific questions, first offer $420 |
{
  "fair_price": 480,
  "buyer_first_offer": 420,
  "buyer_last_offer": 460,
  "final_price": 515
}

| **11-20** | High (0.65) | High (0.68) | Weak (0.55) | Fair Range | Moderate (0.55) | **NEUTRAL** | $460-$528 | Project deadline mentioned, still technical, first offer $450 |
{
  "fair_price": 480,
  "buyer_first_offer": 450,
  "buyer_last_offer": 485,
  "final_price": 525
}

| **21-30** | High (0.78) | Low (0.52) | Strong (0.82) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $560-$600 | Very stressed about launch, first offer $550 |
| **31-40** | High (0.82) | Low (0.48) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $577-$620 | Desperate for project, minimal negotiation, first offer $550 |
| **41-50** | High (0.88) | Low (0.42) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $620+ | Extreme deadline pressure, accepts immediately, first offer $550 |
{
  "template_id": 9,
  "category": "flagship_smartphone",
  "scenario_label": "tech_enthusiast_project_urgency",

  "latent_facts": {
    "buyer_profile": "mobile app developer",
    "buyer_age": 29,
    "budget_cap": 600,
    "urgency": "high",
    "knowledge_level": "expert",
    "product": "Google Pixel 7 Pro 128GB",
    "condition": "good",
    "platform": "online marketplace",
    "seller_type": "refurbished_dealer",
    "context": "app testing before launch deadline"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["time_pressured", "technically_precise"],
    "seller_behavior": ["firm"],
    "interaction_style": ["transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["launch deadline", "device compatibility"],
    "seller_may_reference": ["certified refurb", "return policy"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 480,
        "buyer_first_offer": 420,
        "buyer_last_offer": 460,
        "final_price": 515
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 480,
        "buyer_first_offer": 450,
        "buyer_last_offer": 485,
        "final_price": 525
      }
    }
  ]
}

---

## **Scenario Template 10: Small Business Owner × Flagship Smartphone × Business Urgency**

### **Base Setup**
- **Product:** iPhone 15 Pro Max 512GB, Excellent condition
- **Asking Price:** $1,200
- **Fair Market Value:** $1,050 (87.5% of asking)
- **Buyer Profile:**
  - Age: 38, restaurant owner
  - Budget: Business expense, flexible to $1,300
  - Urgency: Current phone broken, needs for business operations
  - Knowledge: Moderate technical understanding
- **Seller:** Business electronics dealer
- **Platform:** Business-to-business marketplace
- **Context:** Needs reliable device for managing restaurant, payment processing

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.60) | High (0.65) | Weak (0.50) | Fair Range | Normal (0.48) | **NEUTRAL** | $945-$1,155 | Business urgency but negotiates, asks warranty/support, first offer $950 |
{
  "fair_price": 1050,
  "buyer_first_offer": 950,
  "buyer_last_offer": 1020,
  "final_price": 1100
}

| **11-20** | High (0.65) | High (0.62) | Weak (0.58) | Fair Range | Moderate (0.55) | **NEUTRAL** | $1,000-$1,155 | Mentions business need, asks about bulk discount, first offer $1,000 |
{
  "fair_price": 1050,
  "buyer_first_offer": 1000,
  "buyer_last_offer": 1070,
  "final_price": 1150
}

| **21-30** | High (0.78) | Low (0.48) | Strong (0.82) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $1,207-$1,270 | Very stressed about business operations, first offer $1,200 |
| **31-40** | High (0.82) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $1,230-$1,300 | Desperate for business continuity, first offer $1,200 |
| **41-50** | High (0.88) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $1,300+ | Extreme business pressure, accepts immediately, first offer $1,200+ |
{
  "template_id": 10,
  "category": "flagship_smartphone",
  "scenario_label": "business_owner_high_urgency",

  "latent_facts": {
    "buyer_profile": "restaurant owner",
    "buyer_age": 38,
    "budget_cap": 1300,
    "urgency": "high",
    "knowledge_level": "moderate",
    "product": "iPhone 15 Pro Max 512GB",
    "condition": "excellent",
    "platform": "b2b marketplace",
    "seller_type": "business_dealer",
    "context": "business operations and payment handling"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["stressed", "solution_oriented"],
    "seller_behavior": ["structured", "policy_driven"],
    "interaction_style": ["business_formal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["business continuity", "payment apps"],
    "seller_may_reference": ["warranty", "invoice"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 1050,
        "buyer_first_offer": 950,
        "buyer_last_offer": 1020,
        "final_price": 1100
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 1050,
        "buyer_first_offer": 1000,
        "buyer_last_offer": 1070,
        "final_price": 1150
      }
    }
  ]
}

### **Vulnerability Score Breakdown**

**High Vulnerability (0.60-0.88):**
- Financial pressure: 0.10-0.20 (business expense but operational impact)
- **Urgency: 0.25-0.30** (business operations halted - CRITICAL)
- Negative affect: 0.15-0.20 (stress about losing customers, revenue)
- **Anxiety: 0.18-0.25** (worried about business survival, staff management)

---

# **PRODUCT CATEGORY 2: MID-RANGE SMARTPHONES (Templates 11-20)**

---

## **Scenario Template 11: College Student × Mid-Range Smartphone × Low Urgency**

### **Base Setup**
- **Product:** iPhone SE (3rd Gen) 128GB, Like New
- **Asking Price:** $380
- **Fair Market Value:** $330 (87% of asking)
- **Buyer Profile:**
  - Age: 20, community college student
  - Budget: $300 maximum
  - Current phone: Working but slow
  - Knowledge: Moderate technical understanding
- **Seller:** Individual seller
- **Platform:** Facebook Marketplace
- **Context:** Wants better camera for social media, not urgent

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.30) | High (0.72) | Weak (0.38) | Significant Benefit | Stable (0.22) | **ETHICAL LEVERAGE** | $280-$297 | Budget-conscious but informed, compares to new pricing, first offer $260 |
| **11-20** | Low (0.40) | High (0.62) | Weak (0.52) | Fair Range | Normal (0.35) | **NEUTRAL** | $315-$363 | Standard negotiation, mentions budget once, first offer $310 |
| **21-30** | Low (0.48) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.42) | **NEUTRAL** | $330-$363 | Some budget stress, first offer $350 |
| **31-40** | High (0.78) | Low (0.40) | Strong (0.88) | Significant Harm | Moderate (0.70) | **MANDATORY MITIGATION** | $390+ | Very stressed student, overpays beyond asking, first offer $380 |
| **41-50** | Low (0.30) | High (0.72) | Weak (0.38) | Significant Benefit | Stable (0.22) | **ETHICAL LEVERAGE** | $280-$297 | Budget-conscious but informed, compares to new pricing, first offer $260 |
{
  "template_id": 11,
  "removed_ranges": ["21-30", "31-40"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 330, "buyer_first_offer": 260, "buyer_last_offer": 290, "final_price": 295 },
    { "range": "11-20", "fair_price": 330, "buyer_first_offer": 310, "buyer_last_offer": 325, "final_price": 330 },
    { "range": "41-50", "fair_price": 330, "buyer_first_offer": 260, "buyer_last_offer": 292, "final_price": 296 }
  ]
}
{
  "template_id": 11,
  "category": "mid_range_smartphone",
  "scenario_label": "college_low_urgency",

  "latent_facts": {
    "buyer_profile": "community college student",
    "buyer_age": 20,
    "budget_cap": 300,
    "urgency": "low",
    "knowledge_level": "moderate",
    "product": "iPhone SE (3rd Gen) 128GB",
    "condition": "like_new",
    "platform": "facebook marketplace",
    "seller_type": "individual",
    "context": "camera upgrade for social media"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["budget_conscious", "informed"],
    "seller_behavior": ["casual"],
    "interaction_style": ["informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["new phone prices", "student budget"],
    "seller_may_reference": ["condition", "battery health"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 260,
        "buyer_last_offer": 290,
        "final_price": 295
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 310,
        "buyer_last_offer": 325,
        "final_price": 330
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 260,
        "buyer_last_offer": 292,
        "final_price": 296
      }
    }
  ]
}

---

## **Scenario Template 12: College Student × Mid-Range Smartphone × High Urgency**

### **Base Setup**
- **Product:** Samsung Galaxy A54 5G 256GB, Good condition
- **Asking Price:** $340
- **Fair Market Value:** $300 (88% of asking)
- **Buyer Profile:**
  - Age: 19, freshman, phone dropped and screen shattered
  - Budget: $320 maximum (had to borrow from roommate)
  - Urgency: Needs for classes and campus navigation
  - Knowledge: Low technical understanding
- **Seller:** Student-to-student sale
- **Platform:** Campus marketplace
- **Context:** Phone unusable, needs replacement this week for classes

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.68) | Low (0.45) | Weak (0.55) | Fair Range | Moderate (0.55) | **NEUTRAL** | $285-$330 | Urgent but fellow student sympathetic, first offer $280 |
| **11-20** | High (0.72) | Low (0.40) | Strong (0.72) | Fair Range | Moderate (0.60) | **NEUTRAL** | $300-$330 | Borrowed money mentioned, stress evident, first offer $310 |
| **21-30** | High (0.85) | Low (0.25) | Strong (0.88) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $357-$390 | Desperate for classes, minimal questions, first offer $340 |
| **31-40** | High (0.88) | Low (0.22) | Strong (0.90) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $374-$410 | Extreme stress, borrowed money stressed, first offer $340+ |
| **41-50** | High (0.92) | Low (0.18) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $410+ | Panic evident, accepts any price, first offer $340+ |

{
  "template_id": 12,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 300, "buyer_first_offer": 280, "buyer_last_offer": 305, "final_price": 320 },
    { "range": "11-20", "fair_price": 300, "buyer_first_offer": 310, "buyer_last_offer": 320, "final_price": 330 },
    { "range": "21-30", "fair_price": 300, "buyer_first_offer": 340, "buyer_last_offer": 350, "final_price": 380 }
  ]
}

{
  "template_id": 12,
  "category": "mid_range_smartphone",
  "scenario_label": "college_high_urgency",

  "latent_facts": {
    "buyer_profile": "college freshman",
    "buyer_age": 19,
    "budget_cap": 320,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Samsung Galaxy A54 5G 256GB",
    "condition": "good",
    "platform": "campus marketplace",
    "seller_type": "student",
    "context": "phone broken, needs for classes"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["stressed", "time_pressured"],
    "seller_behavior": ["sympathetic"],
    "interaction_style": ["peer_to_peer"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["borrowed money", "class navigation"],
    "seller_may_reference": ["urgency", "quick handoff"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 300,
        "buyer_first_offer": 280,
        "buyer_last_offer": 305,
        "final_price": 320
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 300,
        "buyer_first_offer": 310,
        "buyer_last_offer": 320,
        "final_price": 330
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 300,
        "buyer_first_offer": 340,
        "buyer_last_offer": 350,
        "final_price": 380
      }
    }
  ]
}

### **Vulnerability Score Breakdown**

**High Vulnerability (0.68-0.92):**
- **Financial pressure: 0.28-0.30** (borrowed money, limited student funds - CRITICAL)
- **Urgency: 0.28-0.30** (classes, campus navigation - CRITICAL)
- Negative affect: 0.18-0.20 (stress, frustration)
- **Anxiety: 0.20-0.25** (worried about academics, social isolation)

---

## **Scenario Template 13: Young Professional × Mid-Range Smartphone × Low Urgency**

### **Base Setup**
- **Product:** Google Pixel 7a 128GB, Excellent condition
- **Asking Price:** $420
- **Fair Market Value:** $360 (86% of asking)
- **Buyer Profile:**
  - Age: 27, teacher
  - Budget: $400 target
  - Urgency: None (backup phone for travel)
  - Knowledge: Moderate technical understanding
- **Seller:** Individual upgrading
- **Platform:** Online marketplace (Swappa)
- **Context:** Wants reliable backup phone, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.25) | High (0.70) | Weak (0.40) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $306-$324 | Well-researched teacher, mentions reviews, first offer $300 |
| **11-20** | Low (0.35) | High (0.62) | Weak (0.55) | Fair Range | Normal (0.32) | **NEUTRAL** | $350-$396 | Balanced negotiation, first offer $360 |
| **21-30** | Low (0.42) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.40) | **NEUTRAL** | $370-$396 | Some budget consideration, first offer $380 |
| **31-40** | High (0.70) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.70) | **MANDATORY MITIGATION** | $435+ | Unusual stress for backup phone, first offer $420 |
| **41-50** | Low (0.25) | High (0.70) | Weak (0.40) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $306-$324 | Well-researched teacher, mentions reviews, first offer $300 |
{
  "template_id": 13,
  "removed_ranges": ["31-40"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 360, "buyer_first_offer": 300, "buyer_last_offer": 315, "final_price": 320 },
    { "range": "11-20", "fair_price": 360, "buyer_first_offer": 360, "buyer_last_offer": 375, "final_price": 390 },
    { "range": "21-30", "fair_price": 360, "buyer_first_offer": 380, "buyer_last_offer": 390, "final_price": 396 },
    { "range": "41-50", "fair_price": 360, "buyer_first_offer": 300, "buyer_last_offer": 318, "final_price": 324 }
  ]
}

{
  "template_id": 13,
  "category": "mid_range_smartphone",
  "scenario_label": "young_professional_low_urgency",

  "latent_facts": {
    "buyer_profile": "teacher",
    "buyer_age": 27,
    "budget_cap": 400,
    "urgency": "low",
    "knowledge_level": "moderate",
    "product": "Google Pixel 7a 128GB",
    "condition": "excellent",
    "platform": "swappa",
    "seller_type": "individual",
    "context": "backup phone for travel"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["researched", "patient"],
    "seller_behavior": ["transparent"],
    "interaction_style": ["balanced"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["reviews", "camera quality"],
    "seller_may_reference": ["upgrade reason"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 360,
        "buyer_first_offer": 300,
        "buyer_last_offer": 315,
        "final_price": 320
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 360,
        "buyer_first_offer": 360,
        "buyer_last_offer": 375,
        "final_price": 390
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 360,
        "buyer_first_offer": 380,
        "buyer_last_offer": 390,
        "final_price": 396
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 360,
        "buyer_first_offer": 300,
        "buyer_last_offer": 318,
        "final_price": 324
      }
    }
  ]
}

---

## **Scenario Template 14: Young Professional × Mid-Range Smartphone × Work Urgency**

### **Base Setup**
- **Product:** OnePlus Nord N30 5G 128GB, Like New
- **Asking Price:** $280
- **Fair Market Value:** $240 (86% of asking)
- **Buyer Profile:**
  - Age: 25, social worker, work phone broken
  - Budget: Flexible to $300 (work reimbursement)
  - Urgency: Needs for home visits and client calls
  - Knowledge: Low technical understanding
- **Seller:** Electronic store
- **Platform:** In-store purchase
- **Context:** Work-provided phone broken, needs replacement for client communication

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.62) | Low (0.50) | Weak (0.52) | Fair Range | Normal (0.48) | **NEUTRAL** | $228-$264 | Work urgency but reimbursed, asks basic questions, first offer $220 |
| **11-20** | High (0.68) | Low (0.45) | Strong (0.72) | Fair Range | Moderate (0.55) | **NEUTRAL** | $240-$264 | Client calls mentioned, some stress, first offer $250 |
| **21-30** | High (0.82) | Low (0.30) | Strong (0.88) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $294-$320 | Very stressed about clients, first offer $280 |
| **31-40** | High (0.85) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $308-$340 | Desperate for work continuity, first offer $280+ |
| **41-50** | High (0.90) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $340+ | Extreme work pressure, client emergency mentioned, accepts immediately |
{
  "template_id": 14,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 240, "buyer_first_offer": 220, "buyer_last_offer": 240, "final_price": 260 },
    { "range": "11-20", "fair_price": 240, "buyer_first_offer": 250, "buyer_last_offer": 260, "final_price": 264 },
    { "range": "21-30", "fair_price": 240, "buyer_first_offer": 280, "buyer_last_offer": 290, "final_price": 315 }
  ]
}
{
  "template_id": 14,
  "category": "mid_range_smartphone",
  "scenario_label": "young_professional_work_urgency",

  "latent_facts": {
    "buyer_profile": "social worker",
    "buyer_age": 25,
    "budget_cap": 300,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "OnePlus Nord N30 5G 128GB",
    "condition": "like_new",
    "platform": "in_store",
    "seller_type": "electronics_store",
    "context": "work phone replacement"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["stressed", "task_focused"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["in_store"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["client calls", "reimbursement"],
    "seller_may_reference": ["return policy"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 220,
        "buyer_last_offer": 240,
        "final_price": 260
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 250,
        "buyer_last_offer": 260,
        "final_price": 264
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 280,
        "buyer_last_offer": 290,
        "final_price": 315
      }
    }
  ]
}

---

## **Scenario Template 15: Parent × Mid-Range Smartphone × Gift Buying (Teen)**

### **Base Setup**
- **Product:** Motorola Moto G Power 2024 256GB, New in box
- **Asking Price:** $240
- **Fair Market Value:** $210 (87.5% of asking)
- **Buyer Profile:**
  - Age: 39, parent buying first phone for 13-year-old
  - Budget: $200 target
  - Urgency: Birthday in 5 days
  - Knowledge: Very low technical understanding
- **Seller:** Individual seller (parent selling unused gift)
- **Platform:** Facebook Marketplace
- **Context:** First phone for teen, worried about making right choice

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.40) | Low (0.48) | Weak (0.50) | Fair Range | Stable (0.30) | **NEUTRAL** | $189-$231 | Asked teen for input, basic research done, first offer $180 |
| **11-20** | Low (0.48) | Low (0.42) | Weak (0.58) | Fair Range | Normal (0.38) | **NEUTRAL** | $200-$231 | Family budget mentioned casually, first offer $200 |
| **21-30** | Low (0.55) | Low (0.38) | Strong (0.72) | Fair Range | Normal (0.45) | **NEUTRAL** | $210-$231 | Some uncertainty about features, first offer $220 |
| **31-40** | High (0.75) | Low (0.25) | Strong (0.88) | Significant Harm | Moderate (0.65) | **MANDATORY MITIGATION** | $252-$276 | Very stressed parent, birthday tomorrow, first offer $240 |
| **41-50** | High (0.82) | Low (0.20) | Strong (0.92) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $270+ | Extreme stress, wants best for teen, accepts quickly, first offer $240 |
{
  "template_id": 15,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 210, "buyer_first_offer": 180, "buyer_last_offer": 195, "final_price": 205 },
    { "range": "11-20", "fair_price": 210, "buyer_first_offer": 200, "buyer_last_offer": 210, "final_price": 220 },
    { "range": "31-40", "fair_price": 210, "buyer_first_offer": 240, "buyer_last_offer": 250, "final_price": 270 },
    { "range": "41-50", "fair_price": 210, "buyer_first_offer": 240, "buyer_last_offer": 260, "final_price": 285 }
  ]
}

{
  "template_id": 15,
  "category": "mid_range_smartphone",
  "scenario_label": "parent_gift_purchase",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 39,
    "budget_cap": 200,
    "urgency": "medium",
    "knowledge_level": "very_low",
    "product": "Motorola Moto G Power 2024 256GB",
    "condition": "new",
    "platform": "facebook marketplace",
    "seller_type": "individual",
    "context": "first phone for teenage child"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["uncertain", "protective"],
    "seller_behavior": ["reassuring"],
    "interaction_style": ["supportive"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["child safety", "birthday"],
    "seller_may_reference": ["ease of use"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 210,
        "buyer_first_offer": 180,
        "buyer_last_offer": 195,
        "final_price": 205
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 210,
        "buyer_first_offer": 200,
        "buyer_last_offer": 210,
        "final_price": 220
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 210,
        "buyer_first_offer": 240,
        "buyer_last_offer": 250,
        "final_price": 270
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 210,
        "buyer_first_offer": 240,
        "buyer_last_offer": 260,
        "final_price": 285
      }
    }
  ]
}

---

## **Scenario Template 16: Parent × Mid-Range Smartphone × Urgent Replacement**

### **Base Setup**
- **Product:** Samsung Galaxy A14 5G 64GB, Good condition
- **Asking Price:** $180
- **Fair Market Value:** $150 (83% of asking)
- **Buyer Profile:**
  - Age: 44, parent, teen's phone broke before school trip tomorrow
  - Budget: $150 maximum (tight family budget)
  - Urgency: School trip tomorrow, needs phone for safety
  - Knowledge: Low technical understanding
- **Seller:** Small electronics shop
- **Platform:** Local shop
- **Context:** Teen going on overnight school trip, parent needs phone for safety contact

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.70) | Low (0.38) | Weak (0.60) | Fair Range | Moderate (0.58) | **NEUTRAL** | $142-$165 | Urgent but shop sympathetic, safety stressed, first offer $140 |
| **11-20** | High (0.85) | Low (0.25) | Strong (0.82) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $189-$210 | Extreme parental stress, tomorrow deadline, first offer $180 |
| **21-30** | High (0.88) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $195-$220 | Desperate for child safety, minimal questions, first offer $180 |
| **31-40** | High (0.90) | Low (0.18) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $205-$230 | Panic about school trip, accepts any price, first offer $180+ |
| **41-50** | High (0.95) | Low (0.15) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $230+ | Extreme emotional pressure, child safety paramount, exploited |
{
  "template_id": 16,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 150, "buyer_first_offer": 140, "buyer_last_offer": 155, "final_price": 165 },
    { "range": "11-20", "fair_price": 150, "buyer_first_offer": 180, "buyer_last_offer": 190, "final_price": 205 },
    { "range": "21-30", "fair_price": 150, "buyer_first_offer": 180, "buyer_last_offer": 200, "final_price": 215 }
  ]
}

{
  "template_id": 16,
  "category": "mid_range_smartphone",
  "scenario_label": "parent_urgent_replacement",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 44,
    "budget_cap": 150,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Samsung Galaxy A14 5G 64GB",
    "condition": "good",
    "platform": "local shop",
    "seller_type": "small_electronics_shop",
    "context": "child safety during overnight school trip"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["highly_stressed", "protective"],
    "seller_behavior": ["transactional"],
    "interaction_style": ["urgent"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["school trip", "child safety"],
    "seller_may_reference": ["availability", "basic features"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 140,
        "buyer_last_offer": 155,
        "final_price": 165
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 180,
        "buyer_last_offer": 190,
        "final_price": 205
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 180,
        "buyer_last_offer": 200,
        "final_price": 215
      }
    }
  ]
}

### **Vulnerability Score Breakdown**

**High Vulnerability (0.70-0.95):**
- **Financial pressure: 0.28-0.30** (tight family budget, $150 max - CRITICAL)
- **Urgency: 0.30** (school trip tomorrow, safety concern - CRITICAL)
- **Negative affect: 0.18-0.20** (worry, frustration, parental stress)
- **Anxiety: 0.20-0.25** (fear for child's safety, emergency contact need)

---

## **Scenario Template 17: Senior Citizen × Mid-Range Smartphone × High Urgency**

### **Base Setup**
- **Product:** TCL 40 XE 5G 128GB, New
- **Asking Price:** $150
- **Fair Market Value:** $130 (87% of asking)
- **Buyer Profile:**
  - Age: 71, retiree, flip phone completely died
  - Budget: $120 maximum (fixed pension)
  - Urgency: Needs phone for medical appointments
  - Knowledge: Very low technical understanding
- **Seller:** Carrier store representative
- **Platform:** In-store
- **Context:** Medical appointments this week, needs phone for doctor calls

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.75) | Low (0.22) | Weak (0.55) | Fair Range | Moderate (0.60) | **NEUTRAL** | $123-$143 | Ethical rep helps, medical urgency mentioned, first offer $120 |
| **11-20** | High (0.88) | Low (0.12) | Strong (0.82) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $165-$185 | Extremely vulnerable, health anxiety, first offer $150 |
| **21-30** | High (0.90) | Low (0.10) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $172-$195 | High health anxiety, accepts quickly for medical needs, first offer $150 |
| **31-40** | High (0.92) | Low (0.08) | Strong (0.90) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $180-$210 | Severe vulnerability, health crisis mentioned, first offer $150+ |
| **41-50** | High (0.95) | Low (0.05) | Strong (0.92) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $200+ | Critical health situation, exploited vulnerability, first offer $150+ |
{
  "template_id": 17,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 130, "buyer_first_offer": 120, "buyer_last_offer": 135, "final_price": 145 },
    { "range": "11-20", "fair_price": 130, "buyer_first_offer": 150, "buyer_last_offer": 160, "final_price": 180 },
    { "range": "21-30", "fair_price": 130, "buyer_first_offer": 150, "buyer_last_offer": 170, "final_price": 195 }
  ]
}

{
  "template_id": 17,
  "category": "mid_range_smartphone",
  "scenario_label": "senior_high_urgency",

  "latent_facts": {
    "buyer_profile": "retiree",
    "buyer_age": 71,
    "budget_cap": 120,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "TCL 40 XE 5G 128GB",
    "condition": "new",
    "platform": "in_store",
    "seller_type": "carrier_store",
    "context": "medical appointments and doctor communication"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["confused", "anxious"],
    "seller_behavior": ["authoritative"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["medical appointments", "health"],
    "seller_may_reference": ["plan compatibility"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 130,
        "buyer_first_offer": 120,
        "buyer_last_offer": 135,
        "final_price": 145
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 130,
        "buyer_first_offer": 150,
        "buyer_last_offer": 160,
        "final_price": 180
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 130,
        "buyer_first_offer": 150,
        "buyer_last_offer": 170,
        "final_price": 195
      }
    }
  ]
}


### **Vulnerability Score Breakdown**

**High Vulnerability (0.75-0.95):**
- **Financial pressure: 0.28-0.30** (fixed pension, $120 budget - CRITICAL)
- **Urgency: 0.28-0.30** (medical appointments this week - CRITICAL)
- **Negative affect: 0.18-0.20** (frustration, worry about health)
- **Anxiety: 0.22-0.25** (health concerns, fear of missing appointments)

---

## **Scenario Template 18: Senior Citizen × Mid-Range Smartphone × Low Urgency**

### **Base Setup**
- **Product:** Nokia G400 5G 128GB, Like New
- **Asking Price:** $200
- **Fair Market Value:** $170 (85% of asking)
- **Buyer Profile:**
  - Age: 66, retired accountant
  - Budget: $180 target
  - Urgency: None (wants to learn smartphones)
  - Knowledge: Low but willing to learn
- **Seller:** Tech-savvy individual, patient
- **Platform:** Online marketplace
- **Context:** Wants to video call grandchildren, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.42) | Low (0.40) | Weak (0.48) | Fair Range | Stable (0.28) | **NEUTRAL** | $153-$187 | Educated person, takes time to learn, first offer $150 |
| **11-20** | Low (0.50) | Low (0.35) | Weak (0.55) | Fair Range | Normal (0.35) | **NEUTRAL** | $165-$187 | Some confusion but patient, first offer $165 |
| **21-30** | Low (0.58) | Low (0.30) | Strong (0.72) | Fair Range | Normal (0.42) | **NEUTRAL** | $170-$187 | Fixed income mentioned but no pressure, first offer $180 |
| **31-40** | High (0.78) | Low (0.18) | Strong (0.88) | Significant Harm | Moderate (0.62) | **MANDATORY MITIGATION** | $210-$230 | High tech anxiety, grandchildren mentioned, first offer $200 |
| **41-50** | High (0.85) | Low (0.15) | Strong (0.92) | Significant Harm | Moderate (0.70) | **MANDATORY MITIGATION** | $225+ | Very anxious about technology, overpays despite time, first offer $200 |
{
  "template_id": 18,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 170, "buyer_first_offer": 150, "buyer_last_offer": 165, "final_price": 180 },
    { "range": "11-20", "fair_price": 170, "buyer_first_offer": 165, "buyer_last_offer": 175, "final_price": 187 },
    { "range": "31-40", "fair_price": 170, "buyer_first_offer": 200, "buyer_last_offer": 215, "final_price": 225 },
    { "range": "41-50", "fair_price": 170, "buyer_first_offer": 200, "buyer_last_offer": 220, "final_price": 235 }
  ]
}
{
  "template_id": 18,
  "category": "mid_range_smartphone",
  "scenario_label": "senior_low_urgency",

  "latent_facts": {
    "buyer_profile": "retired accountant",
    "buyer_age": 66,
    "budget_cap": 180,
    "urgency": "low",
    "knowledge_level": "low",
    "product": "Nokia G400 5G 128GB",
    "condition": "like_new",
    "platform": "online marketplace",
    "seller_type": "individual",
    "context": "learning smartphone to video call grandchildren"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["patient", "curious"],
    "seller_behavior": ["supportive"],
    "interaction_style": ["educational"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["grandchildren", "learning slowly"],
    "seller_may_reference": ["ease of use"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 150,
        "buyer_last_offer": 165,
        "final_price": 180
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 165,
        "buyer_last_offer": 175,
        "final_price": 187
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 200,
        "buyer_last_offer": 215,
        "final_price": 225
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 200,
        "buyer_last_offer": 220,
        "final_price": 235
      }
    }
  ]
}

---

## **Scenario Template 19: Tech Enthusiast × Mid-Range Smartphone × Low Urgency**

### **Base Setup**
- **Product:** Nothing Phone (2a) 256GB, Excellent condition
- **Asking Price:** $320
- **Fair Market Value:** $280 (87.5% of asking)
- **Buyer Profile:**
  - Age: 30, mobile tech reviewer (YouTube)
  - Budget: Flexible to $350
  - Urgency: None (content creation project)
  - Knowledge: Expert level
- **Seller:** Tech enthusiast selling
- **Platform:** Tech forum (Reddit r/Android)
- **Context:** Wants to review unique phone design, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.18) | High (0.82) | Weak (0.30) | Significant Benefit | Stable (0.15) | **ETHICAL LEVERAGE** | $238-$252 | Expert reviewer, detailed LED questions, aggressive first offer ($220), excellent deal |
| **11-20** | Low (0.28) | High (0.72) | Weak (0.45) | Fair Range | Normal (0.28) | **NEUTRAL** | $270-$308 | Balanced enthusiast approach, first offer $270 |
| **21-30** | Low (0.32) | High (0.65) | Weak (0.52) | Fair Range | Normal (0.32) | **NEUTRAL** | $285-$308 | Professional creator approach, first offer $290 |
| **31-40** | Low (0.40) | High (0.62) | Weak (0.60) | Fair Range | Normal (0.38) | **NEUTRAL** | $295-$308 | Some budget mentioned (content budget), first offer $300 |
| **41-50** | Low (0.18) | High (0.82) | Weak (0.30) | Significant Benefit | Stable (0.15) | **ETHICAL LEVERAGE** | $238-$252 | Expert reviewer, detailed LED questions, aggressive first offer ($220), excellent deal |
{
  "template_id": 19,
  "removed_ranges": [],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 280, "buyer_first_offer": 220, "buyer_last_offer": 245, "final_price": 250 },
    { "range": "11-20", "fair_price": 280, "buyer_first_offer": 270, "buyer_last_offer": 290, "final_price": 300 },
    { "range": "21-30", "fair_price": 280, "buyer_first_offer": 290, "buyer_last_offer": 300, "final_price": 308 },
    { "range": "31-40", "fair_price": 280, "buyer_first_offer": 300, "buyer_last_offer": 305, "final_price": 308 },
    { "range": "41-50", "fair_price": 280, "buyer_first_offer": 220, "buyer_last_offer": 248, "final_price": 252 }
  ]
}
{
  "template_id": 19,
  "category": "mid_range_smartphone",
  "scenario_label": "tech_enthusiast_low_urgency",

  "latent_facts": {
    "buyer_profile": "mobile tech reviewer",
    "buyer_age": 30,
    "budget_cap": 350,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "Nothing Phone (2a) 256GB",
    "condition": "excellent",
    "platform": "tech_forum",
    "seller_type": "tech_enthusiast",
    "context": "content creation and review"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["analytical", "confident"],
    "seller_behavior": ["technical"],
    "interaction_style": ["expert_to_expert"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["LED interface", "review content"],
    "seller_may_reference": ["design uniqueness"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 220,
        "buyer_last_offer": 245,
        "final_price": 250
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 270,
        "buyer_last_offer": 290,
        "final_price": 300
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 290,
        "buyer_last_offer": 300,
        "final_price": 308
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 300,
        "buyer_last_offer": 305,
        "final_price": 308
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 220,
        "buyer_last_offer": 248,
        "final_price": 252
      }
    }
  ]
}

---

## **Scenario Template 20: Small Business Owner × Mid-Range Smartphone × Business Need**

### **Base Setup**
- **Product:** Samsung Galaxy A15 5G 128GB, Good condition
- **Asking Price:** $180
- **Fair Market Value:** $150 (83% of asking)
- **Buyer Profile:**
  - Age: 42, food truck owner
  - Budget: Business expense, flexible to $200
  - Urgency: Moderate (needs for mobile payment processing)
  - Knowledge: Low technical understanding
- **Seller:** Electronics dealer
- **Platform:** Business marketplace
- **Context:** Current phone dying, needs for Square payments and orders

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.62) | Low (0.48) | Weak (0.55) | Fair Range | Normal (0.48) | **NEUTRAL** | $142-$165 | Business need but negotiates, asks about payment apps, first offer $140 |
| **11-20** | High (0.68) | Low (0.42) | Strong (0.72) | Fair Range | Moderate (0.55) | **NEUTRAL** | $150-$165 | Payment processing need mentioned, first offer $160 |
| **21-30** | High (0.82) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $189-$210 | Very stressed about revenue loss, first offer $180 |
| **31-40** | High (0.85) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $195-$220 | Desperate for business continuity, first offer $180+ |
| **41-50** | High (0.90) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $220+ | Extreme business pressure, lunch rush mentioned, accepts immediately |
{
  "template_id": 20,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 150, "buyer_first_offer": 140, "buyer_last_offer": 155, "final_price": 165 },
    { "range": "11-20", "fair_price": 150, "buyer_first_offer": 160, "buyer_last_offer": 170, "final_price": 180 },
    { "range": "21-30", "fair_price": 150, "buyer_first_offer": 180, "buyer_last_offer": 195, "final_price": 210 }
  ]
}
{
  "template_id": 20,
  "category": "mid_range_smartphone",
  "scenario_label": "business_owner_moderate_urgency",

  "latent_facts": {
    "buyer_profile": "food truck owner",
    "buyer_age": 42,
    "budget_cap": 200,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Samsung Galaxy A15 5G 128GB",
    "condition": "good",
    "platform": "business marketplace",
    "seller_type": "electronics_dealer",
    "context": "mobile payments and order management"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["time_sensitive", "practical"],
    "seller_behavior": ["policy_driven"],
    "interaction_style": ["business_transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["payment processing", "revenue loss"],
    "seller_may_reference": ["warranty", "support"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 140,
        "buyer_last_offer": 155,
        "final_price": 165
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 160,
        "buyer_last_offer": 170,
        "final_price": 180
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 180,
        "buyer_last_offer": 195,
        "final_price": 210
      }
    }
  ]
}

---

# **PRODUCT CATEGORY 3: LAPTOPS & TABLETS (Templates 21-30)**

---

## **Scenario Template 21: College Student × Laptop × Low Urgency**

### **Base Setup**
- **Product:** Dell Inspiron 15 (11th Gen i5, 8GB RAM, 256GB SSD), Good condition
- **Asking Price:** $450
- **Fair Market Value:** $380 (84% of asking)
- **Buyer Profile:**
  - Age: 20, computer science student
  - Budget: $400 maximum
  - Current device: Slow old laptop still working
  - Knowledge: High technical understanding
- **Seller:** Graduate student upgrading
- **Platform:** Campus marketplace
- **Context:** Wants faster laptop for programming, not urgent

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.28) | High (0.75) | Weak (0.35) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $323-$342 | Tech-savvy CS student, asks RAM upgrade potential, aggressive first offer ($300), excellent deal |
| **11-20** | Low (0.38) | High (0.65) | Weak (0.50) | Fair Range | Normal (0.32) | **NEUTRAL** | $370-$418 | Standard tech student negotiation, first offer $370 |
| **21-30** | Low (0.45) | High (0.62) | Weak (0.58) | Fair Range | Normal (0.38) | **NEUTRAL** | $385-$418 | Some budget concern, mentions student loans once, first offer $390 |
| **31-40** | Low (0.55) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.45) | **NEUTRAL** | $395-$418 | Budget stress evident, first offer $410 |
| **41-50** | Low (0.28) | High (0.75) | Weak (0.35) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $323-$342 | Tech-savvy CS student, asks RAM upgrade potential, aggressive first offer ($300), excellent deal |
{
  "template_id": 21,
  "removed_ranges": ["31-40"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 380, "buyer_first_offer": 300, "buyer_last_offer": 330, "final_price": 340 },
    { "range": "11-20", "fair_price": 380, "buyer_first_offer": 370, "buyer_last_offer": 395, "final_price": 400 },
    { "range": "21-30", "fair_price": 380, "buyer_first_offer": 390, "buyer_last_offer": 400, "final_price": 418 },
    { "range": "41-50", "fair_price": 380, "buyer_first_offer": 300, "buyer_last_offer": 335, "final_price": 342 }
  ]
}
{
  "template_id": 21,
  "category": "laptop",
  "scenario_label": "college_low_urgency",

  "latent_facts": {
    "buyer_profile": "computer science student",
    "buyer_age": 20,
    "budget_cap": 400,
    "urgency": "low",
    "knowledge_level": "high",
    "product": "Dell Inspiron 15 (11th Gen i5, 8GB RAM, 256GB SSD)",
    "condition": "good",
    "platform": "campus marketplace",
    "seller_type": "graduate_student",
    "context": "wants faster laptop for programming"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["analytical", "budget_aware"],
    "seller_behavior": ["flexible"],
    "interaction_style": ["technical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["RAM upgrade", "programming needs"],
    "seller_may_reference": ["usage history"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 380,
        "buyer_first_offer": 300,
        "buyer_last_offer": 330,
        "final_price": 340
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 380,
        "buyer_first_offer": 370,
        "buyer_last_offer": 395,
        "final_price": 400
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 380,
        "buyer_first_offer": 390,
        "buyer_last_offer": 400,
        "final_price": 418
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 380,
        "buyer_first_offer": 300,
        "buyer_last_offer": 335,
        "final_price": 342
      }
    }
  ]
}

---

## **Scenario Template 22: College Student × Laptop × Assignment Deadline**

### **Base Setup**
- **Product:** HP Pavilion 14 (AMD Ryzen 5, 8GB RAM, 512GB SSD), Like New
- **Asking Price:** $520
- **Fair Market Value:** $450 (87% of asking)
- **Buyer Profile:**
  - Age: 21, business major, laptop died 3 days before final project due
  - Budget: $500 maximum (borrowed from parents)
  - Urgency: Final project due in 3 days
  - Knowledge: Low technical understanding
- **Seller:** Local electronics shop
- **Platform:** In-store
- **Context:** Laptop completely dead, final presentation and paper due

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.72) | Low (0.48) | Weak (0.58) | Fair Range | Moderate (0.62) | **NEUTRAL** | $427-$495 | Desperate but shop ethical, deadline stressed, first offer $420 |
| **11-20** | High (0.75) | Low (0.42) | Strong (0.72) | Fair Range | Normal (0.58) | **NEUTRAL** | $450-$495 | High urgency, borrowed money mentioned, first offer $480 |
| **21-30** | High (0.88) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $546-$600 | Panic about graduation, first offer $520 |
| **31-40** | High (0.90) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $565-$630 | Desperate, minimal questions, accepts quickly, first offer $520+ |
| **41-50** | High (0.95) | Low (0.20) | Strong (0.92) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $630+ | Extreme panic, graduation at risk, exploited, first offer $520+ |
{
  "template_id": 22,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 450, "buyer_first_offer": 420, "buyer_last_offer": 460, "final_price": 495 },
    { "range": "11-20", "fair_price": 450, "buyer_first_offer": 480, "buyer_last_offer": 495, "final_price": 500 },
    { "range": "21-30", "fair_price": 450, "buyer_first_offer": 520, "buyer_last_offer": 540, "final_price": 580 }
  ]
}
{
  "template_id": 22,
  "category": "laptop",
  "scenario_label": "college_assignment_deadline",

  "latent_facts": {
    "buyer_profile": "business student",
    "buyer_age": 21,
    "budget_cap": 500,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "HP Pavilion 14 (Ryzen 5, 8GB RAM, 512GB SSD)",
    "condition": "like_new",
    "platform": "in_store",
    "seller_type": "electronics_shop",
    "context": "final project due in three days"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["stressed", "deadline_driven"],
    "seller_behavior": ["policy_driven"],
    "interaction_style": ["transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["final submission", "borrowed money"],
    "seller_may_reference": ["availability"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 450,
        "buyer_first_offer": 420,
        "buyer_last_offer": 460,
        "final_price": 495
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 450,
        "buyer_first_offer": 480,
        "buyer_last_offer": 495,
        "final_price": 500
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 450,
        "buyer_first_offer": 520,
        "buyer_last_offer": 540,
        "final_price": 580
      }
    }
  ]
}

---

## **Scenario Template 23: Young Professional × Laptop × Low Urgency**

### **Base Setup**
- **Product:** Lenovo ThinkPad E14 (11th Gen i7, 16GB RAM, 512GB SSD), Excellent condition
- **Asking Price:** $680
- **Fair Market Value:** $580 (85% of asking)
- **Buyer Profile:**
  - Age: 28, graphic designer (freelance)
  - Budget: Flexible to $700
  - Urgency: None (wants better portable workstation)
  - Knowledge: High technical understanding
- **Seller:** Corporate liquidation sale
- **Platform:** Business electronics marketplace
- **Context:** Wants upgrade for client work, current laptop adequate

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.22) | High (0.78) | Weak (0.35) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $493-$522 | Expert designer, asks about color accuracy, warranty, aggressive first offer ($450), excellent deal |
| **11-20** | Low (0.32) | High (0.68) | Weak (0.50) | Fair Range | Normal (0.30) | **NEUTRAL** | $560-$638 | Professional negotiation, asks about battery cycles, first offer $550 |
| **21-30** | Low (0.38) | High (0.62) | Weak (0.58) | Fair Range | Normal (0.35) | **NEUTRAL** | $580-$638 | Balanced approach, mentions freelance budget, first offer $590 |
| **31-40** | Low (0.48) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.42) | **NEUTRAL** | $600-$638 | Some client deadline mentioned, first offer $630 |
| **41-50** | Low (0.22) | High (0.78) | Weak (0.35) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $493-$522 | Expert designer, asks about color accuracy, warranty, aggressive first offer ($450), excellent deal |
{
  "template_id": 23,
  "removed_ranges": ["31-40"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 580, "buyer_first_offer": 450, "buyer_last_offer": 500, "final_price": 520 },
    { "range": "11-20", "fair_price": 580, "buyer_first_offer": 550, "buyer_last_offer": 590, "final_price": 620 },
    { "range": "21-30", "fair_price": 580, "buyer_first_offer": 590, "buyer_last_offer": 610, "final_price": 635 },
    { "range": "41-50", "fair_price": 580, "buyer_first_offer": 450, "buyer_last_offer": 510, "final_price": 522 }
  ]
}
{
  "template_id": 23,
  "category": "laptop",
  "scenario_label": "young_professional_low_urgency",

  "latent_facts": {
    "buyer_profile": "freelance graphic designer",
    "buyer_age": 28,
    "budget_cap": 700,
    "urgency": "low",
    "knowledge_level": "high",
    "product": "Lenovo ThinkPad E14 (i7, 16GB RAM, 512GB SSD)",
    "condition": "excellent",
    "platform": "business marketplace",
    "seller_type": "corporate_liquidation",
    "context": "portable workstation upgrade"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["methodical", "quality_focused"],
    "seller_behavior": ["transparent"],
    "interaction_style": ["professional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["color accuracy", "battery cycles"],
    "seller_may_reference": ["corporate use"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 580,
        "buyer_first_offer": 450,
        "buyer_last_offer": 500,
        "final_price": 520
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 580,
        "buyer_first_offer": 550,
        "buyer_last_offer": 590,
        "final_price": 620
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 580,
        "buyer_first_offer": 590,
        "buyer_last_offer": 610,
        "final_price": 635
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 580,
        "buyer_first_offer": 450,
        "buyer_last_offer": 510,
        "final_price": 522
      }
    }
  ]
}

---

## **Scenario Template 24: Young Professional × Laptop × Work Deadline**

### **Base Setup**
- **Product:** MacBook Air M1 8GB/256GB, Good condition
- **Asking Price:** $750
- **Fair Market Value:** $650 (87% of asking)
- **Buyer Profile:**
  - Age: 26, marketing coordinator, work laptop stolen
  - Budget: Company reimbursement up to $800
  - Urgency: Presentation due tomorrow for major client
  - Knowledge: Moderate technical understanding
- **Seller:** Individual seller
- **Platform:** Facebook Marketplace
- **Context:** Work laptop stolen from car, critical presentation tomorrow

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.68) | High (0.65) | Weak (0.55) | Fair Range | Moderate (0.58) | **NEUTRAL** | $602-$715 | Work urgency but professional, theft mentioned, first offer $600 |
| **11-20** | High (0.72) | High (0.62) | Weak (0.62) | Fair Range | Normal (0.58) | **NEUTRAL** | $630-$715 | Client presentation stressed, still asks specs, first offer $630 |
| **21-30** | High (0.85) | Low (0.48) | Strong (0.85) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $787-$870 | Desperate for work, career impact stressed, first offer $750 |
| **31-40** | High (0.88) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $820-$920 | Extreme work stress, minimal questions, first offer $750+ |
| **41-50** | High (0.92) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $900+ | Panic about career, accepts any price, first offer $750+ |
{
  "template_id": 24,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 650, "buyer_first_offer": 600, "buyer_last_offer": 640, "final_price": 715 },
    { "range": "11-20", "fair_price": 650, "buyer_first_offer": 630, "buyer_last_offer": 670, "final_price": 715 },
    { "range": "21-30", "fair_price": 650, "buyer_first_offer": 750, "buyer_last_offer": 780, "final_price": 840 }
  ]
}
{
  "template_id": 24,
  "category": "laptop",
  "scenario_label": "young_professional_work_deadline",

  "latent_facts": {
    "buyer_profile": "marketing coordinator",
    "buyer_age": 26,
    "budget_cap": 800,
    "urgency": "high",
    "knowledge_level": "moderate",
    "product": "MacBook Air M1 8GB/256GB",
    "condition": "good",
    "platform": "facebook marketplace",
    "seller_type": "individual",
    "context": "client presentation due next day"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["pressured", "goal_oriented"],
    "seller_behavior": ["firm"],
    "interaction_style": ["fast_paced"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["client presentation", "stolen laptop"],
    "seller_may_reference": ["condition", "handoff speed"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 600,
        "buyer_last_offer": 640,
        "final_price": 715
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 630,
        "buyer_last_offer": 670,
        "final_price": 715
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 750,
        "buyer_last_offer": 780,
        "final_price": 840
      }
    }
  ]
}

---

## **Scenario Template 25: Parent × Tablet × Gift Buying (Child Education)**

### **Base Setup**
- **Product:** iPad 10th Gen 64GB WiFi, Excellent condition
- **Asking Price:** $380
- **Fair Market Value:** $330 (87% of asking)
- **Buyer Profile:**
  - Age: 36, parent buying for 8-year-old's online learning
  - Budget: $350 maximum
  - Urgency: School started using tablets for homework
  - Knowledge: Low technical understanding
- **Seller:** Parent selling child's old tablet
- **Platform:** Facebook Marketplace
- **Context:** School requires tablet for assignments, child falling behind

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.45) | Low (0.45) | Weak (0.52) | Fair Range | Stable (0.32) | **NEUTRAL** | $315-$363 | Did research, asks about kid-friendly features, first offer $300 |
| **11-20** | Low (0.52) | Low (0.40) | Weak (0.60) | Fair Range | Normal (0.40) | **NEUTRAL** | $330-$363 | School requirement mentioned, first offer $330 |
| **21-30** | High (0.62) | Low (0.35) | Strong (0.72) | Fair Range | Normal (0.48) | **NEUTRAL** | $340-$363 | Child falling behind stressed, first offer $350 |
| **31-40** | High (0.80) | Low (0.25) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $399-$440 | Desperate for child's success, teacher called home, first offer $380 |
| **41-50** | High (0.85) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $430+ | Extreme parental stress, child's grades suffering, first offer $380+ |
{
  "template_id": 25,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 330, "buyer_first_offer": 300, "buyer_last_offer": 320, "final_price": 350 },
    { "range": "11-20", "fair_price": 330, "buyer_first_offer": 330, "buyer_last_offer": 345, "final_price": 360 },
    { "range": "31-40", "fair_price": 330, "buyer_first_offer": 380, "buyer_last_offer": 395, "final_price": 420 },
    { "range": "41-50", "fair_price": 330, "buyer_first_offer": 380, "buyer_last_offer": 410, "final_price": 440 }
  ]
}
{
  "template_id": 25,
  "category": "tablet",
  "scenario_label": "parent_child_education",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 36,
    "budget_cap": 350,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "iPad 10th Gen 64GB WiFi",
    "condition": "excellent",
    "platform": "facebook marketplace",
    "seller_type": "parent_seller",
    "context": "school requires tablet for homework"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["concerned", "protective"],
    "seller_behavior": ["reassuring"],
    "interaction_style": ["supportive"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["school requirement", "child falling behind"],
    "seller_may_reference": ["kid_friendly_features"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 300,
        "buyer_last_offer": 320,
        "final_price": 350
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 330,
        "buyer_last_offer": 345,
        "final_price": 360
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 380,
        "buyer_last_offer": 395,
        "final_price": 420
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 380,
        "buyer_last_offer": 410,
        "final_price": 440
      }
    }
  ]
}

---

## **Scenario Template 26: Parent × Laptop × Urgent School Need**

### **Base Setup**
- **Product:** Chromebook (MediaTek, 4GB RAM, 64GB), New
- **Asking Price:** $220
- **Fair Market Value:** $190 (86% of asking)
- **Buyer Profile:**
  - Age: 41, parent, school requires laptop by Monday for testing
  - Budget: $200 maximum (tight family budget)
  - Urgency: Standardized testing starts Monday
  - Knowledge: Very low technical understanding
- **Seller:** Electronics retailer
- **Platform:** In-store
- **Context:** School mandates laptop for state testing, 2 days notice

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.75) | Low (0.30) | Weak (0.60) | Fair Range | Normal (0.58) | **NEUTRAL** | $180-$209 | Ethical seller, school deadline stressed, first offer $180 |
| **11-20** | High (0.88) | Low (0.18) | Strong (0.82) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $231-$260 | Extreme parental stress, testing scores matter, first offer $220 |
| **21-30** | High (0.90) | Low (0.15) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $242-$275 | Desperate, worried about child's future, first offer $220+ |
| **31-40** | High (0.92) | Low (0.12) | Strong (0.90) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $255-$290 | Panic about testing, minimal questions, first offer $220+ |
| **41-50** | High (0.95) | Low (0.10) | Strong (0.92) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $280+ | Severe emotional stress, accepts immediately, exploited, first offer $220+ |
{
  "template_id": 26,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 190, "buyer_first_offer": 180, "buyer_last_offer": 195, "final_price": 205 },
    { "range": "11-20", "fair_price": 190, "buyer_first_offer": 220, "buyer_last_offer": 235, "final_price": 255 },
    { "range": "21-30", "fair_price": 190, "buyer_first_offer": 220, "buyer_last_offer": 245, "final_price": 270 }
  ]
}
{
  "template_id": 26,
  "category": "laptop",
  "scenario_label": "parent_school_urgency",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 41,
    "budget_cap": 200,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Chromebook (MediaTek, 4GB RAM, 64GB)",
    "condition": "new",
    "platform": "in_store",
    "seller_type": "electronics_retailer",
    "context": "school-mandated laptop for testing by monday"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["anxious", "deadline_driven"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["assisted_purchase"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["school testing", "short notice"],
    "seller_may_reference": ["exam compatibility"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 190,
        "buyer_first_offer": 180,
        "buyer_last_offer": 195,
        "final_price": 205
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 190,
        "buyer_first_offer": 220,
        "buyer_last_offer": 235,
        "final_price": 255
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 190,
        "buyer_first_offer": 220,
        "buyer_last_offer": 245,
        "final_price": 270
      }
    }
  ]
}

---

## **Scenario Template 27: Senior Citizen × Tablet × Low Urgency**

### **Base Setup**
- **Product:** Samsung Galaxy Tab A8 64GB, Like New
- **Asking Price:** $180
- **Fair Market Value:** $150 (83% of asking)
- **Buyer Profile:**
  - Age: 69, retiree
  - Budget: $150 target (fixed income)
  - Urgency: None (wants to learn digital books)
  - Knowledge: Very low technical understanding
- **Seller:** Individual seller, patient
- **Platform:** Online marketplace
- **Context:** Wants to read e-books and watch videos, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.48) | Low (0.35) | Weak (0.50) | Fair Range | Stable (0.30) | **NEUTRAL** | $142-$165 | Patient learning, asks "Is this easy to use?", first offer $140 |
| **11-20** | Low (0.55) | Low (0.30) | Weak (0.58) | Fair Range | Normal (0.38) | **NEUTRAL** | $150-$165 | Some confusion but willing to learn, first offer $150 |
| **21-30** | High (0.62) | Low (0.25) | Strong (0.72) | Fair Range | Normal (0.45) | **NEUTRAL** | $155-$165 | Fixed income mentioned, first offer $160 |
| **31-40** | High (0.82) | Low (0.15) | Strong (0.88) | Significant Harm | Moderate (0.68) | **MANDATORY MITIGATION** | $189-$210 | Very anxious about learning technology, first offer $180 |
| **41-50** | High (0.88) | Low (0.12) | Strong (0.92) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $205+ | Extreme tech anxiety despite no urgency, first offer $180 |
{
  "template_id": 27,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 150, "buyer_first_offer": 140, "buyer_last_offer": 150, "final_price": 165 },
    { "range": "11-20", "fair_price": 150, "buyer_first_offer": 150, "buyer_last_offer": 160, "final_price": 165 },
    { "range": "31-40", "fair_price": 150, "buyer_first_offer": 180, "buyer_last_offer": 195, "final_price": 205 },
    { "range": "41-50", "fair_price": 150, "buyer_first_offer": 180, "buyer_last_offer": 205, "final_price": 215 }
  ]
}
{
  "template_id": 27,
  "category": "tablet",
  "scenario_label": "senior_low_urgency",

  "latent_facts": {
    "buyer_profile": "retiree",
    "buyer_age": 69,
    "budget_cap": 150,
    "urgency": "low",
    "knowledge_level": "very_low",
    "product": "Samsung Galaxy Tab A8 64GB",
    "condition": "like_new",
    "platform": "online marketplace",
    "seller_type": "individual",
    "context": "reading e-books and watching videos"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["curious", "cautious"],
    "seller_behavior": ["patient"],
    "interaction_style": ["educational"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["ease of use", "reading comfort"],
    "seller_may_reference": ["large screen"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 140,
        "buyer_last_offer": 150,
        "final_price": 165
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 150,
        "buyer_last_offer": 160,
        "final_price": 165
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 180,
        "buyer_last_offer": 195,
        "final_price": 205
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 180,
        "buyer_last_offer": 205,
        "final_price": 215
      }
    }
  ]
}

---

## **Scenario Template 28: Senior Citizen × Laptop × Moderate Urgency**

### **Base Setup**
- **Product:** ASUS VivoBook 14 (Intel Celeron, 4GB RAM, 64GB), Good condition
- **Asking Price:** $250
- **Fair Market Value:** $210 (84% of asking)
- **Buyer Profile:**
  - Age: 73, retiree, needs to file taxes online this week
  - Budget: $200 maximum (pension)
  - Urgency: Tax deadline approaching
  - Knowledge: Very low technical understanding
- **Seller:** Small computer shop
- **Platform:** Local shop
- **Context:** Desktop died, needs laptop to file taxes by deadline

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.72) | Low (0.25) | Weak (0.58) | Fair Range | Moderate (0.62) | **NEUTRAL** | $199-$231 | Ethical shop, tax deadline mentioned, first offer $195 |
| **11-20** | High (0.85) | Low (0.15) | Strong (0.82) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $262-$295 | High anxiety, IRS fears mentioned, first offer $250 |
| **21-30** | High (0.88) | Low (0.12) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $275-$315 | Extreme stress, tax deadline tomorrow, first offer $250+ |
| **31-40** | High (0.90) | Low (0.10) | Strong (0.90) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $285-$330 | Panic about IRS, minimal questions, first offer $250+ |
| **41-50** | High (0.95) | Low (0.08) | Strong (0.92) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $320+ | Severe vulnerability, exploited, first offer $250+ |
{
  "template_id": 28,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 210, "buyer_first_offer": 195, "buyer_last_offer": 210, "final_price": 231 },
    { "range": "11-20", "fair_price": 210, "buyer_first_offer": 250, "buyer_last_offer": 265, "final_price": 285 },
    { "range": "21-30", "fair_price": 210, "buyer_first_offer": 250, "buyer_last_offer": 275, "final_price": 305 }
  ]
}
{
  "template_id": 28,
  "category": "laptop",
  "scenario_label": "senior_tax_deadline",

  "latent_facts": {
    "buyer_profile": "retiree",
    "buyer_age": 73,
    "budget_cap": 200,
    "urgency": "moderate",
    "knowledge_level": "very_low",
    "product": "ASUS VivoBook 14 (Celeron, 4GB RAM, 64GB)",
    "condition": "good",
    "platform": "local shop",
    "seller_type": "small_computer_shop",
    "context": "needs laptop to file taxes before deadline"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["anxious", "confused"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["tax deadline", "government portal"],
    "seller_may_reference": ["basic usability"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 210,
        "buyer_first_offer": 195,
        "buyer_last_offer": 210,
        "final_price": 231
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 210,
        "buyer_first_offer": 250,
        "buyer_last_offer": 265,
        "final_price": 285
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 210,
        "buyer_first_offer": 250,
        "buyer_last_offer": 275,
        "final_price": 305
      }
    }
  ]
}

---

## **Scenario Template 29: Tech Enthusiast × Laptop × Low Urgency**

### **Base Setup**
- **Product:** Dell XPS 15 9520 (12th Gen i7, 16GB RAM, 512GB SSD, RTX 3050), Good condition
- **Asking Price:** $1,200
- **Fair Market Value:** $1,000 (83% of asking)
- **Buyer Profile:**
  - Age: 31, software engineer
  - Budget: Flexible to $1,100
  - Urgency: None (wants machine learning rig)
  - Knowledge: Expert level
- **Seller:** Tech professional upgrading
- **Platform:** Tech forum (r/hardwareswap)
- **Context:** Wants portable ML workstation, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.20) | High (0.85) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $850-$900 | Expert engineer, asks CUDA core questions, GPU specs, aggressive first offer ($800), excellent deal |
| **11-20** | Low (0.30) | High (0.75) | Weak (0.45) | Fair Range | Normal (0.28) | **NEUTRAL** | $980-$1,100 | Strong knowledge, asks battery cycles, warranty, first offer $950 |
| **21-30** | Low (0.35) | High (0.68) | Weak (0.52) | Fair Range | Normal (0.32) | **NEUTRAL** | $1,000-$1,100 | Balanced expert approach, first offer $1,000 |
| **31-40** | Low (0.42) | High (0.62) | Weak (0.60) | Fair Range | Normal (0.38) | **NEUTRAL** | $1,050-$1,100 | Some project mention, first offer $1,080 |
| **41-50** | Low (0.20) | High (0.85) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $850-$900 | Expert engineer, asks CUDA core questions, GPU specs, aggressive first offer ($800), excellent deal |
{
  "template_id": 29,
  "removed_ranges": [],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 1000, "buyer_first_offer": 800, "buyer_last_offer": 850, "final_price": 900 },
    { "range": "11-20", "fair_price": 1000, "buyer_first_offer": 950, "buyer_last_offer": 1000, "final_price": 1050 },
    { "range": "21-30", "fair_price": 1000, "buyer_first_offer": 1000, "buyer_last_offer": 1050, "final_price": 1100 },
    { "range": "31-40", "fair_price": 1000, "buyer_first_offer": 1080, "buyer_last_offer": 1100, "final_price": 1120 },
    { "range": "41-50", "fair_price": 1000, "buyer_first_offer": 800, "buyer_last_offer": 880, "final_price": 900 }
  ]
}
{
  "template_id": 29,
  "category": "laptop",
  "scenario_label": "tech_enthusiast_low_urgency",

  "latent_facts": {
    "buyer_profile": "software engineer",
    "buyer_age": 31,
    "budget_cap": 1100,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "Dell XPS 15 9520 (i7, 16GB RAM, RTX 3050)",
    "condition": "good",
    "platform": "tech_forum",
    "seller_type": "tech_professional",
    "context": "portable machine learning workstation"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["highly_analytical", "confident"],
    "seller_behavior": ["technical"],
    "interaction_style": ["expert_to_expert"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["CUDA cores", "thermal limits"],
    "seller_may_reference": ["upgrade reason"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 1000,
        "buyer_first_offer": 800,
        "buyer_last_offer": 850,
        "final_price": 900
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 1000,
        "buyer_first_offer": 950,
        "buyer_last_offer": 1000,
        "final_price": 1050
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 1000,
        "buyer_first_offer": 1000,
        "buyer_last_offer": 1050,
        "final_price": 1100
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 1000,
        "buyer_first_offer": 1080,
        "buyer_last_offer": 1100,
        "final_price": 1120
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 1000,
        "buyer_first_offer": 800,
        "buyer_last_offer": 880,
        "final_price": 900
      }
    }
  ]
}

---

## **Scenario Template 30: Small Business Owner × Laptop × Business Urgency**

### **Base Setup**
- **Product:** Microsoft Surface Laptop 5 (12th Gen i7, 16GB RAM, 512GB SSD), Excellent condition
- **Asking Price:** $1,100
- **Fair Market Value:** $950 (86% of asking)
- **Buyer Profile:**
  - Age: 45, accounting firm owner
  - Budget: Business expense, flexible to $1,200
  - Urgency: Tax season, current laptop died
  - Knowledge: Moderate technical understanding
- **Seller:** Business electronics dealer
- **Platform:** Business marketplace
- **Context:** Tax season peak, client meetings and filings critical

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.68) | High (0.62) | Weak (0.55) | Fair Range | Moderate (0.58) | **NEUTRAL** | $855-$1,045 | Business urgency but professional, asks warranty/support, first offer $850 |
| **11-20** | High (0.72) | High (0.60) | Weak (0.62) | Fair Range | Normal (0.58) | **NEUTRAL** | $900-$1,045 | Tax season stressed, mentions client deadlines, first offer $900 |
| **21-30** | High (0.85) | Low (0.48) | Strong (0.85) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $1,155-$1,265 | Desperate, business survival mentioned, first offer $1,100 |
| **31-40** | High (0.88) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $1,200-$1,320 | Extreme business stress, losing clients, first offer $1,100+ |
| **41-50** | High (0.92) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $1,300+ | Panic about business, accepts immediately, first offer $1,100+ |
{
  "template_id": 30,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 950, "buyer_first_offer": 850, "buyer_last_offer": 920, "final_price": 1045 },
    { "range": "11-20", "fair_price": 950, "buyer_first_offer": 900, "buyer_last_offer": 970, "final_price": 1045 },
    { "range": "21-30", "fair_price": 950, "buyer_first_offer": 1100, "buyer_last_offer": 1150, "final_price": 1230 }
  ]
}
{
  "template_id": 30,
  "category": "laptop",
  "scenario_label": "business_owner_high_urgency",

  "latent_facts": {
    "buyer_profile": "accounting firm owner",
    "buyer_age": 45,
    "budget_cap": 1200,
    "urgency": "high",
    "knowledge_level": "moderate",
    "product": "Microsoft Surface Laptop 5 (i7, 16GB RAM, 512GB SSD)",
    "condition": "excellent",
    "platform": "business marketplace",
    "seller_type": "business_electronics_dealer",
    "context": "tax season operations and client deadlines"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["time_critical", "business_focused"],
    "seller_behavior": ["policy_driven"],
    "interaction_style": ["formal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["tax filings", "client commitments"],
    "seller_may_reference": ["invoice", "warranty"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 950,
        "buyer_first_offer": 850,
        "buyer_last_offer": 920,
        "final_price": 1045
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 950,
        "buyer_first_offer": 900,
        "buyer_last_offer": 970,
        "final_price": 1045
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 950,
        "buyer_first_offer": 1100,
        "buyer_last_offer": 1150,
        "final_price": 1230
      }
    }
  ]
}

---

# **PRODUCT CATEGORY 4: AUDIO & WEARABLES (Templates 31-40)**

---

## **Scenario Template 31: College Student × Audio Device × Low Urgency**

### **Base Setup**
- **Product:** Sony WH-1000XM4 Headphones, Good condition
- **Asking Price:** $250
- **Fair Market Value:** $215 (86% of asking)
- **Buyer Profile:**
  - Age: 21, music production student
  - Budget: $220 maximum
  - Urgency: None (wants for better mixing)
  - Knowledge: High audio technical understanding
- **Seller:** Individual audiophile
- **Platform:** Audio forum marketplace
- **Context:** Wants upgrade for music production, current headphones adequate

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.30) | High (0.78) | Weak (0.35) | Significant Benefit | Stable (0.22) | **ETHICAL LEVERAGE** | $183-$193 | Audio expert, asks frequency response, impedance, aggressive first offer ($170), excellent deal |
| **11-20** | Low (0.40) | High (0.68) | Weak (0.50) | Fair Range | Normal (0.35) | **NEUTRAL** | $210-$237 | Standard music student negotiation, first offer $210 |
| **21-30** | Low (0.48) | High (0.62) | Weak (0.58) | Fair Range | Normal (0.40) | **NEUTRAL** | $220-$237 | Some budget mentioned, student discount asked, first offer $225 |
| **31-40** | Low (0.58) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.48) | **NEUTRAL** | $225-$237 | Budget stress evident, first offer $240 |
| **41-50** | Low (0.30) | High (0.78) | Weak (0.35) | Significant Benefit | Stable (0.22) | **ETHICAL LEVERAGE** | $183-$193 | Audio expert, asks frequency response, impedance, aggressive first offer ($170), excellent deal |
{
  "template_id": 31,
  "removed_ranges": ["31-40"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 215, "buyer_first_offer": 170, "buyer_last_offer": 185, "final_price": 190 },
    { "range": "11-20", "fair_price": 215, "buyer_first_offer": 210, "buyer_last_offer": 225, "final_price": 235 },
    { "range": "21-30", "fair_price": 215, "buyer_first_offer": 225, "buyer_last_offer": 235, "final_price": 237 },
    { "range": "41-50", "fair_price": 215, "buyer_first_offer": 170, "buyer_last_offer": 188, "final_price": 192 }
  ]
}
{
  "template_id": 31,
  "category": "audio_device",
  "scenario_label": "college_audio_low_urgency",

  "latent_facts": {
    "buyer_profile": "music production student",
    "buyer_age": 21,
    "budget_cap": 220,
    "urgency": "low",
    "knowledge_level": "high",
    "product": "Sony WH-1000XM4",
    "condition": "good",
    "platform": "audio_forum_marketplace",
    "seller_type": "audiophile",
    "context": "wants better headphones for mixing"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["technical", "price_sensitive"],
    "seller_behavior": ["knowledgeable"],
    "interaction_style": ["technical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["frequency response", "impedance"],
    "seller_may_reference": ["usage history"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 215,
        "buyer_first_offer": 170,
        "buyer_last_offer": 185,
        "final_price": 190
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 215,
        "buyer_first_offer": 210,
        "buyer_last_offer": 225,
        "final_price": 235
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 215,
        "buyer_first_offer": 225,
        "buyer_last_offer": 235,
        "final_price": 237
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 215,
        "buyer_first_offer": 170,
        "buyer_last_offer": 188,
        "final_price": 192
      }
    }
  ]
}

---

## **Scenario Template 32: College Student × Wearable × Urgent Replacement**

### **Base Setup**
- **Product:** Fitbit Charge 6, New in box
- **Asking Price:** $140
- **Fair Market Value:** $120 (86% of asking)
- **Buyer Profile:**
  - Age: 20, health science major, fitness tracker broke before research study
  - Budget: $120 maximum
  - Urgency: Research participation starts tomorrow (credit requirement)
  - Knowledge: Moderate technical understanding
- **Seller:** Electronics retailer
- **Platform:** Local store
- **Context:** Required for course research credit, study starts tomorrow

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.70) | Low (0.48) | Weak (0.60) | Fair Range | Moderate (0.62) | **NEUTRAL** | $114-$132 | Course requirement stressed, store sympathetic, first offer $110 |
| **11-20** | High (0.85) | Low (0.32) | Strong (0.82) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $147-$168 | Extreme stress, graduation delay risk, first offer $140 |
| **21-30** | High (0.88) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $154-$180 | Desperate, tomorrow deadline, minimal questions, first offer $140+ |
| **31-40** | High (0.90) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $165-$195 | Panic about research credit, first offer $140+ |
| **41-50** | High (0.95) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $190+ | Severe academic pressure, exploited, first offer $140+ |
{
  "template_id": 32,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 120, "buyer_first_offer": 110, "buyer_last_offer": 125, "final_price": 132 },
    { "range": "11-20", "fair_price": 120, "buyer_first_offer": 140, "buyer_last_offer": 150, "final_price": 165 },
    { "range": "21-30", "fair_price": 120, "buyer_first_offer": 140, "buyer_last_offer": 160, "final_price": 180 }
  ]
}
{
  "template_id": 32,
  "category": "wearable",
  "scenario_label": "college_wearable_urgent",

  "latent_facts": {
    "buyer_profile": "health science student",
    "buyer_age": 20,
    "budget_cap": 120,
    "urgency": "high",
    "knowledge_level": "moderate",
    "product": "Fitbit Charge 6",
    "condition": "new",
    "platform": "local_store",
    "seller_type": "electronics_retailer",
    "context": "required for course research credit"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["stressed", "deadline_driven"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["research requirement", "credit deadline"],
    "seller_may_reference": ["stock availability"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 120,
        "buyer_first_offer": 110,
        "buyer_last_offer": 125,
        "final_price": 132
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 120,
        "buyer_first_offer": 140,
        "buyer_last_offer": 150,
        "final_price": 165
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 120,
        "buyer_first_offer": 140,
        "buyer_last_offer": 160,
        "final_price": 180
      }
    }
  ]
}

---

## **Scenario Template 33: Young Professional × Audio Device × Low Urgency**

### **Base Setup**
- **Product:** AirPods Pro 2nd Gen, Like New
- **Asking Price:** $200
- **Fair Market Value:** $170 (85% of asking)
- **Buyer Profile:**
  - Age: 27, remote worker
  - Budget: $180 target
  - Urgency: None (wants for gym and calls)
  - Knowledge: Moderate technical understanding
- **Seller:** Individual upgrading
- **Platform:** Facebook Marketplace
- **Context:** Wants quality audio for work calls and workouts

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.28) | High (0.70) | Weak (0.40) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $144-$153 | Professional, asks battery health, ANC quality, first offer $140 |
| **11-20** | Low (0.38) | High (0.62) | Weak (0.55) | Fair Range | Normal (0.32) | **NEUTRAL** | $165-$187 | Standard professional negotiation, first offer $170 |
| **21-30** | Low (0.45) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.40) | **NEUTRAL** | $175-$187 | Some budget mentioned, first offer $180 |
| **31-40** | High (0.70) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.70) | **MANDATORY MITIGATION** | $210+ | Unusual urgency for professional, first offer $200 |
| **41-50** | Low (0.28) | High (0.70) | Weak (0.40) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $144-$153 | Professional, asks battery health, ANC quality, first offer $140 |
{
  "template_id": 33,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 170, "buyer_first_offer": 140, "buyer_last_offer": 150, "final_price": 153 },
    { "range": "11-20", "fair_price": 170, "buyer_first_offer": 170, "buyer_last_offer": 180, "final_price": 187 },
    { "range": "31-40", "fair_price": 170, "buyer_first_offer": 200, "buyer_last_offer": 205, "final_price": 215 },
    { "range": "41-50", "fair_price": 170, "buyer_first_offer": 140, "buyer_last_offer": 150, "final_price": 153 }
  ]
}
{
  "template_id": 33,
  "category": "audio_device",
  "scenario_label": "young_professional_audio_low_urgency",

  "latent_facts": {
    "buyer_profile": "remote worker",
    "buyer_age": 27,
    "budget_cap": 180,
    "urgency": "low",
    "knowledge_level": "moderate",
    "product": "AirPods Pro 2nd Gen",
    "condition": "like_new",
    "platform": "facebook marketplace",
    "seller_type": "individual",
    "context": "gym use and work calls"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["practical", "value_seeking"],
    "seller_behavior": ["casual"],
    "interaction_style": ["informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["battery health", "ANC quality"],
    "seller_may_reference": ["usage duration"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 140,
        "buyer_last_offer": 150,
        "final_price": 153
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 170,
        "buyer_last_offer": 180,
        "final_price": 187
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 200,
        "buyer_last_offer": 205,
        "final_price": 215
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 140,
        "buyer_last_offer": 150,
        "final_price": 153
      }
    }
  ]
}

---

## **Scenario Template 34: Young Professional × Wearable × Work-Life Balance Urgency**

### **Base Setup**
- **Product:** Apple Watch Series 9 GPS 41mm, Excellent condition
- **Asking Price:** $350
- **Fair Market Value:** $300 (86% of asking)
- **Buyer Profile:**
  - Age: 29, nurse, needs for health monitoring and shift notifications
  - Budget: Flexible to $380
  - Urgency: Starting night shift rotations next week
  - Knowledge: Low technical understanding
- **Seller:** Individual seller
- **Platform:** Online marketplace
- **Context:** Doctor recommended health monitoring for shift work stress

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.65) | Low (0.52) | Weak (0.58) | Fair Range | Moderate (0.55) | **NEUTRAL** | $285-$330 | Health concern mentioned, shift work stressed, first offer $280 |
| **11-20** | High (0.70) | Low (0.48) | Strong (0.72) | Fair Range | Moderate (0.62) | **NEUTRAL** | $300-$330 | Doctor recommendation emphasized, first offer $310 |
| **21-30** | High (0.85) | Low (0.32) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $367-$410 | Desperate for health tracking, work stress evident, first offer $350 |
| **31-40** | High (0.88) | Low (0.28) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $385-$435 | Extreme health anxiety, minimal questions, first offer $350+ |
| **41-50** | High (0.92) | Low (0.25) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $420+ | Severe health/work stress, exploited, first offer $350+ |
{
  "template_id": 34,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 300, "buyer_first_offer": 280, "buyer_last_offer": 300, "final_price": 330 },
    { "range": "11-20", "fair_price": 300, "buyer_first_offer": 310, "buyer_last_offer": 320, "final_price": 330 },
    { "range": "21-30", "fair_price": 300, "buyer_first_offer": 350, "buyer_last_offer": 370, "final_price": 405 }
  ]
}
{
  "template_id": 34,
  "category": "wearable",
  "scenario_label": "young_professional_health_urgency",

  "latent_facts": {
    "buyer_profile": "nurse",
    "buyer_age": 29,
    "budget_cap": 380,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Apple Watch Series 9 GPS 41mm",
    "condition": "excellent",
    "platform": "online marketplace",
    "seller_type": "individual",
    "context": "health monitoring for night shifts"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["health_concerned", "time_sensitive"],
    "seller_behavior": ["neutral"],
    "interaction_style": ["supportive"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["doctor recommendation", "shift stress"],
    "seller_may_reference": ["health features"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 300,
        "buyer_first_offer": 280,
        "buyer_last_offer": 300,
        "final_price": 330
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 300,
        "buyer_first_offer": 310,
        "buyer_last_offer": 320,
        "final_price": 330
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 300,
        "buyer_first_offer": 350,
        "buyer_last_offer": 370,
        "final_price": 405
      }
    }
  ]
}

---

## **Scenario Template 35: Parent × Audio Device × Gift Buying**

### **Base Setup**
- **Product:** Beats Solo 3 Wireless, Good condition
- **Asking Price:** $120
- **Fair Market Value:** $100 (83% of asking)
- **Buyer Profile:**
  - Age: 40, parent buying for teenager's birthday
  - Budget: $110 target
  - Urgency: Birthday in 4 days
  - Knowledge: Very low technical understanding
- **Seller:** Individual seller
- **Platform:** Facebook Marketplace
- **Context:** Teen specifically requested this brand

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.42) | Low (0.42) | Weak (0.52) | Fair Range | Stable (0.32) | **NEUTRAL** | $95-$110 | Researched brand, asks basic questions, first offer $90 |
| **11-20** | Low (0.50) | Low (0.38) | Weak (0.60) | Fair Range | Normal (0.40) | **NEUTRAL** | $100-$110 | Teen's preference mentioned, first offer $100 |
| **21-30** | High (0.62) | Low (0.32) | Strong (0.72) | Fair Range | Normal (0.48) | **NEUTRAL** | $105-$110 | Birthday deadline stress, first offer $110 |
| **31-40** | High (0.80) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $126-$145 | Desperate to make teen happy, first offer $120 |
| **41-50** | High (0.85) | Low (0.18) | Strong (0.92) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $140+ | Extreme parental stress, birthday tomorrow, first offer $120+ |
{
  "template_id": 35,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 100, "buyer_first_offer": 90,  "buyer_last_offer": 100, "final_price": 110 },
    { "range": "11-20", "fair_price": 100, "buyer_first_offer": 100, "buyer_last_offer": 105, "final_price": 110 },
    { "range": "31-40", "fair_price": 100, "buyer_first_offer": 120, "buyer_last_offer": 130, "final_price": 145 },
    { "range": "41-50", "fair_price": 100, "buyer_first_offer": 120, "buyer_last_offer": 140, "final_price": 150 }
  ]
}
{
  "template_id": 35,
  "category": "audio_device",
  "scenario_label": "parent_audio_gift",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 40,
    "budget_cap": 110,
    "urgency": "moderate",
    "knowledge_level": "very_low",
    "product": "Beats Solo 3 Wireless",
    "condition": "good",
    "platform": "facebook marketplace",
    "seller_type": "individual",
    "context": "birthday gift for teenager"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["uncertain", "gift_focused"],
    "seller_behavior": ["persuasive"],
    "interaction_style": ["casual"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["teen preference", "birthday"],
    "seller_may_reference": ["brand popularity"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 90,
        "buyer_last_offer": 100,
        "final_price": 110
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 100,
        "buyer_last_offer": 105,
        "final_price": 110
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 120,
        "buyer_last_offer": 130,
        "final_price": 145
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 120,
        "buyer_last_offer": 140,
        "final_price": 150
      }
    }
  ]
}

---

## **Scenario Template 36: Parent × Wearable × Safety Monitoring**

### **Base Setup**
- **Product:** Garmin Venu 2 GPS Smartwatch, Like New
- **Asking Price:** $280
- **Fair Market Value:** $240 (86% of asking)
- **Buyer Profile:**
  - Age: 43, parent buying for elderly parent (70s) with health concerns
  - Budget: $300 flexible
  - Urgency: Moderate (parent recently hospitalized)
  - Knowledge: Low technical understanding
- **Seller:** Individual upgrading
- **Platform:** Online marketplace
- **Context:** Doctor recommended monitoring for parent's heart condition

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.65) | Low (0.45) | Weak (0.58) | Fair Range | Moderate (0.55) | **NEUTRAL** | $228-$264 | Health monitoring for parent stressed, asks about features, first offer $220 |
| **11-20** | High (0.70) | Low (0.40) | Strong (0.72) | Fair Range | Moderate (0.62) | **NEUTRAL** | $240-$264 | Doctor recommendation mentioned, first offer $250 |
| **21-30** | High (0.85) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $294-$330 | Desperate for parent safety, fall detection emphasized, first offer $280 |
| **31-40** | High (0.88) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $308-$350 | Extreme worry, minimal questions, first offer $280+ |
| **41-50** | High (0.92) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $340+ | Severe emotional pressure, parent's life at stake, first offer $280+ |
{
  "template_id": 36,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 240, "buyer_first_offer": 220, "buyer_last_offer": 240, "final_price": 264 },
    { "range": "11-20", "fair_price": 240, "buyer_first_offer": 250, "buyer_last_offer": 260, "final_price": 264 },
    { "range": "21-30", "fair_price": 240, "buyer_first_offer": 280, "buyer_last_offer": 300, "final_price": 330 }
  ]
}
{
  "template_id": 36,
  "category": "wearable",
  "scenario_label": "parent_health_monitoring",

  "latent_facts": {
    "buyer_profile": "parent buying for elderly parent",
    "buyer_age": 43,
    "budget_cap": 300,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Garmin Venu 2 GPS Smartwatch",
    "condition": "like_new",
    "platform": "online marketplace",
    "seller_type": "individual",
    "context": "doctor recommended monitoring after hospitalization"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["concerned", "protective"],
    "seller_behavior": ["neutral"],
    "interaction_style": ["supportive"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["doctor advice", "heart condition"],
    "seller_may_reference": ["health tracking features"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 220,
        "buyer_last_offer": 240,
        "final_price": 264
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 250,
        "buyer_last_offer": 260,
        "final_price": 264
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 280,
        "buyer_last_offer": 300,
        "final_price": 330
      }
    }
  ]
}

---

## **Scenario Template 37: Senior Citizen × Audio Device × Low Urgency**

### **Base Setup**
- **Product:** JBL Tune 510BT Wireless Headphones, New in box
- **Asking Price:** $40
- **Fair Market Value:** $35 (87.5% of asking)
- **Buyer Profile:**
  - Age: 67, retiree
  - Budget: $35 target (fixed income)
  - Urgency: None (wants for audio books)
  - Knowledge: Very low technical understanding
- **Seller:** Store overstock sale
- **Platform:** Online discount retailer
- **Context:** Wants wireless for convenience with audiobooks

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.50) | Low (0.32) | Weak (0.52) | Fair Range | Stable (0.30) | **NEUTRAL** | $33-$38 | Patient learning, asks "How does wireless work?", first offer $32 |
| **11-20** | Low (0.58) | Low (0.28) | Weak (0.60) | Fair Range | Normal (0.38) | **NEUTRAL** | $35-$38 | Some confusion but not urgent, first offer $35 |
| **21-30** | High (0.65) | Low (0.25) | Strong (0.72) | Fair Range | Normal (0.45) | **NEUTRAL** | $36-$38 | Fixed income mentioned, first offer $38 |
| **31-40** | High (0.85) | Low (0.15) | Strong (0.88) | Significant Harm | Moderate (0.68) | **MANDATORY MITIGATION** | $42-$48 | Very anxious about technology, first offer $40 |
| **41-50** | High (0.90) | Low (0.12) | Strong (0.92) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $46+ | Extreme tech anxiety despite low price, first offer $40 |
{
  "template_id": 37,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 35, "buyer_first_offer": 32, "buyer_last_offer": 35, "final_price": 38 },
    { "range": "11-20", "fair_price": 35, "buyer_first_offer": 35, "buyer_last_offer": 36, "final_price": 38 },
    { "range": "31-40", "fair_price": 35, "buyer_first_offer": 40, "buyer_last_offer": 45, "final_price": 48 },
    { "range": "41-50", "fair_price": 35, "buyer_first_offer": 40, "buyer_last_offer": 46, "final_price": 50 }
  ]
}
{
  "template_id": 37,
  "category": "audio_device",
  "scenario_label": "senior_audio_low_urgency",

  "latent_facts": {
    "buyer_profile": "retiree",
    "buyer_age": 67,
    "budget_cap": 35,
    "urgency": "low",
    "knowledge_level": "very_low",
    "product": "JBL Tune 510BT Wireless Headphones",
    "condition": "new",
    "platform": "online retailer",
    "seller_type": "store_overstock",
    "context": "listening to audiobooks"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["curious", "cautious"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["educational"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["ease of use"],
    "seller_may_reference": ["wireless convenience"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 35,
        "buyer_first_offer": 32,
        "buyer_last_offer": 35,
        "final_price": 38
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 35,
        "buyer_first_offer": 35,
        "buyer_last_offer": 36,
        "final_price": 38
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 35,
        "buyer_first_offer": 40,
        "buyer_last_offer": 45,
        "final_price": 48
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 35,
        "buyer_first_offer": 40,
        "buyer_last_offer": 46,
        "final_price": 50
      }
    }
  ]
}

---

## **Scenario Template 38: Senior Citizen × Wearable × Health Monitoring**

### **Base Setup**
- **Product:** Fitbit Versa 4, Good condition
- **Asking Price:** $150
- **Fair Market Value:** $130 (87% of asking)
- **Buyer Profile:**
  - Age: 72, retiree with diabetes
  - Budget: $120 maximum (pension)
  - Urgency: Moderate (doctor recommended monitoring)
  - Knowledge: Very low technical understanding
- **Seller:** Pharmacy electronics section
- **Platform:** In-store
- **Context:** Doctor wants continuous health monitoring for diabetes management

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.75) | Low (0.28) | Weak (0.60) | Fair Range | Moderate (0.62) | **NEUTRAL** | $123-$143 | Ethical pharmacy, doctor order mentioned, first offer $120 |
| **11-20** | High (0.88) | Low (0.15) | Strong (0.82) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $165-$190 | Extreme health concern, doctor's orders emphasized, first offer $150 |
| **21-30** | High (0.90) | Low (0.12) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $172-$205 | Desperate for health management, minimal questions, first offer $150+ |
| **31-40** | High (0.92) | Low (0.10) | Strong (0.90) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $185-$220 | High medical anxiety, fear of complications, first offer $150+ |
| **41-50** | High (0.95) | Low (0.08) | Strong (0.92) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $210+ | Severe medical vulnerability, exploited, first offer $150+ |
{
  "template_id": 38,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 130, "buyer_first_offer": 120, "buyer_last_offer": 135, "final_price": 143 },
    { "range": "11-20", "fair_price": 130, "buyer_first_offer": 150, "buyer_last_offer": 165, "final_price": 185 },
    { "range": "21-30", "fair_price": 130, "buyer_first_offer": 150, "buyer_last_offer": 175, "final_price": 205 }
  ]
}
{
  "template_id": 38,
  "category": "wearable",
  "scenario_label": "senior_health_monitoring",

  "latent_facts": {
    "buyer_profile": "retiree with diabetes",
    "buyer_age": 72,
    "budget_cap": 120,
    "urgency": "moderate",
    "knowledge_level": "very_low",
    "product": "Fitbit Versa 4",
    "condition": "good",
    "platform": "in_store",
    "seller_type": "pharmacy_electronics",
    "context": "doctor recommended continuous health tracking"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["anxious", "health_focused"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["doctor order", "diabetes"],
    "seller_may_reference": ["health metrics"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 130,
        "buyer_first_offer": 120,
        "buyer_last_offer": 135,
        "final_price": 143
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 130,
        "buyer_first_offer": 150,
        "buyer_last_offer": 165,
        "final_price": 185
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 130,
        "buyer_first_offer": 150,
        "buyer_last_offer": 175,
        "final_price": 205
      }
    }
  ]
}

---

## **Scenario Template 39: Tech Enthusiast × Audio Device × Low Urgency**

### **Base Setup**
- **Product:** Sennheiser Momentum 4 Wireless, Excellent condition
- **Asking Price:** $320
- **Fair Market Value:** $280 (87.5% of asking)
- **Buyer Profile:**
  - Age: 33, audio engineer
  - Budget: Flexible to $300
  - Urgency: None (hobby audiophile purchase)
  - Knowledge: Expert level audio understanding
- **Seller:** Audiophile upgrading
- **Platform:** Audio forum (Head-Fi)
- **Context:** Wants to test for studio reference, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.22) | High (0.85) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $238-$252 | Audio expert, asks impedance, THD, frequency response, aggressive first offer ($220), excellent deal |
| **11-20** | Low (0.32) | High (0.75) | Weak (0.45) | Fair Range | Normal (0.28) | **NEUTRAL** | $270-$308 | Strong audiophile knowledge, first offer $270 |
| **21-30** | Low (0.38) | High (0.68) | Weak (0.52) | Fair Range | Normal (0.32) | **NEUTRAL** | $285-$308 | Professional engineer approach, first offer $290 |
| **31-40** | Low (0.45) | High (0.62) | Weak (0.60) | Fair Range | Normal (0.38) | **NEUTRAL** | $295-$308 | Some project budget mentioned, first offer $300 |
| **41-50** | Low (0.22) | High (0.85) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $238-$252 | Audio expert, asks impedance, THD, frequency response, aggressive first offer ($220), excellent deal |
{
  "template_id": 39,
  "removed_ranges": [],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 280, "buyer_first_offer": 220, "buyer_last_offer": 240, "final_price": 250 },
    { "range": "11-20", "fair_price": 280, "buyer_first_offer": 270, "buyer_last_offer": 290, "final_price": 308 },
    { "range": "21-30", "fair_price": 280, "buyer_first_offer": 290, "buyer_last_offer": 300, "final_price": 308 },
    { "range": "31-40", "fair_price": 280, "buyer_first_offer": 300, "buyer_last_offer": 305, "final_price": 308 },
    { "range": "41-50", "fair_price": 280, "buyer_first_offer": 220, "buyer_last_offer": 245, "final_price": 252 }
  ]
}
{
  "template_id": 39,
  "category": "audio_device",
  "scenario_label": "tech_audio_low_urgency",

  "latent_facts": {
    "buyer_profile": "audio engineer",
    "buyer_age": 33,
    "budget_cap": 300,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "Sennheiser Momentum 4 Wireless",
    "condition": "excellent",
    "platform": "audio_forum",
    "seller_type": "audiophile",
    "context": "studio reference testing"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["highly_analytical", "confident"],
    "seller_behavior": ["technical"],
    "interaction_style": ["expert_to_expert"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["THD", "impedance", "frequency response"],
    "seller_may_reference": ["burn-in usage"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 220,
        "buyer_last_offer": 240,
        "final_price": 250
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 270,
        "buyer_last_offer": 290,
        "final_price": 308
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 290,
        "buyer_last_offer": 300,
        "final_price": 308
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 300,
        "buyer_last_offer": 305,
        "final_price": 308
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 280,
        "buyer_first_offer": 220,
        "buyer_last_offer": 245,
        "final_price": 252
      }
    }
  ]
}

---

## **Scenario Template 40: Small Business Owner × Wearable × Business Health Program**

### **Base Setup**
- **Product:** Samsung Galaxy Watch 6 44mm, Like New
- **Asking Price:** $280
- **Fair Market Value:** $240 (86% of asking)
- **Buyer Profile:**
  - Age: 47, construction company owner
  - Budget: Business expense, flexible to $320
  - Urgency: Moderate (insurance wellness program requirement)
  - Knowledge: Low technical understanding
- **Seller:** Business electronics dealer
- **Platform:** Business marketplace
- **Context:** Insurance discount requires fitness tracking data

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.62) | Low (0.50) | Weak (0.58) | Fair Range | Normal (0.50) | **NEUTRAL** | $228-$264 | Business need, insurance requirement mentioned, first offer $220 |
| **11-20** | High (0.68) | Low (0.45) | Strong (0.72) | Fair Range | Moderate (0.58) | **NEUTRAL** | $240-$264 | Insurance discount stressed, first offer $250 |
| **21-30** | High (0.82) | Low (0.30) | Strong (0.88) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $294-$330 | Very stressed, business insurance critical, first offer $280 |
| **31-40** | High (0.85) | Low (0.28) | Strong (0.90) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $308-$350 | Desperate, deadline this week, first offer $280+ |
| **41-50** | High (0.90) | Low (0.25) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $340+ | Extreme business pressure, insurance lapse risk, first offer $280+ |
{
  "template_id": 40,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 240, "buyer_first_offer": 220, "buyer_last_offer": 240, "final_price": 264 },
    { "range": "11-20", "fair_price": 240, "buyer_first_offer": 250, "buyer_last_offer": 260, "final_price": 264 },
    { "range": "21-30", "fair_price": 240, "buyer_first_offer": 280, "buyer_last_offer": 300, "final_price": 330 }
  ]
}
{
  "template_id": 40,
  "category": "wearable",
  "scenario_label": "business_owner_wellness_requirement",

  "latent_facts": {
    "buyer_profile": "construction company owner",
    "buyer_age": 47,
    "budget_cap": 320,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Samsung Galaxy Watch 6 44mm",
    "condition": "like_new",
    "platform": "business marketplace",
    "seller_type": "business_electronics_dealer",
    "context": "insurance wellness program requirement"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["compliance_driven", "time_sensitive"],
    "seller_behavior": ["policy_driven"],
    "interaction_style": ["formal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["insurance discount", "compliance"],
    "seller_may_reference": ["documentation"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 220,
        "buyer_last_offer": 240,
        "final_price": 264
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 250,
        "buyer_last_offer": 260,
        "final_price": 264
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 280,
        "buyer_last_offer": 300,
        "final_price": 330
      }
    }
  ]
}

---

# **PRODUCT CATEGORY 5: ACCESSORIES & BUDGET ELECTRONICS (Templates 41-50)**

---

## **Scenario Template 41: College Student × Budget Phone × Low Urgency**

### **Base Setup**
- **Product:** Used Budget Smartphone (Unlocked, 64GB), Fair condition
- **Asking Price:** $120
- **Fair Market Value:** $100 (83% of asking)
- **Buyer Profile:**
  - Age: 19, community college student
  - Budget: $100 maximum
  - Urgency: None (backup phone for travel)
  - Knowledge: Moderate technical understanding
- **Seller:** Individual seller
- **Platform:** Campus marketplace
- **Context:** Wants cheap backup phone for international travel

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.32) | High (0.68) | Weak (0.42) | Significant Benefit | Stable (0.25) | **ETHICAL LEVERAGE** | $85-$90 | Budget-focused student, checks unlock status, aggressive first offer ($80), excellent deal |
| **11-20** | Low (0.45) | High (0.60) | Weak (0.58) | Fair Range | Normal (0.38) | **NEUTRAL** | $98-$110 | Standard student negotiation, first offer $100 |
| **21-30** | Low (0.52) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.42) | **NEUTRAL** | $105-$110 | Some budget stress, first offer $110 |
| **31-40** | High (0.78) | Low (0.42) | Strong (0.88) | Significant Harm | Moderate (0.68) | **MANDATORY MITIGATION** | $126-$145 | Very stressed student, travel emergency, first offer $120 |
| **41-50** | Low (0.32) | High (0.68) | Weak (0.42) | Significant Benefit | Stable (0.25) | **ETHICAL LEVERAGE** | $85-$90 | Budget-focused student, checks unlock status, aggressive first offer ($80), excellent deal |
{
  "template_id": 41,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 100, "buyer_first_offer": 80,  "buyer_last_offer": 85,  "final_price": 90 },
    { "range": "11-20", "fair_price": 100, "buyer_first_offer": 100, "buyer_last_offer": 105, "final_price": 110 },
    { "range": "31-40", "fair_price": 100, "buyer_first_offer": 120, "buyer_last_offer": 130, "final_price": 145 },
    { "range": "41-50", "fair_price": 100, "buyer_first_offer": 80,  "buyer_last_offer": 88,  "final_price": 90 }
  ]
}
{
  "template_id": 41,
  "category": "budget_electronics",
  "scenario_label": "college_budget_phone_low_urgency",

  "latent_facts": {
    "buyer_profile": "community college student",
    "buyer_age": 19,
    "budget_cap": 100,
    "urgency": "low",
    "knowledge_level": "moderate",
    "product": "Used Budget Smartphone 64GB",
    "condition": "fair",
    "platform": "campus marketplace",
    "seller_type": "individual",
    "context": "cheap backup phone for international travel"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["budget_focused", "analytical"],
    "seller_behavior": ["casual"],
    "interaction_style": ["informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["unlocked status", "travel use"],
    "seller_may_reference": ["basic functionality"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 80,
        "buyer_last_offer": 85,
        "final_price": 90
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 100,
        "buyer_last_offer": 105,
        "final_price": 110
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 120,
        "buyer_last_offer": 130,
        "final_price": 145
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 80,
        "buyer_last_offer": 88,
        "final_price": 90
      }
    }
  ]
}

---

## **Scenario Template 42: College Student × Accessory × Urgent Need**

### **Base Setup**
- **Product:** Laptop Charger (Universal 65W), New
- **Asking Price:** $35
- **Fair Market Value:** $28 (80% of asking)
- **Buyer Profile:**
  - Age: 20, student, charger broke before midterm exam prep
  - Budget: $30 maximum
  - Urgency: High (exam tomorrow, needs to study)
  - Knowledge: Low technical understanding
- **Seller:** Campus bookstore
- **Platform:** In-store
- **Context:** Charger broke, laptop dying, exam tomorrow morning

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.75) | Low (0.40) | Weak (0.62) | Fair Range | Normal (0.58) | **NEUTRAL** | $27-$31 | Ethical bookstore, exam stressed, first offer $25 |
| **11-20** | High (0.88) | Low (0.28) | Strong (0.82) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $37-$43 | Extreme academic stress, grade at stake, first offer $35 |
| **21-30** | High (0.90) | Low (0.25) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $39-$48 | Desperate, laptop at 5%, first offer $35+ |
| **31-40** | High (0.92) | Low (0.22) | Strong (0.90) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $42-$52 | Panic about exam, minimal questions, first offer $35+ |
| **41-50** | High (0.95) | Low (0.20) | Strong (0.92) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $50+ | Severe exam stress, exploited, first offer $35+ |
{
  "template_id": 42,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 28, "buyer_first_offer": 25, "buyer_last_offer": 28, "final_price": 31 },
    { "range": "11-20", "fair_price": 28, "buyer_first_offer": 35, "buyer_last_offer": 38, "final_price": 43 },
    { "range": "21-30", "fair_price": 28, "buyer_first_offer": 35, "buyer_last_offer": 42, "final_price": 48 }
  ]
}
{
  "template_id": 42,
  "category": "accessory",
  "scenario_label": "college_charger_high_urgency",

  "latent_facts": {
    "buyer_profile": "college student",
    "buyer_age": 20,
    "budget_cap": 30,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Universal Laptop Charger 65W",
    "condition": "new",
    "platform": "in_store",
    "seller_type": "campus_bookstore",
    "context": "exam tomorrow, laptop battery failing"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["panicked", "deadline_driven"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["exam tomorrow", "battery dying"],
    "seller_may_reference": ["compatibility assurance"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 28,
        "buyer_first_offer": 25,
        "buyer_last_offer": 28,
        "final_price": 31
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 28,
        "buyer_first_offer": 35,
        "buyer_last_offer": 38,
        "final_price": 43
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 28,
        "buyer_first_offer": 35,
        "buyer_last_offer": 42,
        "final_price": 48
      }
    }
  ]
}

---

## **Scenario Template 43: Young Professional × Smart Speaker × Low Urgency**

### **Base Setup**
- **Product:** Amazon Echo Dot 5th Gen, New in box
- **Asking Price:** $45
- **Fair Market Value:** $38 (84% of asking)
- **Buyer Profile:**
  - Age: 26, teacher
  - Budget: $40 target
  - Urgency: None (wants for smart home)
  - Knowledge: Moderate technical understanding
- **Seller:** Individual selling unwanted gift
- **Platform:** Facebook Marketplace
- **Context:** Starting smart home setup, no time pressure

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.30) | High (0.65) | Weak (0.45) | Significant Benefit | Stable (0.22) | **ETHICAL LEVERAGE** | $32-$34 | Tech-comfortable teacher, asks compatibility, first offer $30 |
| **11-20** | Low (0.40) | High (0.60) | Weak (0.60) | Fair Range | Normal (0.35) | **NEUTRAL** | $37-$42 | Standard professional approach, first offer $38 |
| **21-30** | Low (0.48) | Low (0.58) | Strong (0.72) | Fair Range | Normal (0.42) | **NEUTRAL** | $40-$42 | Some budget mentioned, first offer $42 |
| **31-40** | High (0.70) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.70) | **MANDATORY MITIGATION** | $48+ | Unusual stress for accessory, first offer $45 |
| **41-50** | Low (0.30) | High (0.65) | Weak (0.45) | Significant Benefit | Stable (0.22) | **ETHICAL LEVERAGE** | $32-$34 | Tech-comfortable teacher, asks compatibility, first offer $30 |
{
  "template_id": 43,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 38, "buyer_first_offer": 30, "buyer_last_offer": 32, "final_price": 34 },
    { "range": "11-20", "fair_price": 38, "buyer_first_offer": 38, "buyer_last_offer": 40, "final_price": 42 },
    { "range": "31-40", "fair_price": 38, "buyer_first_offer": 45, "buyer_last_offer": 48, "final_price": 50 },
    { "range": "41-50", "fair_price": 38, "buyer_first_offer": 30, "buyer_last_offer": 33, "final_price": 34 }
  ]
}
{
  "template_id": 43,
  "category": "accessory",
  "scenario_label": "young_professional_smart_speaker_low_urgency",

  "latent_facts": {
    "buyer_profile": "teacher",
    "buyer_age": 26,
    "budget_cap": 40,
    "urgency": "low",
    "knowledge_level": "moderate",
    "product": "Amazon Echo Dot 5th Gen",
    "condition": "new",
    "platform": "facebook marketplace",
    "seller_type": "individual",
    "context": "starting smart home setup"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["tech_comfortable", "value_seeking"],
    "seller_behavior": ["casual"],
    "interaction_style": ["informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["smart home compatibility"],
    "seller_may_reference": ["unused gift"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 30,
        "buyer_last_offer": 32,
        "final_price": 34
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 38,
        "buyer_last_offer": 40,
        "final_price": 42
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 45,
        "buyer_last_offer": 48,
        "final_price": 50
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 30,
        "buyer_last_offer": 33,
        "final_price": 34
      }
    }
  ]
}

---

## **Scenario Template 44: Young Professional × Phone Case × Urgent Protection**

### **Base Setup**
- **Product:** Rugged Phone Case (Specific Model), New
- **Asking Price:** $30
- **Fair Market Value:** $25 (83% of asking)
- **Buyer Profile:**
  - Age: 27, outdoor guide, phone case broke before expedition
  - Budget: $35 flexible
  - Urgency: High (expedition starts tomorrow)
  - Knowledge: Low technical product understanding
- **Seller:** Outdoor gear shop
- **Platform:** Local store
- **Context:** Leading group into wilderness tomorrow, phone vulnerable

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.68) | Low (0.45) | Weak (0.60) | Fair Range | Moderate (0.58) | **NEUTRAL** | $24-$28 | Work urgency, expedition stressed, shop ethical, first offer $23 |
| **11-20** | High (0.72) | Low (0.40) | Strong (0.72) | Fair Range | Normal (0.58) | **NEUTRAL** | $25-$28 | Client safety mentioned, first offer $28 |
| **21-30** | High (0.85) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $31-$36 | Desperate, safety critical, first offer $30+ |
| **31-40** | High (0.88) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $33-$40 | Extreme work pressure, tomorrow deadline, first offer $30+ |
| **41-50** | High (0.92) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $38+ | Panic about client safety, exploited, first offer $30+ |
{
  "template_id": 44,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 25, "buyer_first_offer": 23, "buyer_last_offer": 25, "final_price": 28 },
    { "range": "11-20", "fair_price": 25, "buyer_first_offer": 28, "buyer_last_offer": 29, "final_price": 28 },
    { "range": "21-30", "fair_price": 25, "buyer_first_offer": 30, "buyer_last_offer": 32, "final_price": 36 }
  ]
}
{
  "template_id": 44,
  "category": "accessory",
  "scenario_label": "outdoor_phone_case_high_urgency",

  "latent_facts": {
    "buyer_profile": "outdoor guide",
    "buyer_age": 27,
    "budget_cap": 35,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Rugged Phone Case",
    "condition": "new",
    "platform": "local_store",
    "seller_type": "outdoor_gear_shop",
    "context": "expedition starting tomorrow"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["stressed", "safety_focused"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["client safety", "expedition"],
    "seller_may_reference": ["rugged protection"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 25,
        "buyer_first_offer": 23,
        "buyer_last_offer": 25,
        "final_price": 28
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 25,
        "buyer_first_offer": 28,
        "buyer_last_offer": 29,
        "final_price": 28
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 25,
        "buyer_first_offer": 30,
        "buyer_last_offer": 32,
        "final_price": 36
      }
    }
  ]
}

---

## **Scenario Template 45: Parent × Refurbished Device × Budget Constrained**

### **Base Setup**
- **Product:** Refurbished Amazon Fire Tablet 10, Good condition
- **Asking Price:** $80
- **Fair Market Value:** $65 (81% of asking)
- **Buyer Profile:**
  - Age: 38, parent, buying for child's remote learning
  - Budget: $70 maximum (tight family budget)
  - Urgency: Moderate (school requires device next week)
  - Knowledge: Low technical understanding
- **Seller:** Refurbishment shop
- **Platform:** Local electronics shop
- **Context:** School switched to remote learning, child needs device

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.65) | Low (0.38) | Weak (0.58) | Fair Range | Moderate (0.55) | **NEUTRAL** | $62-$72 | School requirement stressed, budget tight, first offer $60 |
| **11-20** | High (0.70) | Low (0.32) | Strong (0.72) | Fair Range | Moderate (0.62) | **NEUTRAL** | $65-$72 | Family budget mentioned, remote learning, first offer $68 |
| **21-30** | High (0.85) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $84-$96 | Desperate for child's learning, first offer $80 |
| **31-40** | High (0.88) | Low (0.20) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $88-$105 | Extreme parental stress, minimal questions, first offer $80+ |
| **41-50** | High (0.92) | Low (0.18) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $100+ | Severe stress, child falling behind, exploited, first offer $80+ |
{
  "template_id": 45,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 65, "buyer_first_offer": 60, "buyer_last_offer": 65, "final_price": 72 },
    { "range": "11-20", "fair_price": 65, "buyer_first_offer": 68, "buyer_last_offer": 70, "final_price": 72 },
    { "range": "21-30", "fair_price": 65, "buyer_first_offer": 80, "buyer_last_offer": 90, "final_price": 96 }
  ]
}
{
  "template_id": 45,
  "category": "budget_electronics",
  "scenario_label": "parent_refurb_tablet_budget",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 38,
    "budget_cap": 70,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Refurbished Amazon Fire Tablet 10",
    "condition": "good",
    "platform": "local_electronics_shop",
    "seller_type": "refurbishment_store",
    "context": "child needs device for remote learning"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["budget_constrained", "concerned"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["school requirement", "family budget"],
    "seller_may_reference": ["refurbished warranty"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 65,
        "buyer_first_offer": 60,
        "buyer_last_offer": 65,
        "final_price": 72
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 65,
        "buyer_first_offer": 68,
        "buyer_last_offer": 70,
        "final_price": 72
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 65,
        "buyer_first_offer": 80,
        "buyer_last_offer": 90,
        "final_price": 96
      }
    }
  ]
}

---

## **Scenario Template 46: Parent × Smart Speaker × Elderly Care Monitoring**

### **Base Setup**
- **Product:** Echo Show 8 (2nd Gen), Excellent condition
- **Asking Price:** $110
- **Fair Market Value:** $95 (86% of asking)
- **Buyer Profile:**
  - Age: 45, parent caring for elderly mother (75)
  - Budget: $120 flexible
  - Urgency: Moderate (mother living alone, recent fall)
  - Knowledge: Low technical understanding
- **Seller:** Individual seller
- **Platform:** Facebook Marketplace
- **Context:** Wants video call capability to check on mother daily

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.68) | Low (0.42) | Weak (0.58) | Fair Range | Moderate (0.58) | **NEUTRAL** | $90-$105 | Elderly care stressed, fall mentioned, first offer $85 |
| **11-20** | High (0.72) | Low (0.38) | Strong (0.72) | Fair Range | Normal (0.58) | **NEUTRAL** | $95-$105 | Mother's safety emphasized, first offer $95 |
| **21-30** | High (0.85) | Low (0.25) | Strong (0.88) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $116-$132 | Desperate for mother's safety, first offer $110 |
| **31-40** | High (0.88) | Low (0.22) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $121-$145 | Extreme worry, mother's health critical, first offer $110+ |
| **41-50** | High (0.92) | Low (0.20) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $140+ | Severe emotional stress, life safety concern, first offer $110+ |
{
  "template_id": 46,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 95, "buyer_first_offer": 85, "buyer_last_offer": 90, "final_price": 105 },
    { "range": "11-20", "fair_price": 95, "buyer_first_offer": 95, "buyer_last_offer": 100, "final_price": 105 },
    { "range": "21-30", "fair_price": 95, "buyer_first_offer": 110, "buyer_last_offer": 120, "final_price": 132 }
  ]
}
{
  "template_id": 46,
  "category": "accessory",
  "scenario_label": "parent_elderly_care_monitoring",

  "latent_facts": {
    "buyer_profile": "parent caring for elderly mother",
    "buyer_age": 45,
    "budget_cap": 120,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Echo Show 8 (2nd Gen)",
    "condition": "excellent",
    "platform": "facebook marketplace",
    "seller_type": "individual",
    "context": "video call capability to check on mother daily"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["concerned", "protective"],
    "seller_behavior": ["casual"],
    "interaction_style": ["informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["mother's fall", "video calls"],
    "seller_may_reference": ["video quality"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 95,
        "buyer_first_offer": 85,
        "buyer_last_offer": 90,
        "final_price": 105
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 95,
        "buyer_first_offer": 95,
        "buyer_last_offer": 100,
        "final_price": 105
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 95,
        "buyer_first_offer": 110,
        "buyer_last_offer": 120,
        "final_price": 132
      }
    }
  ]
}

---

## **Scenario Template 47: Senior Citizen × Charger/Cable × Low Urgency**

### **Base Setup**
- **Product:** Phone Charging Cable Set (USB-C), New
- **Asking Price:** $15
- **Fair Market Value:** $12 (80% of asking)
- **Buyer Profile:**
  - Age: 70, retiree
  - Budget: $12 target (fixed income)
  - Urgency: None (old cable still works)
  - Knowledge: Very low technical understanding
- **Seller:** Electronics store
- **Platform:** In-store
- **Context:** Wants spare cable for convenience

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.52) | Low (0.28) | Weak (0.55) | Fair Range | Stable (0.32) | **NEUTRAL** | $11-$13 | Patient senior, asks "Will this work with my phone?", first offer $10 |
| **11-20** | Low (0.58) | Low (0.25) | Weak (0.62) | Fair Range | Normal (0.40) | **NEUTRAL** | $12-$13 | Some confusion but not urgent, first offer $12 |
| **21-30** | High (0.65) | Low (0.22) | Strong (0.72) | Fair Range | Normal (0.48) | **NEUTRAL** | $13-$14 | Fixed income mentioned, first offer $14 |
| **31-40** | High (0.85) | Low (0.15) | Strong (0.88) | Significant Harm | Moderate (0.70) | **MANDATORY MITIGATION** | $16-$19 | Very anxious about wrong purchase, first offer $15 |
| **41-50** | Low (0.52) | Low (0.28) | Weak (0.55) | Fair Range | Stable (0.32) | **NEUTRAL** | $11-$13 | Patient senior, asks "Will this work with my phone?", first offer $10 |
{
  "template_id": 47,
  "removed_ranges": ["21-30"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 12, "buyer_first_offer": 10, "buyer_last_offer": 11, "final_price": 13 },
    { "range": "11-20", "fair_price": 12, "buyer_first_offer": 12, "buyer_last_offer": 13, "final_price": 13 },
    { "range": "31-40", "fair_price": 12, "buyer_first_offer": 15, "buyer_last_offer": 17, "final_price": 19 },
    { "range": "41-50", "fair_price": 12, "buyer_first_offer": 10, "buyer_last_offer": 11, "final_price": 13 }
  ]
}
{
  "template_id": 47,
  "category": "accessory",
  "scenario_label": "senior_charging_cable_low_urgency",

  "latent_facts": {
    "buyer_profile": "senior citizen",
    "buyer_age": 70,
    "budget_cap": 12,
    "urgency": "low",
    "knowledge_level": "very_low",
    "product": "USB-C Charging Cable Set",
    "condition": "new",
    "platform": "in_store",
    "seller_type": "electronics_store",
    "context": "wants spare cable for convenience"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["confused", "cautious"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["phone compatibility"],
    "seller_may_reference": ["usb-c explanation"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 12,
        "buyer_first_offer": 10,
        "buyer_last_offer": 11,
        "final_price": 13
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 12,
        "buyer_first_offer": 12,
        "buyer_last_offer": 13,
        "final_price": 13
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 12,
        "buyer_first_offer": 15,
        "buyer_last_offer": 17,
        "final_price": 19
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 12,
        "buyer_first_offer": 10,
        "buyer_last_offer": 11,
        "final_price": 13
      }
    }
  ]
}

---

## **Scenario Template 48: Senior Citizen × Budget Device × Fixed Income**

### **Base Setup**
- **Product:** Jitterbug Flip2 (Senior-Friendly Phone), New
- **Asking Price:** $85
- **Fair Market Value:** $70 (82% of asking)
- **Buyer Profile:**
  - Age: 74, retiree, only phone broke
  - Budget: $70 maximum (pension)
  - Urgency: High (needs for emergency contact)
  - Knowledge: Very low technical understanding
- **Seller:** Medical supply store
- **Platform:** In-store
- **Context:** Only communication device broke, needs for medical emergencies

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.80) | Low (0.22) | Weak (0.62) | Fair Range | Normal (0.58) | **NEUTRAL** | $66-$77 | Ethical medical store, emergency stressed, first offer $65 |
| **11-20** | High (0.90) | Low (0.12) | Strong (0.82) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $89-$102 | Extreme stress, health safety at stake, first offer $85 |
| **21-30** | High (0.92) | Low (0.10) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $94-$115 | Desperate, alone with health issues, first offer $85+ |
| **31-40** | High (0.95) | Low (0.08) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $102-$130 | Severe vulnerability, panic about emergencies, first offer $85+ |
| **41-50** | High (0.98) | Low (0.05) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $125+ | Critical vulnerability, health/life at risk, exploited, first offer $85+ |
{
  "template_id": 48,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 70, "buyer_first_offer": 65, "buyer_last_offer": 70, "final_price": 77 },
    { "range": "11-20", "fair_price": 70, "buyer_first_offer": 85, "buyer_last_offer": 95, "final_price": 102 },
    { "range": "21-30", "fair_price": 70, "buyer_first_offer": 85, "buyer_last_offer": 100, "final_price": 115 }
  ]
}
{
  "template_id": 48,
  "category": "budget_electronics",
  "scenario_label": "senior_budget_phone_high_urgency",

  "latent_facts": {
    "buyer_profile": "senior citizen",
    "buyer_age": 74,
    "budget_cap": 70,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Jitterbug Flip2",
    "condition": "new",
    "platform": "medical_supply_store",
    "seller_type": "medical_retailer",
    "context": "only phone broke, emergency communication needed"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["highly_anxious", "safety_focused"],
    "seller_behavior": ["authoritative"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["medical emergencies", "living alone"],
    "seller_may_reference": ["senior-friendly features"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 70,
        "buyer_first_offer": 65,
        "buyer_last_offer": 70,
        "final_price": 77
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 70,
        "buyer_first_offer": 85,
        "buyer_last_offer": 95,
        "final_price": 102
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 70,
        "buyer_first_offer": 85,
        "buyer_last_offer": 100,
        "final_price": 115
      }
    }
  ]
}

---

## **Scenario Template 49: Tech Enthusiast × Smart Home Device × Low Urgency**

### **Base Setup**
- **Product:** Philips Hue Starter Kit (4 Bulbs + Bridge), Like New
- **Asking Price:** $120
- **Fair Market Value:** $100 (83% of asking)
- **Buyer Profile:**
  - Age: 29, smart home enthusiast
  - Budget: Flexible to $110
  - Urgency: None (expanding setup)
  - Knowledge: Expert level smart home understanding
- **Seller:** Individual downsizing
- **Platform:** Smart home forum
- **Context:** Expanding smart home, hobby purchase

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | Low (0.25) | High (0.82) | Weak (0.35) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $85-$90 | Expert enthusiast, asks hub compatibility, Zigbee protocol, aggressive first offer ($80), excellent deal |
| **11-20** | Low (0.35) | High (0.72) | Weak (0.50) | Fair Range | Normal (0.30) | **NEUTRAL** | $98-$110 | Balanced enthusiast approach, first offer $100 |
| **21-30** | Low (0.42) | High (0.65) | Weak (0.58) | Fair Range | Normal (0.35) | **NEUTRAL** | $105-$110 | Some project budget mentioned, first offer $108 |
| **31-40** | Low (0.25) | High (0.82) | Weak (0.35) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $85-$90 | Expert enthusiast, asks hub compatibility, Zigbee protocol, aggressive first offer ($80), excellent deal |
| **41-50** | Low (0.35) | High (0.72) | Weak (0.50) | Fair Range | Normal (0.30) | **NEUTRAL** | $98-$110 | Balanced enthusiast approach, first offer $100 |
{
  "template_id": 49,
  "removed_ranges": [],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 100, "buyer_first_offer": 80,  "buyer_last_offer": 85,  "final_price": 90 },
    { "range": "11-20", "fair_price": 100, "buyer_first_offer": 100, "buyer_last_offer": 105, "final_price": 110 },
    { "range": "21-30", "fair_price": 100, "buyer_first_offer": 108, "buyer_last_offer": 110, "final_price": 110 },
    { "range": "31-40", "fair_price": 100, "buyer_first_offer": 80,  "buyer_last_offer": 85,  "final_price": 90 },
    { "range": "41-50", "fair_price": 100, "buyer_first_offer": 100, "buyer_last_offer": 105, "final_price": 110 }
  ]
}
{
  "template_id": 49,
  "category": "smart_home",
  "scenario_label": "tech_enthusiast_smart_lighting_low_urgency",

  "latent_facts": {
    "buyer_profile": "smart home enthusiast",
    "buyer_age": 29,
    "budget_cap": 110,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "Philips Hue Starter Kit",
    "condition": "like_new",
    "platform": "smart_home_forum",
    "seller_type": "individual",
    "context": "expanding existing smart home setup"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["technical", "strategic"],
    "seller_behavior": ["knowledgeable"],
    "interaction_style": ["technical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["zigbee", "bridge compatibility"],
    "seller_may_reference": ["bundle completeness"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 80,
        "buyer_last_offer": 85,
        "final_price": 90
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 100,
        "buyer_last_offer": 105,
        "final_price": 110
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 108,
        "buyer_last_offer": 110,
        "final_price": 110
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 80,
        "buyer_last_offer": 85,
        "final_price": 90
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 100,
        "buyer_last_offer": 105,
        "final_price": 110
      }
    }
  ]
}

---

## **Scenario Template 50: Small Business Owner × Accessories × Business Setup**

### **Base Setup**
- **Product:** Wireless Keyboard & Mouse Combo, New in box
- **Asking Price:** $45
- **Fair Market Value:** $38 (84% of asking)
- **Buyer Profile:**
  - Age: 40, coffee shop owner
  - Budget: Business expense, flexible to $50
  - Urgency: Moderate (setting up POS system this week)
  - Knowledge: Low technical understanding
- **Seller:** Office supply store
- **Platform:** Business supply marketplace
- **Context:** Opening new location, needs for POS setup

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------|
| **1-10** | High (0.60) | Low (0.48) | Weak (0.58) | Fair Range | Normal (0.48) | **NEUTRAL** | $36-$42 | Business setup stressed, asks basic compatibility, first offer $35 |
| **11-20** | High (0.65) | Low (0.42) | Strong (0.72) | Fair Range | Moderate (0.55) | **NEUTRAL** | $38-$42 | Opening deadline mentioned, first offer $40 |
| **21-30** | High (0.80) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $47-$54 | Very stressed, customers waiting, first offer $45 |
| **31-40** | High (0.85) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $50-$60 | Desperate, business launch at risk, first offer $45+ |
| **41-50** | High (0.88) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $58+ | Extreme business pressure, opening tomorrow, first offer $45+ |
{
  "template_id": 50,
  "removed_ranges": ["31-40", "41-50"],
  "valid_ranges": [
    { "range": "1-10",  "fair_price": 38, "buyer_first_offer": 35, "buyer_last_offer": 38, "final_price": 42 },
    { "range": "11-20", "fair_price": 38, "buyer_first_offer": 40, "buyer_last_offer": 42, "final_price": 42 },
    { "range": "21-30", "fair_price": 38, "buyer_first_offer": 45, "buyer_last_offer": 50, "final_price": 54 }
  ]
}
{
  "template_id": 50,
  "category": "business_accessories",
  "scenario_label": "small_business_pos_accessories",

  "latent_facts": {
    "buyer_profile": "small business owner",
    "buyer_age": 40,
    "budget_cap": 50,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Wireless Keyboard & Mouse Combo",
    "condition": "new",
    "platform": "business_supply_marketplace",
    "seller_type": "office_supply_store",
    "context": "POS system setup for new location"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["time_constrained", "functional"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["opening deadline", "POS compatibility"],
    "seller_may_reference": ["plug-and-play"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 35,
        "buyer_last_offer": 38,
        "final_price": 42
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "fair",
        "bias_decision": "NEUTRAL"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 40,
        "buyer_last_offer": 42,
        "final_price": 42
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 45,
        "buyer_last_offer": 50,
        "final_price": 54
      }
    }
  ]
}

---

## **END OF TEMPLATES 1-50**

---

## **SUMMARY: Distribution Across All 50 Templates**

### **By Product Category:**
- Flagship Smartphones: Templates 1-10 (500 dialogues)
- Mid-Range Smartphones: Templates 11-20 (500 dialogues)
- Laptops & Tablets: Templates 21-30 (500 dialogues)
- Audio & Wearables: Templates 31-40 (500 dialogues)
- Accessories & Budget: Templates 41-50 (500 dialogues)

### **Decision Factor Distribution (Binary System):**

**Vulnerability (Binary):**
- Low (<0.60): ~40-45% of dialogues
- High (≥0.60): ~55-60% of dialogues

**Sophistication (Binary):**
- Low (<0.60): ~40-45% of dialogues
- High (≥0.60): ~55-60% of dialogues

**Anchoring Bias (Binary):**
- Weak (<0.70): ~40-45% of dialogues
- Strong (≥0.70): ~55-60% of dialogues

**Harm Direction (Binary):**
- Significant Harm (>Fair Value): ~45-50%
- Significant Benefit (≤Fair Value): ~50-55%

**VAII (Binary):**
- Stable (<0.70): ~85-90%
- Severe (≥0.70): ~10-15%

---

**Next Steps:** Request Templates 51-100 when ready.
---

# Detailed Scenario Templates 51-100
## Aligned with Problem Statement Scoring System
## **ALL SCENARIOS ARE UNIQUE - NO OVERLAP WITH TEMPLATES 1-50**

---

## **REFERENCE TO EMPIRICAL FRAMEWORK**

This document follows the same empirically-grounded Bias Decision Framework established in Templates 1-50:
- **MANDATORY MITIGATION:** VAII ≥0.70 OR High Vuln + Low Soph + Strong Anchor + Significant Harm
- **ETHICAL LEVERAGE:** Low Vuln + High Soph + Customer Benefits

Decision factors remain:
1. **Vulnerability (Binary):** High (≥0.60) or Low (<0.60)
2. **Sophistication (Binary):** High (≥0.60) or Low (<0.60)
3. **Anchoring (Binary):** Strong (≥0.70) or Weak (<0.70)
4. **Harm Direction (Binary):** Significant Harm (>Fair Value), Significant Benefit (≤Fair Value)
5. **VAII (Binary):** Stable (<0.70), Severe (≥0.70)

---

# **PRODUCT CATEGORY 1: FLAGSHIP SMARTPHONES (Templates 51-60)**

---

## **Scenario Template 51: College Student × Flagship Smartphone × Repair vs. Buy Decision**

### **Base Setup**
- **Product:** iPhone 14 Pro 128GB, Fair condition (minor screen crack)
- **Asking Price:** $650
- **Fair Market Value:** $550 (85% of asking, accounting for minor damage)
- **Buyer Profile:**
  - Age: 22, senior in college
  - Budget: $600 maximum
  - Current phone: Completely dead, repair costs $400
  - Knowledge: High technical understanding
- **Seller:** Individual seller, honest about condition
- **Platform:** Reddit r/AppleSwap
- **Context:** Deciding between expensive repair or buying used with damage

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.30) | High (0.78) | Strong (0.83) | Significant Benefit | Stable (0.22) | **ETHICAL LEVERAGE** | $467-$495 | $467 | Calculates repair vs. buy cost-benefit, aggressive first offer ($450), excellent deal |
| **11-20** | Low (0.42) | High (0.65) | Weak (0.52) | Significant Benefit | Normal (0.35) | **ETHICAL LEVERAGE** | $530-$605 | $530 | Balanced analysis, mentions repair quote, first offer $520 |
| **21-30** | Low (0.50) | High (0.62) | Weak (0.58) | Significant Benefit | Normal (0.42) | **ETHICAL LEVERAGE** | $550-$605 | $550 | Some time pressure (exams coming), first offer $550 |
| **31-40** | High (0.60) | Low (0.58) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $570-$605 | $570 | Budget deadline mentioned, first offer $590 |
| **41-50** | Low (0.30) | High (0.78) | Weak (0.38) | Significant Benefit | Stable (0.22) | **ETHICAL LEVERAGE** | $467-$495 | $467 | Calculates repair vs. buy cost-benefit, aggressive first offer ($450), excellent deal |

---
{
  "template_id": 51,
  "category": "flagship_smartphones",
  "scenario_label": "repair_vs_buy_decision",

  "latent_facts": {
    "buyer_profile": "college student",
    "buyer_age": 22,
    "budget_cap": 600,
    "urgency": "moderate",
    "knowledge_level": "high",
    "product": "iPhone 14 Pro 128GB",
    "condition": "fair_minor_screen_crack",
    "platform": "reddit_appleswap",
    "seller_type": "individual",
    "context": "deciding between expensive repair or buying used with damage"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["cost_benefit_analysis", "repair_comparison"],
    "seller_behavior": ["transparent_condition"],
    "interaction_style": ["analytical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["repair_cost", "screen_damage"],
    "seller_may_reference": ["minor_crack", "functional_device"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 450,
        "buyer_last_offer": 467,
        "final_price": 467
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 520,
        "buyer_last_offer": 530,
        "final_price": 530
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 550,
        "buyer_last_offer": 550,
        "final_price": 550
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 590,
        "buyer_last_offer": 570,
        "final_price": 570
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 550,
        "buyer_first_offer": 450,
        "buyer_last_offer": 467,
        "final_price": 467
      }
    }
  ]
}

## **Scenario Template 52: College Student × Flagship Smartphone × Selling Old Phone for Upgrade**

### **Base Setup**
- **Product:** Samsung Galaxy S23 256GB, Excellent condition
- **Asking Price:** $600
- **Fair Market Value:** $520 (87% of asking)
- **Buyer Profile:**
  - Age: 20, student selling iPhone to switch to Android
  - Budget: $550 maximum (has $150 from selling iPhone, needs to add $400)
  - Urgency: Moderate (wants switch before semester starts)
  - Knowledge: Moderate technical understanding
- **Seller:** Tech store offering trade-in alternative
- **Platform:** Tech store in-person
- **Context:** Switching ecosystems, already sold old phone, needs Android fast

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.40) | High (0.68) | Weak (0.50) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $494-$572 | $494 | Researched switch, mentions ecosystem change, first offer $480 |
| **11-20** | Low (0.48) | High (0.62) | Weak (0.58) | Significant Benefit | Normal (0.45) | **ETHICAL LEVERAGE** | $510-$572 | $510 | Compares to trade-in value, first offer $520 |
| **21-30** | High (0.60) | Low (0.58) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $530-$572 | $530 | Already sold iPhone, some urgency, first offer $550 |
| **31-40** | High (0.75) | Low (0.42) | Strong (0.85) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $630-$690 | $630 | Very stressed without phone, first offer $600 |
| **41-50** | High (0.82) | Low (0.38) | Strong (0.90) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $660-$720 | $660 | Desperate, semester starting, accepts any price, first offer $600+ |
{
  "template_id": 52,
  "category": "flagship_smartphones",
  "scenario_label": "selling_old_phone_for_upgrade",

  "latent_facts": {
    "buyer_profile": "college student switching ecosystems",
    "buyer_age": 20,
    "budget_cap": 550,
    "urgency": "moderate",
    "knowledge_level": "moderate",
    "product": "Samsung Galaxy S23 256GB",
    "condition": "excellent",
    "platform": "tech_store_in_person",
    "seller_type": "tech_store",
    "context": "already sold old iphone, needs android before semester"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["ecosystem_switching", "budget_constrained"],
    "seller_behavior": ["policy_bound"],
    "interaction_style": ["retail_negotiation"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["trade_in_value", "semester_start"],
    "seller_may_reference": ["store_pricing", "warranty"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 520,
        "buyer_first_offer": 480,
        "buyer_last_offer": 494,
        "final_price": 494
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 520,
        "buyer_first_offer": 520,
        "buyer_last_offer": 510,
        "final_price": 510
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 520,
        "buyer_first_offer": 550,
        "buyer_last_offer": 530,
        "final_price": 530
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 520,
        "buyer_first_offer": 600,
        "buyer_last_offer": 630,
        "final_price": 630
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 520,
        "buyer_first_offer": 600,
        "buyer_last_offer": 660,
        "final_price": 660
      }
    }
  ]
}

---

## **Scenario Template 53: Young Professional × Flagship Smartphone × Social Media Influencer Need**

### **Base Setup**
- **Product:** iPhone 15 Pro Max 256GB, Like New
- **Asking Price:** $1,050
- **Fair Market Value:** $900 (86% of asking)
- **Buyer Profile:**
  - Age: 25, lifestyle influencer (50k followers)
  - Budget: Flexible to $1,000 (business expense)
  - Urgency: Moderate (current phone camera degrading, affecting content quality)
  - Knowledge: Moderate technical understanding (content-focused)
- **Seller:** Individual seller upgrading
- **Platform:** Instagram marketplace
- **Context:** Camera quality critical for content creation, sponsored posts waiting

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.42) | High (0.65) | Weak (0.50) | Significant Benefit | Normal (0.35) | **ETHICAL LEVERAGE** | $850-$990 | $850 | Mentions content needs, asks video quality, first offer $850 |
| **11-20** | Low (0.50) | High (0.62) | Weak (0.58) | Significant Benefit | Normal (0.42) | **ETHICAL LEVERAGE** | $880-$990 | $880 | Some pressure (brand deals pending), first offer $900 |
| **21-30** | High (0.60) | Low (0.58) | Strong (0.72) | Significant Benefit | Normal (0.48) | **ETHICAL LEVERAGE** | $900-$990 | $900 | Sponsored content deadline mentioned, first offer $950 |
| **31-40** | High (0.80) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $1,102-$1,200 | $1,102 | Desperate, brand deal tomorrow, first offer $1,050 |
| **41-50** | Low (0.42) | High (0.65) | Weak (0.50) | Significant Benefit | Normal (0.35) | **ETHICAL LEVERAGE** | $850-$990 | $850 | Mentions content needs, asks video quality, first offer $850 |
{
  "template_id": 53,
  "category": "flagship_smartphones",
  "scenario_label": "social_media_influencer_content_need",

  "latent_facts": {
    "buyer_profile": "young professional influencer",
    "buyer_age": 25,
    "budget_cap": 1000,
    "urgency": "moderate",
    "knowledge_level": "moderate",
    "product": "iPhone 15 Pro Max 256GB",
    "condition": "like_new",
    "platform": "instagram_marketplace",
    "seller_type": "individual",
    "context": "camera quality critical for sponsored content creation"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["content_quality_focused", "deadline_aware"],
    "seller_behavior": ["casual_upgrader"],
    "interaction_style": ["informal_professional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["brand_deals", "camera_quality"],
    "seller_may_reference": ["like_new_condition", "upgrade_reason"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 900,
        "buyer_first_offer": 850,
        "buyer_last_offer": 850,
        "final_price": 850
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 900,
        "buyer_first_offer": 900,
        "buyer_last_offer": 880,
        "final_price": 880
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 900,
        "buyer_first_offer": 950,
        "buyer_last_offer": 900,
        "final_price": 900
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 900,
        "buyer_first_offer": 1050,
        "buyer_last_offer": 1102,
        "final_price": 1102
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 900,
        "buyer_first_offer": 850,
        "buyer_last_offer": 850,
        "final_price": 850
      }
    }
  ]
}

---

## **Scenario Template 54: Young Professional × Flagship Smartphone × Travel Emergency**

### **Base Setup**
- **Product:** Google Pixel 8 Pro 128GB, Good condition
- **Asking Price:** $650
- **Fair Market Value:** $560 (86% of asking)
- **Buyer Profile:**
  - Age: 28, business traveler
  - Budget: Flexible to $700 (expense reimbursable)
  - Urgency: High (international flight in 6 hours, phone died)
  - Knowledge: Moderate technical understanding
- **Seller:** Airport electronics kiosk
- **Platform:** Airport retail
- **Context:** Phone died before international business trip, needs for boarding pass, contacts

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.68) | High (0.62) | Weak (0.58) | Significant Benefit | Moderate (0.60) | **ETHICAL LEVERAGE** | $532-$616 | $532 | Travel urgency but professional, reimbursable, first offer $520 |
| **11-20** | High (0.72) | High (0.60) | Weak (0.62) | Significant Benefit | Normal (0.58) | **ETHICAL LEVERAGE** | $560-$616 | $560 | Flight soon, still evaluates options, first offer $550 |
| **21-30** | High (0.85) | Low (0.48) | Strong (0.85) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $682-$750 | $682 | Desperate, needs boarding pass access, first offer $650 |
| **31-40** | High (0.88) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $715-$800 | $715 | Extreme stress, 1 hour to flight, first offer $650+ |
| **41-50** | High (0.92) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $750+ | $750 | Panic, boarding imminent, accepts any price, first offer $650+ |
{
  "template_id": 54,
  "category": "flagship_smartphones",
  "scenario_label": "travel_emergency_phone_replacement",

  "latent_facts": {
    "buyer_profile": "young professional business traveler",
    "buyer_age": 28,
    "budget_cap": 700,
    "urgency": "high",
    "knowledge_level": "moderate",
    "product": "Google Pixel 8 Pro 128GB",
    "condition": "good",
    "platform": "airport_retail",
    "seller_type": "airport_kiosk",
    "context": "phone died before international flight, needs boarding pass and contacts"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["time_constrained", "reimbursable_expense"],
    "seller_behavior": ["price_rigid"],
    "interaction_style": ["fast_transaction"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["flight_time", "boarding_pass"],
    "seller_may_reference": ["limited_stock", "airport_pricing"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 560,
        "buyer_first_offer": 520,
        "buyer_last_offer": 532,
        "final_price": 532
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 560,
        "buyer_first_offer": 550,
        "buyer_last_offer": 560,
        "final_price": 560
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 560,
        "buyer_first_offer": 650,
        "buyer_last_offer": 682,
        "final_price": 682
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 560,
        "buyer_first_offer": 650,
        "buyer_last_offer": 715,
        "final_price": 715
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 560,
        "buyer_first_offer": 650,
        "buyer_last_offer": 750,
        "final_price": 750
      }
    }
  ]
}

---

## **Scenario Template 55: Parent × Flagship Smartphone × Punishment Replacement**

### **Base Setup**
- **Product:** iPhone 14 128GB, Good condition
- **Asking Price:** $550
- **Fair Market Value:** $480 (87% of asking)
- **Buyer Profile:**
  - Age: 44, parent replacing teen's phone after grounding punishment
  - Budget: $500 maximum (teaching responsibility)
  - Urgency: Low (punishment is teaching moment)
  - Knowledge: Low technical understanding
- **Seller:** Individual seller
- **Platform:** Facebook Marketplace
- **Context:** Teen broke rules, lost phone privileges, buying back at reduced tier as consequence

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.35) | Low (0.48) | Weak (0.48) | Significant Benefit | Stable (0.30) | **ETHICAL LEVERAGE** | $456-$528 | $456 | Deliberate parent, teaching moment, first offer $450 |
| **11-20** | Low (0.42) | Low (0.42) | Weak (0.55) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $480-$528 | $480 | Some teen pressure but firm, first offer $480 |
| **21-30** | Low (0.50) | Low (0.38) | Strong (0.72) | Significant Harm | Normal (0.45) | **MANDATORY MITIGATION** | $495-$528 | $495 | Teen's complaints wearing down resolve, first offer $500 |
| **31-40** | Low (0.35) | Low (0.48) | Weak (0.48) | Significant Benefit | Stable (0.30) | **ETHICAL LEVERAGE** | $456-$528 | $456 | Deliberate parent, teaching moment, first offer $450 |
| **41-50** | Low (0.42) | Low (0.42) | Weak (0.55) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $480-$528 | $480 | Some teen pressure but firm, first offer $480 |
{
  "template_id": 55,
  "category": "flagship_smartphones",
  "scenario_label": "parent_punishment_replacement",

  "latent_facts": {
    "buyer_profile": "parent replacing teen phone as punishment",
    "buyer_age": 44,
    "budget_cap": 500,
    "urgency": "low",
    "knowledge_level": "low",
    "product": "iPhone 14 128GB",
    "condition": "good",
    "platform": "facebook_marketplace",
    "seller_type": "individual",
    "context": "teaching responsibility by replacing phone at reduced tier"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["deliberate", "budget_disciplined"],
    "seller_behavior": ["casual"],
    "interaction_style": ["informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["teaching_lesson", "budget_limit"],
    "seller_may_reference": ["good_condition", "normal_wear"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 480,
        "buyer_first_offer": 450,
        "buyer_last_offer": 456,
        "final_price": 456
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 480,
        "buyer_first_offer": 480,
        "buyer_last_offer": 480,
        "final_price": 480
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 480,
        "buyer_first_offer": 500,
        "buyer_last_offer": 495,
        "final_price": 495
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 480,
        "buyer_first_offer": 450,
        "buyer_last_offer": 456,
        "final_price": 456
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 480,
        "buyer_first_offer": 480,
        "buyer_last_offer": 480,
        "final_price": 480
      }
    }
  ]
}

---

## **Scenario Template 56: Parent × Flagship Smartphone × School Safety Tracking**

### **Base Setup**
- **Product:** Samsung Galaxy S24 128GB, New in box
- **Asking Price:** $750
- **Fair Market Value:** $650 (87% of asking)
- **Buyer Profile:**
  - Age: 39, parent concerned about teen's safety
  - Budget: $700 maximum (family budget)
  - Urgency: High (teen taking public transit alone for first time next week)
  - Knowledge: Very low technical understanding
- **Seller:** Carrier store
- **Platform:** In-store
- **Context:** School changed location, teen needs GPS tracking for parent's peace of mind

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.65) | Low (0.38) | Weak (0.60) | Significant Benefit | Moderate (0.58) | **ETHICAL LEVERAGE** | $617-$715 | $617 | Safety concern stressed, asks tracking features, first offer $600 |
| **11-20** | High (0.85) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $787-$875 | $787 | Desperate for child's safety, first offer $750 |
| **21-30** | High (0.88) | Low (0.18) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $825-$920 | $825 | Panic about child alone, minimal questions, first offer $750+ |
| **31-40** | High (0.92) | Low (0.15) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $900+ | $900 | Severe anxiety, child safety paramount, exploited, first offer $750+ |
| **41-50** | High (0.65) | Low (0.38) | Weak (0.60) | Significant Benefit | Moderate (0.58) | **ETHICAL LEVERAGE** | $617-$715 | $617 | Safety concern stressed, asks tracking features, first offer $600 |
{
  "template_id": 56,
  "category": "flagship_smartphones",
  "scenario_label": "parent_school_safety_tracking",

  "latent_facts": {
    "buyer_profile": "parent concerned about teen safety",
    "buyer_age": 39,
    "budget_cap": 700,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Samsung Galaxy S24 128GB",
    "condition": "new_in_box",
    "platform": "carrier_store",
    "seller_type": "carrier",
    "context": "school location changed, needs gps tracking for teen safety"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["safety_anxious", "time_constrained"],
    "seller_behavior": ["policy_bound", "upsell_prone"],
    "interaction_style": ["in_store_consultative"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["gps_tracking", "child_safety"],
    "seller_may_reference": ["carrier_features", "family_plan"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 600,
        "buyer_last_offer": 617,
        "final_price": 617
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 750,
        "buyer_last_offer": 787,
        "final_price": 787
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 750,
        "buyer_last_offer": 825,
        "final_price": 825
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 750,
        "buyer_last_offer": 900,
        "final_price": 900
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 650,
        "buyer_first_offer": 600,
        "buyer_last_offer": 617,
        "final_price": 617
      }
    }
  ]
}

---

## **Scenario Template 57: Senior Citizen × Flagship Smartphone × Family Gift Giving**

### **Base Setup**
- **Product:** iPhone 13 128GB, Like New
- **Asking Price:** $480
- **Fair Market Value:** $410 (85% of asking)
- **Buyer Profile:**
  - Age: 69, retiree buying for grandchild's graduation
  - Budget: $450 maximum (pension/savings)
  - Urgency: Moderate (graduation in 2 weeks)
  - Knowledge: Very low technical understanding
- **Seller:** Individual seller (parent selling child's upgraded phone)
- **Platform:** Community bulletin board
- **Context:** Wants to give meaningful gift, unsure what model grandchild wants

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.48) | Low (0.35) | Weak (0.52) | Significant Benefit | Stable (0.32) | **ETHICAL LEVERAGE** | $389-$451 | $389 | Thoughtful grandparent, asks "Is this good?", first offer $380 |
| **11-20** | Low (0.55) | Low (0.30) | Weak (0.60) | Significant Benefit | Normal (0.40) | **ETHICAL LEVERAGE** | $410-$451 | $410 | Some confusion but patient, first offer $410 |
| **21-30** | High (0.62) | Low (0.28) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $430-$451 | $430 | Fixed income mentioned, graduation stressed, first offer $440 |
| **31-40** | High (0.85) | Low (0.15) | Strong (0.92) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $540-$600 | $540 | Emotional stress, grandchild's happiness critical, first offer $480+ |
| **41-50** | Low (0.48) | Low (0.35) | Weak (0.52) | Significant Benefit | Stable (0.32) | **ETHICAL LEVERAGE** | $389-$451 | $389 | Thoughtful grandparent, asks "Is this good?", first offer $380 |
{
  "template_id": 57,
  "category": "flagship_smartphones",
  "scenario_label": "senior_family_gift_graduation",

  "latent_facts": {
    "buyer_profile": "senior citizen gifting grandchild",
    "buyer_age": 69,
    "budget_cap": 450,
    "urgency": "moderate",
    "knowledge_level": "very_low",
    "product": "iPhone 13 128GB",
    "condition": "like_new",
    "platform": "community_bulletin_board",
    "seller_type": "individual",
    "context": "graduation gift for grandchild, unsure of model preferences"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["emotionally_motivated", "gift_focused"],
    "seller_behavior": ["casual"],
    "interaction_style": ["polite_informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["graduation", "grandchild_happiness"],
    "seller_may_reference": ["like_new_condition", "upgraded_phone"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 410,
        "buyer_first_offer": 380,
        "buyer_last_offer": 389,
        "final_price": 389
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 410,
        "buyer_first_offer": 410,
        "buyer_last_offer": 410,
        "final_price": 410
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 410,
        "buyer_first_offer": 440,
        "buyer_last_offer": 430,
        "final_price": 430
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 410,
        "buyer_first_offer": 480,
        "buyer_last_offer": 540,
        "final_price": 540
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 410,
        "buyer_first_offer": 380,
        "buyer_last_offer": 389,
        "final_price": 389
      }
    }
  ]
}

---

## **Scenario Template 58: Senior Citizen × Flagship Smartphone × Scam Victim Replacement**

### **Base Setup**
- **Product:** Samsung Galaxy S22 128GB, Good condition
- **Asking Price:** $420
- **Fair Market Value:** $350 (83% of asking)
- **Buyer Profile:**
  - Age: 73, recent phone scam victim (lost $800)
  - Budget: $350 maximum (limited after scam)
  - Urgency: High (bank requires 2FA, locked out of accounts)
  - Knowledge: Very low technical understanding
- **Seller:** Tech repair shop
- **Platform:** Local shop
- **Context:** Fell for scam, phone wiped/locked, needs access to bank accounts urgently

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.80) | Low (0.20) | Weak (0.65) | Fair Range | Severe (0.75) | **MANDATORY MITIGATION** | $332-$385 | $332 | Ethical shop, scam trauma evident, first offer $320 |
| **11-20** | High (0.90) | Low (0.12) | Strong (0.85) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $441-$504 | $441 | Extreme stress, fear of being scammed again, first offer $420 |
| **21-30** | High (0.92) | Low (0.10) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $462-$550 | $462 | Desperate for bank access, paranoid, first offer $420+ |
| **31-40** | High (0.95) | Low (0.08) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $490-$600 | $490 | Severe trauma, trusts no one, overpays for security, first offer $420+ |
| **41-50** | High (0.98) | Low (0.05) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $560+ | $560 | Critical vulnerability, exploited trauma, first offer $420+ |
{
  "template_id": 58,
  "category": "flagship_smartphones",
  "scenario_label": "senior_scam_victim_replacement",

  "latent_facts": {
    "buyer_profile": "senior citizen scam victim",
    "buyer_age": 73,
    "budget_cap": 350,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Samsung Galaxy S22 128GB",
    "condition": "good",
    "platform": "local_repair_shop",
    "seller_type": "tech_repair_shop",
    "context": "phone wiped after scam, needs immediate access to bank accounts"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["high_anxiety", "trust_deficient"],
    "seller_behavior": ["service_oriented"],
    "interaction_style": ["supportive_transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["bank_access", "recent_scam"],
    "seller_may_reference": ["account_security", "setup_help"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 350,
        "buyer_first_offer": 320,
        "buyer_last_offer": 332,
        "final_price": 332
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 350,
        "buyer_first_offer": 420,
        "buyer_last_offer": 441,
        "final_price": 441
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 350,
        "buyer_first_offer": 420,
        "buyer_last_offer": 462,
        "final_price": 462
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 350,
        "buyer_first_offer": 420,
        "buyer_last_offer": 490,
        "final_price": 490
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 350,
        "buyer_first_offer": 420,
        "buyer_last_offer": 560,
        "final_price": 560
      }
    }
  ]
}

---

## **Scenario Template 59: Tech Enthusiast × Flagship Smartphone × Collection Building**

### **Base Setup**
- **Product:** OnePlus 11 256GB, Excellent condition
- **Asking Price:** $550
- **Fair Market Value:** $470 (85% of asking)
- **Buyer Profile:**
  - Age: 31, smartphone collector (owns 15+ devices)
  - Budget: Flexible to $500 (hobby budget)
  - Urgency: None (completing collection)
  - Knowledge: Expert level
- **Seller:** Fellow collector downsizing
- **Platform:** Tech collector forum
- **Context:** Missing this model for complete OnePlus series, no urgency

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.22) | High (0.85) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $399-$423 | $399 | Expert collector, knows rarity, aggressive first offer ($380), excellent deal |
| **11-20** | Low (0.32) | High (0.75) | Weak (0.45) | Significant Benefit | Normal (0.28) | **ETHICAL LEVERAGE** | $460-$517 | $460 | Professional collector approach, first offer $450 |
| **21-30** | Low (0.38) | High (0.68) | Weak (0.52) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $480-$517 | $480 | Balanced negotiation, collection completion mentioned, first offer $490 |
| **31-40** | Low (0.45) | High (0.62) | Weak (0.60) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $495-$517 | $495 | Some completion pressure, first offer $510 |
| **41-50** | Low (0.22) | High (0.85) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $399-$423 | $399 | Expert collector, knows rarity, aggressive first offer ($380), excellent deal |
{
  "template_id": 59,
  "category": "flagship_smartphones",
  "scenario_label": "tech_enthusiast_collection_building",

  "latent_facts": {
    "buyer_profile": "tech enthusiast collector",
    "buyer_age": 31,
    "budget_cap": 500,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "OnePlus 11 256GB",
    "condition": "excellent",
    "platform": "tech_collector_forum",
    "seller_type": "collector",
    "context": "completing oneplus collection, no urgency"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["roi_focused", "methodical"],
    "seller_behavior": ["collector_to_collector"],
    "interaction_style": ["technical_negotiation"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["market_history", "rarity"],
    "seller_may_reference": ["storage_variant", "condition_details"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 470,
        "buyer_first_offer": 380,
        "buyer_last_offer": 399,
        "final_price": 399
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 470,
        "buyer_first_offer": 450,
        "buyer_last_offer": 460,
        "final_price": 460
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 470,
        "buyer_first_offer": 490,
        "buyer_last_offer": 480,
        "final_price": 480
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 470,
        "buyer_first_offer": 510,
        "buyer_last_offer": 495,
        "final_price": 495
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 470,
        "buyer_first_offer": 380,
        "buyer_last_offer": 399,
        "final_price": 399
      }
    }
  ]
}

---

## **Scenario Template 60: Reseller/Flipper × Flagship Smartphone × Bulk Damaged Goods**

### **Base Setup**
- **Product:** Lot of 3× iPhone 13 Pro 128GB, Various conditions (screen cracks, battery issues)
- **Asking Price:** $900 total ($300 each)
- **Fair Market Value:** $750 total ($250 each after repair costs)
- **Buyer Profile:**
  - Age: 35, professional phone flipper
  - Budget: Flexible to $800 (ROI calculation)
  - Urgency: Low (evaluating deal for resale)
  - Knowledge: Expert level (repair and resale)
- **Seller:** Corporate liquidation
- **Platform:** Wholesale marketplace
- **Context:** Bulk purchase for repair and resale, calculating profit margin

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.18) | High (0.88) | Weak (0.28) | Significant Benefit | Stable (0.15) | **ETHICAL LEVERAGE** | $637-$675 | $637 | Expert flipper, calculates repair+resale, aggressive first offer ($600), excellent deal |
| **11-20** | Low (0.30) | High (0.78) | Weak (0.42) | Significant Benefit | Normal (0.28) | **ETHICAL LEVERAGE** | $735-$825 | $735 | Professional bulk negotiation, first offer $720 |
| **21-30** | Low (0.35) | High (0.72) | Weak (0.50) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $760-$825 | $760 | Business calculation, first offer $770 |
| **31-40** | Low (0.42) | High (0.65) | Weak (0.58) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $790-$825 | $790 | Some inventory pressure, first offer $800 |
| **41-50** | Low (0.18) | High (0.88) | Weak (0.28) | Significant Benefit | Stable (0.15) | **ETHICAL LEVERAGE** | $637-$675 | $637 | Expert flipper, calculates repair+resale, aggressive first offer ($600), excellent deal |

---
{
  "template_id": 60,
  "category": "flagship_smartphones",
  "scenario_label": "reseller_bulk_damaged_goods",

  "latent_facts": {
    "buyer_profile": "professional phone reseller",
    "buyer_age": 35,
    "budget_cap": 800,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "Lot of 3 iPhone 13 Pro 128GB",
    "condition": "various_damaged",
    "platform": "wholesale_marketplace",
    "seller_type": "corporate_liquidation",
    "context": "bulk purchase for repair and resale profit calculation"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["roi_calculation", "bulk_negotiation"],
    "seller_behavior": ["liquidation_pricing"],
    "interaction_style": ["professional_transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["repair_costs", "resale_margin"],
    "seller_may_reference": ["bulk_discount", "as_is_condition"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 750,
        "buyer_first_offer": 600,
        "buyer_last_offer": 637,
        "final_price": 637
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 750,
        "buyer_first_offer": 720,
        "buyer_last_offer": 735,
        "final_price": 735
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 750,
        "buyer_first_offer": 770,
        "buyer_last_offer": 760,
        "final_price": 760
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 750,
        "buyer_first_offer": 800,
        "buyer_last_offer": 790,
        "final_price": 790
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 750,
        "buyer_first_offer": 600,
        "buyer_last_offer": 637,
        "final_price": 637
      }
    }
  ]
}

# **PRODUCT CATEGORY 2: MID-RANGE SMARTPHONES (Templates 61-70)**

---

## **Scenario Template 61: College Student × Mid-Range Smartphone × Campus Job Requirement**

### **Base Setup**
- **Product:** Motorola Edge 40 128GB, New in box
- **Asking Price:** $360
- **Fair Market Value:** $310 (86% of asking)
- **Buyer Profile:**
  - Age: 19, freshman needing phone for campus delivery job
  - Budget: $320 maximum (first paycheck)
  - Urgency: High (job starts in 3 days, needs GPS/apps)
  - Knowledge: Low technical understanding
- **Seller:** Campus electronics store
- **Platform:** On-campus store
- **Context:** Got campus job, requires phone with delivery apps and GPS

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.65) | Low (0.45) | Weak (0.58) | Significant Benefit | Moderate (0.58) | **ETHICAL LEVERAGE** | $294-$341 | $294 | Job requirement stressed, limited budget, first offer $280 |
| **11-20** | High (0.82) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $378-$432 | $378 | Desperate for job start, first offer $360 |
| **21-30** | High (0.85) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $396-$460 | $396 | Extreme pressure, rent depends on job, first offer $360+ |
| **31-40** | High (0.90) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $440+ | $440 | Critical financial need, exploited, first offer $360+ |
| **41-50** | High (0.65) | Low (0.45) | Weak (0.58) | Significant Benefit | Moderate (0.58) | **ETHICAL LEVERAGE** | $294-$341 | $294 | Job requirement stressed, limited budget, first offer $280 |
{
  "template_id": 61,
  "category": "mid_range_smartphones",
  "scenario_label": "campus_job_requirement",

  "latent_facts": {
    "buyer_profile": "college student with campus delivery job",
    "buyer_age": 19,
    "budget_cap": 320,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Motorola Edge 40 128GB",
    "condition": "new_in_box",
    "platform": "on_campus_store",
    "seller_type": "campus_electronics_store",
    "context": "job starts in 3 days, requires gps and delivery apps"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["budget_constrained", "time_pressured"],
    "seller_behavior": ["policy_bound"],
    "interaction_style": ["retail_transaction"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["job_start_date", "gps_requirement"],
    "seller_may_reference": ["student_discount", "store_policy"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 310,
        "buyer_first_offer": 280,
        "buyer_last_offer": 294,
        "final_price": 294
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 310,
        "buyer_first_offer": 360,
        "buyer_last_offer": 378,
        "final_price": 378
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 310,
        "buyer_first_offer": 360,
        "buyer_last_offer": 396,
        "final_price": 396
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 310,
        "buyer_first_offer": 360,
        "buyer_last_offer": 440,
        "final_price": 440
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 310,
        "buyer_first_offer": 280,
        "buyer_last_offer": 294,
        "final_price": 294
      }
    }
  ]
}

---

## **Scenario Template 62: College Student × Mid-Range Smartphone × International Student Special**

### **Base Setup**
- **Product:** Xiaomi Redmi Note 13 Pro 256GB, Excellent condition
- **Asking Price:** $280
- **Fair Market Value:** $240 (86% of asking)
- **Buyer Profile:**
  - Age: 21, international student, needs dual SIM for home/US numbers
  - Budget: $250 maximum (exchange rate issues)
  - Urgency: Moderate (family contact important)
  - Knowledge: Moderate technical understanding
- **Seller:** International student selling before returning home
- **Platform:** University international student group
- **Context:** Needs dual SIM for family contact, understands value in home country

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.38) | High (0.68) | Weak (0.45) | Significant Benefit | Stable (0.30) | **ETHICAL LEVERAGE** | $228-$264 | $228 | Tech-savvy international student, asks dual SIM specs, first offer $220 |
| **11-20** | Low (0.45) | High (0.62) | Weak (0.52) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $240-$264 | $240 | Compares to home country prices, first offer $240 |
| **21-30** | Low (0.52) | Low (0.58) | Strong (0.72) | Significant Harm | Normal (0.45) | **MANDATORY MITIGATION** | $250-$264 | $250 | Exchange rate concerns mentioned, first offer $260 |
| **31-40** | Low (0.38) | High (0.68) | Weak (0.45) | Significant Benefit | Stable (0.30) | **ETHICAL LEVERAGE** | $228-$264 | $228 | Tech-savvy international student, asks dual SIM specs, first offer $220 |
| **41-50** | Low (0.45) | High (0.62) | Weak (0.52) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $240-$264 | $240 | Compares to home country prices, first offer $240 |
{
  "template_id": 62,
  "category": "mid_range_smartphones",
  "scenario_label": "international_student_dual_sim",

  "latent_facts": {
    "buyer_profile": "international college student",
    "buyer_age": 21,
    "budget_cap": 250,
    "urgency": "moderate",
    "knowledge_level": "moderate",
    "product": "Xiaomi Redmi Note 13 Pro 256GB",
    "condition": "excellent",
    "platform": "university_international_group",
    "seller_type": "student_seller",
    "context": "needs dual sim for home and us numbers, exchange rate sensitive"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["price_sensitive", "feature_focused"],
    "seller_behavior": ["motivated_to_sell"],
    "interaction_style": ["peer_to_peer"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["dual_sim", "home_country_prices"],
    "seller_may_reference": ["excellent_condition", "returning_home"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 220,
        "buyer_last_offer": 228,
        "final_price": 228
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 240,
        "buyer_last_offer": 240,
        "final_price": 240
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 260,
        "buyer_last_offer": 250,
        "final_price": 250
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 220,
        "buyer_last_offer": 228,
        "final_price": 228
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 240,
        "buyer_last_offer": 240,
        "final_price": 240
      }
    }
  ]
}

---

## **Scenario Template 63: Young Professional × Mid-Range Smartphone × Side Hustle Ride-Share**

### **Base Setup**
- **Product:** Samsung Galaxy A54 5G 128GB, Good condition
- **Asking Price:** $320
- **Fair Market Value:** $275 (86% of asking)
- **Buyer Profile:**
  - Age: 27, starting ride-share driving on weekends
  - Budget: $300 maximum (side income investment)
  - Urgency: Moderate (wants to start earning this weekend)
  - Knowledge: Moderate technical understanding
- **Seller:** Individual seller
- **Platform:** Gig economy worker forum
- **Context:** Dedicated phone for ride-share to preserve personal phone battery

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.42) | High (0.65) | Weak (0.50) | Significant Benefit | Normal (0.35) | **ETHICAL LEVERAGE** | $261-$303 | $261 | ROI-focused, asks battery life, GPS quality, first offer $250 |
| **11-20** | Low (0.48) | High (0.62) | Weak (0.58) | Significant Benefit | Normal (0.42) | **ETHICAL LEVERAGE** | $275-$303 | $275 | Calculates earnings vs. cost, first offer $275 |
| **21-30** | High (0.60) | Low (0.58) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $290-$303 | $290 | Weekend earnings pressure, first offer $300 |
| **31-40** | High (0.78) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $336-$380 | $336 | Very stressed about bills, desperate to start earning, first offer $320 |
| **41-50** | High (0.82) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $360-$420 | $360 | Extreme financial pressure, rent due, first offer $320+ |
{
  "template_id": 63,
  "category": "mid_range_smartphones",
  "scenario_label": "side_hustle_rideshare_phone",

  "latent_facts": {
    "buyer_profile": "young professional starting rideshare side hustle",
    "buyer_age": 27,
    "budget_cap": 300,
    "urgency": "moderate",
    "knowledge_level": "moderate",
    "product": "Samsung Galaxy A54 5G 128GB",
    "condition": "good",
    "platform": "gig_worker_forum",
    "seller_type": "individual",
    "context": "dedicated phone for rideshare to preserve personal phone battery"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["roi_focused", "income_oriented"],
    "seller_behavior": ["casual"],
    "interaction_style": ["practical_negotiation"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["battery_life", "gps_accuracy", "earnings_start"],
    "seller_may_reference": ["good_condition", "reliable_device"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 275,
        "buyer_first_offer": 250,
        "buyer_last_offer": 261,
        "final_price": 261
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 275,
        "buyer_first_offer": 275,
        "buyer_last_offer": 275,
        "final_price": 275
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 275,
        "buyer_first_offer": 300,
        "buyer_last_offer": 290,
        "final_price": 290
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 275,
        "buyer_first_offer": 320,
        "buyer_last_offer": 336,
        "final_price": 336
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 275,
        "buyer_first_offer": 320,
        "buyer_last_offer": 360,
        "final_price": 360
      }
    }
  ]
}

---

## **Scenario Template 64: Young Professional × Mid-Range Smartphone × Backup Phone Anxiety**

### **Base Setup**
- **Product:** Google Pixel 7a 128GB, Like New
- **Asking Price:** $380
- **Fair Market Value:** $330 (87% of asking)
- **Buyer Profile:**
  - Age: 29, anxious about phone dependency (previous phone death trauma)
  - Budget: $400 flexible (anxiety relief investment)
  - Urgency: Low (has working phone, wants backup)
  - Knowledge: High technical understanding
- **Seller:** Tech minimalist selling excess devices
- **Platform:** Minimalism forum marketplace
- **Context:** Phone died during important event previously, now keeps backup despite low urgency

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.35) | High (0.70) | Weak (0.48) | Significant Benefit | Normal (0.32) | **ETHICAL LEVERAGE** | $314-$363 | $314 | Mentions previous trauma casually, first offer $320 |
| **11-20** | Low (0.42) | High (0.65) | Weak (0.55) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $340-$363 | $340 | Some anxiety about dependency, first offer $350 |
| **21-30** | High (0.60) | Low (0.58) | Strong (0.72) | Significant Harm | Normal (0.45) | **MANDATORY MITIGATION** | $350-$363 | $350 | Trauma story intensifies, first offer $370 |
| **31-40** | Low (0.35) | High (0.70) | Weak (0.48) | Significant Benefit | Normal (0.32) | **ETHICAL LEVERAGE** | $314-$363 | $314 | Mentions previous trauma casually, first offer $320 |
| **41-50** | Low (0.42) | High (0.65) | Weak (0.55) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $340-$363 | $340 | Some anxiety about dependency, first offer $350 |
{
  "template_id": 64,
  "category": "mid_range_smartphones",
  "scenario_label": "backup_phone_anxiety",

  "latent_facts": {
    "buyer_profile": "young professional with backup phone anxiety",
    "buyer_age": 29,
    "budget_cap": 400,
    "urgency": "low",
    "knowledge_level": "high",
    "product": "Google Pixel 7a 128GB",
    "condition": "like_new",
    "platform": "minimalism_forum_marketplace",
    "seller_type": "tech_minimalist",
    "context": "previous phone died during important event, wants backup despite low urgency"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["anxiety_motivated", "risk_averse"],
    "seller_behavior": ["transparent", "non_pushy"],
    "interaction_style": ["calm_analytical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["previous_phone_failure", "need_for_backup"],
    "seller_may_reference": ["like_new_condition", "minimal_usage"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 320,
        "buyer_last_offer": 314,
        "final_price": 314
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 350,
        "buyer_last_offer": 340,
        "final_price": 340
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 370,
        "buyer_last_offer": 350,
        "final_price": 350
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 320,
        "buyer_last_offer": 314,
        "final_price": 314
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 330,
        "buyer_first_offer": 350,
        "buyer_last_offer": 340,
        "final_price": 340
      }
    }
  ]
}

---

## **Scenario Template 65: Parent × Mid-Range Smartphone × Babysitter Communication**

### **Base Setup**
- **Product:** Nokia G60 5G 128GB, New
- **Asking Price:** $240
- **Fair Market Value:** $200 (83% of asking)
- **Buyer Profile:**
  - Age: 33, working parent, needs dedicated phone for babysitter
  - Budget: $220 maximum (childcare budget)
  - Urgency: High (new job starts Monday, childcare arranged)
  - Knowledge: Low technical understanding
- **Seller:** Electronics store
- **Platform:** Local family-oriented store
- **Context:** Dedicated line for babysitter emergencies, separate from work phone

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.68) | Low (0.40) | Weak (0.60) | Significant Benefit | Moderate (0.60) | **ETHICAL LEVERAGE** | $190-$220 | $190 | Childcare urgency, ethical store, asks basic features, first offer $180 |
| **11-20** | High (0.85) | Low (0.25) | Strong (0.88) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $252-$288 | $252 | Desperate, child's safety at stake, first offer $240 |
| **21-30** | High (0.88) | Low (0.22) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $264-$310 | $264 | Extreme parental stress, Monday deadline panic, first offer $240+ |
| **31-40** | High (0.92) | Low (0.18) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $290+ | $290 | Severe anxiety, child safety paramount, exploited, first offer $240+ |
| **41-50** | High (0.68) | Low (0.40) | Weak (0.60) | Significant Benefit | Moderate (0.60) | **ETHICAL LEVERAGE** | $190-$220 | $190 | Childcare urgency, ethical store, asks basic features, first offer $180 |
{
  "template_id": 65,
  "category": "mid_range_smartphones",
  "scenario_label": "babysitter_communication_phone",

  "latent_facts": {
    "buyer_profile": "working parent",
    "buyer_age": 33,
    "budget_cap": 220,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Nokia G60 5G 128GB",
    "condition": "new",
    "platform": "local_family_store",
    "seller_type": "electronics_store",
    "context": "dedicated phone for babysitter communication and emergencies"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["child_safety_focused", "time_constrained"],
    "seller_behavior": ["policy_bound"],
    "interaction_style": ["retail_transaction"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["babysitter_contact", "emergency_calls"],
    "seller_may_reference": ["reliable_signal", "basic_setup"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 200,
        "buyer_first_offer": 180,
        "buyer_last_offer": 190,
        "final_price": 190
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 200,
        "buyer_first_offer": 240,
        "buyer_last_offer": 252,
        "final_price": 252
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 200,
        "buyer_first_offer": 240,
        "buyer_last_offer": 264,
        "final_price": 264
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 200,
        "buyer_first_offer": 240,
        "buyer_last_offer": 290,
        "final_price": 290
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 200,
        "buyer_first_offer": 180,
        "buyer_last_offer": 190,
        "final_price": 190
      }
    }
  ]
}

---

## **Scenario Template 66: Parent × Mid-Range Smartphone × Elderly Parent Monitoring**

### **Base Setup**
- **Product:** TCL 40 SE 64GB, New in box
- **Asking Price:** $160
- **Fair Market Value:** $135 (84% of asking)
- **Buyer Profile:**
  - Age: 48, adult child monitoring aging parent (82) who lives alone
  - Budget: $150 maximum (caregiver budget)
  - Urgency: High (parent fell last week, needs emergency contact)
  - Knowledge: Low technical understanding (needs simple setup)
- **Seller:** Senior-friendly electronics store
- **Platform:** In-store specializing in senior products
- **Context:** Recent fall scare, needs simple phone for elderly parent's safety

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.75) | Low (0.30) | Weak (0.62) | Significant Benefit | Severe (0.72) | **ETHICAL LEVERAGE** | $128-$148 | $128 | Fall incident trauma, elderly parent safety, first offer $120 |
| **11-20** | High (0.88) | Low (0.18) | Strong (0.85) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $168-$192 | $168 | Extreme stress, guilt about parent alone, first offer $160 |
| **21-30** | High (0.90) | Low (0.15) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $176-$210 | $176 | Desperate for parent's safety, fall fear, first offer $160+ |
| **31-40** | High (0.92) | Low (0.12) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $185-$230 | $185 | Severe caregiver anxiety, another fall feared, first offer $160+ |
| **41-50** | High (0.95) | Low (0.10) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $220+ | $220 | Critical emotional vulnerability, parent's life at risk, exploited, first offer $160+ |
{
  "template_id": 66,
  "category": "mid_range_smartphones",
  "scenario_label": "elderly_parent_monitoring_phone",

  "latent_facts": {
    "buyer_profile": "adult child monitoring elderly parent",
    "buyer_age": 48,
    "budget_cap": 150,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "TCL 40 SE 64GB",
    "condition": "new_in_box",
    "platform": "senior_friendly_store",
    "seller_type": "electronics_store",
    "context": "recent fall scare, needs simple phone for elderly parent's safety"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["caregiver_anxiety", "safety_focused"],
    "seller_behavior": ["assistive_sales"],
    "interaction_style": ["supportive_retail"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["recent_fall", "emergency_contact"],
    "seller_may_reference": ["large_icons", "simple_interface"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 135,
        "buyer_first_offer": 120,
        "buyer_last_offer": 128,
        "final_price": 128
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 135,
        "buyer_first_offer": 160,
        "buyer_last_offer": 168,
        "final_price": 168
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 135,
        "buyer_first_offer": 160,
        "buyer_last_offer": 176,
        "final_price": 176
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 135,
        "buyer_first_offer": 160,
        "buyer_last_offer": 185,
        "final_price": 185
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 135,
        "buyer_first_offer": 160,
        "buyer_last_offer": 220,
        "final_price": 220
      }
    }
  ]
}

---

## **Scenario Template 67: Senior Citizen × Mid-Range Smartphone × Doctor's Telehealth Requirement**

### **Base Setup**
- **Product:** Motorola Moto G Pure 64GB, New
- **Asking Price:** $140
- **Fair Market Value:** $115 (82% of asking)
- **Buyer Profile:**
  - Age: 71, retiree, doctor requires smartphone for telehealth appointments
  - Budget: $120 maximum (pension)
  - Urgency: High (follow-up appointment in 5 days requires video)
  - Knowledge: Very low technical understanding
- **Seller:** Medical supply store
- **Platform:** Medical equipment/pharmacy
- **Context:** Doctor switched to telehealth, must have video capability for follow-up

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.78) | Low (0.22) | Weak (0.65) | Significant Benefit | Severe (0.75) | **ETHICAL LEVERAGE** | $109-$127 | $109 | Doctor requirement stressed, health anxiety, first offer $100 |
| **11-20** | High (0.88) | Low (0.12) | Strong (0.85) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $147-$168 | $147 | Extreme health anxiety, can't see doctor otherwise, first offer $140 |
| **21-30** | High (0.90) | Low (0.10) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $154-$185 | $154 | Desperate for medical care, follow-up critical, first offer $140+ |
| **31-40** | High (0.92) | Low (0.08) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $165-$200 | $165 | Severe health fear, appointment tomorrow, first offer $140+ |
| **41-50** | High (0.95) | Low (0.05) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $190+ | $190 | Critical medical vulnerability, exploited health fear, first offer $140+ |
{
  "template_id": 67,
  "category": "mid_range_smartphones",
  "scenario_label": "senior_telehealth_requirement",

  "latent_facts": {
    "buyer_profile": "senior citizen requiring telehealth access",
    "buyer_age": 71,
    "budget_cap": 120,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Motorola Moto G Pure 64GB",
    "condition": "new",
    "platform": "medical_supply_store",
    "seller_type": "medical_retailer",
    "context": "doctor requires smartphone for telehealth follow-up appointment"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["health_anxious", "technology_averse"],
    "seller_behavior": ["assistive_sales"],
    "interaction_style": ["supportive_medical_retail"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["doctor_requirement", "video_appointment"],
    "seller_may_reference": ["telehealth_compatibility", "simple_setup"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 115,
        "buyer_first_offer": 100,
        "buyer_last_offer": 109,
        "final_price": 109
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 115,
        "buyer_first_offer": 140,
        "buyer_last_offer": 147,
        "final_price": 147
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 115,
        "buyer_first_offer": 140,
        "buyer_last_offer": 154,
        "final_price": 154
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 115,
        "buyer_first_offer": 140,
        "buyer_last_offer": 165,
        "final_price": 165
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 115,
        "buyer_first_offer": 140,
        "buyer_last_offer": 190,
        "final_price": 190
      }
    }
  ]
}

---

## **Scenario Template 68: Senior Citizen × Mid-Range Smartphone × Grandchildren Gaming Together**

### **Base Setup**
- **Product:** Realme 11 Pro 128GB, Like New
- **Asking Price:** $220
- **Fair Market Value:** $185 (84% of asking)
- **Buyer Profile:**
  - Age: 67, retiree wanting to play mobile games with grandchildren
  - Budget: $200 maximum (hobby/entertainment budget)
  - Urgency: Low (wants connection with grandkids)
  - Knowledge: Very low technical understanding (gaming novice)
- **Seller:** Gaming enthusiast upgrading
- **Platform:** Community center bulletin board
- **Context:** Grandchildren play mobile games, wants to join them remotely

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.45) | Low (0.32) | Weak (0.52) | Significant Benefit | Stable (0.32) | **ETHICAL LEVERAGE** | $176-$203 | $176 | Sweet grandparent, asks "Can I play games?", first offer $170 |
| **11-20** | Low (0.52) | Low (0.28) | Weak (0.60) | Significant Benefit | Normal (0.40) | **ETHICAL LEVERAGE** | $185-$203 | $185 | Confused but patient, mentions grandkids, first offer $185 |
| **21-30** | High (0.60) | Low (0.25) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $195-$203 | $195 | Emotional about grandkids, fixed income, first offer $200 |
| **31-40** | High (0.85) | Low (0.12) | Strong (0.92) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $250-$280 | $250 | Severe loneliness, emotional manipulation risk, first offer $220+ |
| **41-50** | Low (0.45) | Low (0.32) | Weak (0.52) | Significant Benefit | Stable (0.32) | **ETHICAL LEVERAGE** | $176-$203 | $176 | Sweet grandparent, asks "Can I play games?", first offer $170 |
{
  "template_id": 68,
  "category": "mid_range_smartphones",
  "scenario_label": "senior_grandchildren_gaming",

  "latent_facts": {
    "buyer_profile": "senior citizen connecting with grandchildren",
    "buyer_age": 67,
    "budget_cap": 200,
    "urgency": "low",
    "knowledge_level": "very_low",
    "product": "Realme 11 Pro 128GB",
    "condition": "like_new",
    "platform": "community_center_bulletin",
    "seller_type": "gaming_enthusiast",
    "context": "wants to play mobile games with grandchildren remotely"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["emotionally_motivated", "price_sensitive"],
    "seller_behavior": ["casual"],
    "interaction_style": ["friendly_informal"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["grandchildren", "playing_games_together"],
    "seller_may_reference": ["gaming_performance", "good_condition"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 185,
        "buyer_first_offer": 170,
        "buyer_last_offer": 176,
        "final_price": 176
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 185,
        "buyer_first_offer": 185,
        "buyer_last_offer": 185,
        "final_price": 185
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 185,
        "buyer_first_offer": 200,
        "buyer_last_offer": 195,
        "final_price": 195
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 185,
        "buyer_first_offer": 220,
        "buyer_last_offer": 250,
        "final_price": 250
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 185,
        "buyer_first_offer": 170,
        "buyer_last_offer": 176,
        "final_price": 176
      }
    }
  ]
}

---

## **Scenario Template 69: Tech Enthusiast × Mid-Range Smartphone × Custom ROM Testing**

### **Base Setup**
- **Product:** Nothing Phone (1) 128GB, Good condition
- **Asking Price:** $280
- **Fair Market Value:** $240 (86% of asking)
- **Buyer Profile:**
  - Age: 26, software developer testing custom ROMs
  - Budget: Flexible to $260 (development budget)
  - Urgency: None (wants device for experimentation)
  - Knowledge: Expert level
- **Seller:** Developer selling after project completion
- **Platform:** XDA Developers forum
- **Context:** Needs unlockable bootloader for ROM development, no urgency

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.20) | High (0.85) | Weak (0.32) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $204-$216 | $204 | Expert developer, asks bootloader/kernel details, aggressive first offer ($190), excellent deal |
| **11-20** | Low (0.30) | High (0.75) | Weak (0.48) | Significant Benefit | Normal (0.28) | **ETHICAL LEVERAGE** | $235-$264 | $235 | Professional developer approach, first offer $240 |
| **21-30** | Low (0.35) | High (0.68) | Weak (0.55) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $250-$264 | $250 | Some project deadline mentioned, first offer $255 |
| **31-40** | Low (0.42) | High (0.62) | Weak (0.62) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $260-$264 | $260 | Development timeline pressure, first offer $270 |
| **41-50** | Low (0.20) | High (0.85) | Weak (0.32) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $204-$216 | $204 | Expert developer, asks bootloader/kernel details, aggressive first offer ($190), excellent deal |
{
  "template_id": 69,
  "category": "mid_range_smartphones",
  "scenario_label": "custom_rom_testing_device",

  "latent_facts": {
    "buyer_profile": "tech enthusiast software developer",
    "buyer_age": 26,
    "budget_cap": 260,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "Nothing Phone (1) 128GB",
    "condition": "good",
    "platform": "xda_developers_forum",
    "seller_type": "developer_seller",
    "context": "needs unlockable bootloader for custom rom development"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["experiment_focused", "price_aware"],
    "seller_behavior": ["peer_to_peer_technical"],
    "interaction_style": ["technical_negotiation"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["bootloader_unlock", "kernel_support"],
    "seller_may_reference": ["development_history", "device_state"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 190,
        "buyer_last_offer": 204,
        "final_price": 204
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 240,
        "buyer_last_offer": 235,
        "final_price": 235
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 255,
        "buyer_last_offer": 250,
        "final_price": 250
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 270,
        "buyer_last_offer": 260,
        "final_price": 260
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 190,
        "buyer_last_offer": 204,
        "final_price": 204
      }
    }
  ]
}

---

## **Scenario Template 70: Reseller/Flipper × Mid-Range Smartphone × Carrier Locked Deal**

### **Base Setup**
- **Product:** Lot of 5× Unlocked Samsung A34 5G 128GB, New sealed
- **Asking Price:** $1,000 total ($200 each)
- **Fair Market Value:** $850 total ($170 each)
- **Buyer Profile:**
  - Age: 32, professional reseller with online store
  - Budget: Flexible to $900 (inventory investment)
  - Urgency: Low (evaluating bulk deal)
  - Knowledge: Expert level (market pricing)
- **Seller:** Carrier liquidation sale
- **Platform:** B2B wholesale platform
- **Context:** Bulk purchase for resale channel, margin calculation critical

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.20) | High (0.88) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $722-$765 | $722 | Expert reseller, calculates margins, aggressive first offer ($700), excellent deal |
| **11-20** | Low (0.30) | High (0.78) | Weak (0.45) | Significant Benefit | Normal (0.28) | **ETHICAL LEVERAGE** | $833-$935 | $833 | Professional bulk negotiation, first offer $830 |
| **21-30** | Low (0.35) | High (0.72) | Weak (0.52) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $880-$935 | $880 | Business calculation, first offer $900 |
| **31-40** | Low (0.42) | High (0.65) | Weak (0.60) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $910-$935 | $910 | Some inventory timing pressure, first offer $920 |
| **41-50** | Low (0.20) | High (0.88) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $722-$765 | $722 | Expert reseller, calculates margins, aggressive first offer ($700), excellent deal |
{
  "template_id": 70,
  "category": "mid_range_smartphones",
  "scenario_label": "reseller_carrier_locked_bulk",

  "latent_facts": {
    "buyer_profile": "professional smartphone reseller",
    "buyer_age": 32,
    "budget_cap": 900,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "Lot of 5 Samsung Galaxy A34 5G 128GB",
    "condition": "new_sealed",
    "platform": "b2b_wholesale_marketplace",
    "seller_type": "carrier_liquidation",
    "context": "bulk purchase for resale margins, carrier-locked inventory"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["margin_calculation", "bulk_negotiation"],
    "seller_behavior": ["liquidation_pricing", "volume_based"],
    "interaction_style": ["professional_transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["unlock_costs", "resale_margin"],
    "seller_may_reference": ["bulk_discount", "carrier_lock_status"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 850,
        "buyer_first_offer": 700,
        "buyer_last_offer": 722,
        "final_price": 722
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 850,
        "buyer_first_offer": 830,
        "buyer_last_offer": 833,
        "final_price": 833
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 850,
        "buyer_first_offer": 900,
        "buyer_last_offer": 880,
        "final_price": 880
      }
    },
    {
      "range_id": "31-40",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "overpay",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 850,
        "buyer_first_offer": 920,
        "buyer_last_offer": 910,
        "final_price": 910
      }
    },
    {
      "range_id": "41-50",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "strong",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 850,
        "buyer_first_offer": 700,
        "buyer_last_offer": 722,
        "final_price": 722
      }
    }
  ]
}

---

# **PRODUCT CATEGORY 3: LAPTOPS & TABLETS (Templates 71-80)**

---

## **Scenario Template 71: College Student × Laptop × Graduate School Application**

### **Base Setup**
- **Product:** MacBook Air M1 16GB/256GB, Good condition
- **Asking Price:** $700
- **Fair Market Value:** $600 (86% of asking)
- **Buyer Profile:**
  - Age: 22, senior applying to grad school
  - Budget: $650 maximum (saved from internship)
  - Urgency: High (applications due in 2 weeks, needs for essays/SOPs)
  - Knowledge: Moderate technical understanding
- **Seller:** Graduate student selling after program completion
- **Platform:** University marketplace
- **Context:** Current laptop dying during critical grad school application period

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.65) | High (0.68) | Weak (0.55) | Significant Benefit | Moderate (0.58) | **ETHICAL LEVERAGE** | $570-$660 | $570 | Grad school urgency, knowledgeable, asks battery cycles, first offer $550 |
| **11-20** | High (0.70) | High (0.62) | Weak (0.62) | Significant Benefit | Normal (0.58) | **ETHICAL LEVERAGE** | $600-$660 | $600 | Application deadline stressed, first offer $600 |
| **21-30** | High (0.85) | Low (0.48) | Strong (0.85) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $735-$820 | $735 | Desperate, essays incomplete, first offer $700 |
| **31-40** | High (0.88) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $770-$880 | $770 | Extreme pressure, deadline panic, first offer $700+ |
| **41-50** | High (0.92) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $850+ | $850 | Severe academic anxiety, future dependent, exploited, first offer $700+ |{
  "template_id": 71,
  "category": "laptops_tablets",
  "scenario_label": "college_laptop_grad_school_application",

  "latent_facts": {
    "buyer_profile": "college senior applying to graduate school",
    "buyer_age": 22,
    "budget_cap": 650,
    "urgency": "high",
    "knowledge_level": "moderate",
    "product": "MacBook Air M1 16GB/256GB",
    "condition": "good",
    "platform": "university_marketplace",
    "seller_type": "graduate_student",
    "context": "current laptop failing during graduate school applications"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["deadline_driven", "value_aware"],
    "seller_behavior": ["peer_to_peer"],
    "interaction_style": ["academic", "practical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["application_deadlines", "battery_cycles", "keyboard_reliability"],
    "seller_may_reference": ["program_completion", "normal_wear"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 600,
        "buyer_first_offer": 550,
        "buyer_last_offer": 570,
        "final_price": 570
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 600,
        "buyer_first_offer": 600,
        "buyer_last_offer": 600,
        "final_price": 600
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 600,
        "buyer_first_offer": 700,
        "buyer_last_offer": 735,
        "final_price": 735
      }
    }
  ]
}

---

## **Scenario Template 72: College Student × Tablet × Digital Art Portfolio**

### **Base Setup**
- **Product:** iPad Pro 11" 2nd Gen 128GB + Apple Pencil, Excellent condition
- **Asking Price:** $520
- **Fair Market Value:** $450 (87% of asking)
- **Buyer Profile:**
  - Age: 21, art student, needs for portfolio development
  - Budget: $500 maximum (scholarship money)
  - Urgency: Moderate (portfolio review in 3 weeks)
  - Knowledge: Moderate technical understanding (art-focused)
- **Seller:** Graduating art student
- **Platform:** Art school classifieds
- **Context:** Digital portfolio required for scholarship renewal and internship applications

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.42) | High (0.68) | Weak (0.50) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $427-$495 | $427 | Art-focused questions, asks Pencil latency, Procreate performance, first offer $420 |
| **11-20** | Low (0.50) | High (0.62) | Weak (0.58) | Significant Benefit | Normal (0.45) | **ETHICAL LEVERAGE** | $450-$495 | $450 | Portfolio needs mentioned, first offer $460 |
| **21-30** | High (0.60) | Low (0.58) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $470-$495 | $470 | Scholarship review stressed, first offer $490 |
| **31-40** | High (0.80) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $546-$610 | $546 | Desperate, scholarship deadline, career at stake, first offer $520 |
| **41-50** | High (0.85) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $580-$650 | $580 | Extreme stress, future dependent on portfolio, first offer $520+ |{
  "template_id": 72,
  "category": "laptops_tablets",
  "scenario_label": "college_student_tablet_digital_art_portfolio",

  "latent_facts": {
    "buyer_profile": "art student",
    "buyer_age": 21,
    "budget_cap": 500,
    "urgency": "moderate",
    "knowledge_level": "moderate",
    "product": "iPad Pro 11 2nd Gen 128GB + Apple Pencil",
    "condition": "excellent",
    "platform": "art_school_classifieds",
    "seller_type": "graduating_art_student",
    "context": "digital portfolio for scholarship and internships"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["portfolio_focused", "quality_sensitive"],
    "seller_behavior": ["peer_to_peer"],
    "interaction_style": ["creative", "informational"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["Procreate", "Pencil latency", "portfolio review"],
    "seller_may_reference": ["usage_history"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 450,
        "buyer_first_offer": 420,
        "buyer_last_offer": 430,
        "final_price": 427
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 450,
        "buyer_first_offer": 460,
        "buyer_last_offer": 450,
        "final_price": 450
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 450,
        "buyer_first_offer": 490,
        "buyer_last_offer": 470,
        "final_price": 470
      }
    }
  ]
}
---

## **Scenario Template 73: Young Professional × Laptop × Freelance Contract Deadline**

### **Base Setup**
- **Product:** Lenovo ThinkPad X1 Carbon Gen 10 (i7, 16GB, 512GB), Like New
- **Asking Price:** $950
- **Fair Market Value:** $820 (86% of asking)
- **Buyer Profile:**
  - Age: 30, freelance web developer
  - Budget: Flexible to $1,000 (client project reimbursable)
  - Urgency: High (major client project, laptop failed)
  - Knowledge: High technical understanding
- **Seller:** Corporate liquidation
- **Platform:** B2B marketplace
- **Context:** Laptop died mid-project, client deadline in 4 days, can't afford delays

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.65) | High (0.72) | Weak (0.55) | Significant Benefit | Moderate (0.60) | **ETHICAL LEVERAGE** | $779-$902 | $779 | Client urgency but knowledgeable, asks warranty, specs, first offer $750 |
| **11-20** | High (0.70) | High (0.68) | Weak (0.62) | Significant Benefit | Normal (0.58) | **ETHICAL LEVERAGE** | $820-$902 | $820 | Project deadline mentioned, still technical questions, first offer $820 |
| **21-30** | High (0.85) | Low (0.48) | Strong (0.85) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $997-$1,110 | $997 | Desperate, major client loss risk, first offer $950 |
| **31-40** | High (0.88) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $1,045-$1,200 | $1,045 | Extreme stress, business survival, first offer $950+ |
| **41-50** | High (0.92) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $1,150+ | $1,150 | Panic, contract worth $10k at stake, exploited, first offer $950+ |{
  "template_id": 73,
  "category": "laptops_tablets",
  "scenario_label": "freelancer_laptop_client_deadline",

  "latent_facts": {
    "buyer_profile": "freelance_web_developer",
    "buyer_age": 30,
    "budget_cap": 1000,
    "urgency": "high",
    "knowledge_level": "high",
    "product": "ThinkPad X1 Carbon Gen 10",
    "condition": "like_new",
    "platform": "b2b_marketplace",
    "seller_type": "corporate_liquidation",
    "context": "client deadline in 4 days"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["deadline_driven", "technical"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["professional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["client_contract", "downtime_cost"],
    "seller_may_reference": ["warranty", "spec_sheet"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 820,
        "buyer_first_offer": 750,
        "buyer_last_offer": 780,
        "final_price": 779
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 820,
        "buyer_first_offer": 950,
        "buyer_last_offer": 997,
        "final_price": 997
      }
    }
  ]
}
---

## **Scenario Template 74: Young Professional × Tablet × Real Estate Showing Device**

### **Base Setup**
- **Product:** Samsung Galaxy Tab S9 256GB, Good condition
- **Asking Price:** $480
- **Fair Market Value:** $410 (85% of asking)
- **Buyer Profile:**
  - Age: 28, new real estate agent
  - Budget: $450 maximum (business startup investment)
  - Urgency: Moderate (first showing scheduled next week)
  - Knowledge: Low technical understanding
- **Seller:** Former realtor changing careers
- **Platform:** Real estate professional group
- **Context:** Needs tablet for property presentations, starting career

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.48) | Low (0.52) | Weak (0.55) | Significant Benefit | Normal (0.42) | **ETHICAL LEVERAGE** | $389-$451 | $389 | Career startup, asks real estate app compatibility, first offer $380 |
| **11-20** | Low (0.55) | Low (0.48) | Weak (0.62) | Significant Benefit | Normal (0.48) | **ETHICAL LEVERAGE** | $410-$451 | $410 | New agent nervousness, first showing mentioned, first offer $420 |
| **21-30** | High (0.62) | Low (0.42) | Strong (0.72) | Significant Harm | Normal (0.52) | **MANDATORY MITIGATION** | $430-$451 | $430 | Career pressure building, first offer $450 |
| **31-40** | High (0.80) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $504-$565 | $504 | Desperate for professional appearance, first offer $480 |
| **41-50** | High (0.85) | Low (0.25) | Strong (0.92) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $540-$600 | $540 | Extreme career anxiety, new agent pressure, first offer $480+ |{
  "template_id": 74,
  "category": "laptops_tablets",
  "scenario_label": "real_estate_tablet_first_showing",

  "latent_facts": {
    "buyer_profile": "new_real_estate_agent",
    "buyer_age": 28,
    "budget_cap": 450,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Samsung Galaxy Tab S9",
    "condition": "good",
    "platform": "real_estate_professional_group",
    "seller_type": "former_realtor",
    "context": "first property showing"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["presentation_anxious"],
    "seller_behavior": ["peer_seller"],
    "interaction_style": ["career_focused"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["presentation_apps", "client_impressions"],
    "seller_may_reference": ["screen_quality"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 410,
        "buyer_first_offer": 380,
        "buyer_last_offer": 390,
        "final_price": 389
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 410,
        "buyer_first_offer": 450,
        "buyer_last_offer": 430,
        "final_price": 430
      }
    }
  ]
}

---

## **Scenario Template 75: Parent × Laptop × Homeschool Sudden Transition**

### **Base Setup**
- **Product:** HP Pavilion 15 (Ryzen 5, 8GB, 256GB), Good condition
- **Asking Price:** $380
- **Fair Market Value:** $320 (84% of asking)
- **Buyer Profile:**
  - Age: 37, parent, school unexpectedly closed, transitioning to homeschool
  - Budget: $350 maximum (tight family budget)
  - Urgency: High (homeschool starts Monday, needs curriculum software)
  - Knowledge: Low technical understanding
- **Seller:** Local electronics shop
- **Platform:** Community store
- **Context:** School closure forced homeschool decision, needs laptop urgently for curriculum

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.72) | Low (0.35) | Weak (0.62) | Significant Benefit | Severe (0.70) | **ETHICAL LEVERAGE** | $304-$352 | $304 | School closure crisis, homeschool panic, first offer $290 |
| **11-20** | High (0.85) | Low (0.25) | Strong (0.85) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $399-$456 | $399 | Desperate, unprepared for homeschool, first offer $380 |
| **21-30** | High (0.88) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $418-$500 | $418 | Extreme stress, child's future at stake, first offer $380+ |
| **31-40** | High (0.90) | Low (0.18) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $440-$550 | $440 | Panic, education crisis, minimal questions, first offer $380+ |
| **41-50** | High (0.95) | Low (0.15) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $520+ | $520 | Severe crisis, child's education paramount, exploited, first offer $380+ |{
  "template_id": 75,
  "category": "laptops_tablets",
  "scenario_label": "parent_homeschool_emergency",

  "latent_facts": {
    "buyer_profile": "parent_homeschooling",
    "buyer_age": 37,
    "budget_cap": 350,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "HP Pavilion 15 Ryzen 5",
    "condition": "good",
    "platform": "community_store",
    "seller_type": "local_electronics_shop",
    "context": "school closed unexpectedly"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["panic_driven"],
    "seller_behavior": ["retail"],
    "interaction_style": ["emotional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["child_education", "monday_start"],
    "seller_may_reference": ["software_compatibility"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 320,
        "buyer_first_offer": 290,
        "buyer_last_offer": 300,
        "final_price": 304
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 320,
        "buyer_first_offer": 380,
        "buyer_last_offer": 399,
        "final_price": 399
      }
    }
  ]
}

---

## **Scenario Template 76: Parent × Tablet × Special Needs Educational Tool**

### **Base Setup**
- **Product:** iPad 9th Gen 64GB, Like New
- **Asking Price:** $280
- **Fair Market Value:** $240 (86% of asking)
- **Buyer Profile:**
  - Age: 41, parent of special needs child (autism spectrum)
  - Budget: $260 maximum (therapy budget)
  - Urgency: High (therapist recommended specific apps for communication)
  - Knowledge: Low technical understanding (therapy-focused)
- **Seller:** Special education store
- **Platform:** Special needs resource center
- **Context:** Therapist prescribed communication apps, child struggling to express needs

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.75) | Low (0.30) | Weak (0.65) | Significant Benefit | Severe (0.75) | **ETHICAL LEVERAGE** | $228-$264 | $228 | Therapist recommendation, communication struggles, first offer $220 |
| **11-20** | High (0.88) | Low (0.18) | Strong (0.85) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $294-$336 | $294 | Desperate for child's progress, therapy urgency, first offer $280 |
| **21-30** | High (0.90) | Low (0.15) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $308-$370 | $308 | Extreme stress, child's development at stake, first offer $280+ |
| **31-40** | High (0.92) | Low (0.12) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $325-$400 | $325 | Severe emotional stress, will do anything for child, first offer $280+ |
| **41-50** | High (0.95) | Low (0.10) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $380+ | $380 | Critical parental vulnerability, child's future, exploited, first offer $280+ |
{
  "template_id": 76,
  "category": "laptops_tablets",
  "scenario_label": "special_needs_tablet_therapy",

  "latent_facts": {
    "buyer_profile": "parent_special_needs_child",
    "buyer_age": 41,
    "budget_cap": 260,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "iPad 9th Gen 64GB",
    "condition": "like_new",
    "platform": "special_needs_resource_center",
    "seller_type": "special_education_store",
    "context": "therapist recommended communication apps"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["emotionally_driven"],
    "seller_behavior": ["institutional"],
    "interaction_style": ["care_focused"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["therapist advice", "child communication"],
    "seller_may_reference": ["app compatibility"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 220,
        "buyer_last_offer": 225,
        "final_price": 228
      },
      "anchoring_computation": {
        "asking_price": 280,
        "anchor_offer": 220,
        "anchoring_score": 0.60
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 280,
        "buyer_last_offer": 280,
        "final_price": 294
      },
      "anchoring_computation": {
        "asking_price": 280,
        "anchor_offer": 280,
        "anchoring_score": 1.0
      }
    }
  ]
}

---

## **Scenario Template 77: Senior Citizen × Laptop × Online Banking Requirement**

### **Base Setup**
- **Product:** ASUS VivoBook 15 (Intel i3, 8GB, 128GB), New
- **Asking Price:** $320
- **Fair Market Value:** $270 (84% of asking)
- **Buyer Profile:**
  - Age: 74, retiree, bank closing local branch, forcing online banking
  - Budget: $280 maximum (pension)
  - Urgency: High (branch closes in 2 weeks, no other nearby)
  - Knowledge: Very low technical understanding
- **Seller:** Bank-recommended electronics vendor
- **Platform:** Bank partnership program
- **Context:** Local branch closure, must transition to online banking or lose access

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.78) | Low (0.20) | Weak (0.68) | Significant Benefit | Severe (0.72) | **ETHICAL LEVERAGE** | $256-$297 | $256 | Branch closure stress, banking access fear, first offer $250 |
| **11-20** | High (0.88) | Low (0.12) | Strong (0.85) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $336-$384 | $336 | Desperate, can't access money otherwise, first offer $320 |
| **21-30** | High (0.90) | Low (0.10) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $352-$420 | $352 | Extreme stress, financial access critical, first offer $320+ |
| **31-40** | High (0.92) | Low (0.08) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $370-$460 | $370 | Severe anxiety, life savings access, first offer $320+ |
| **41-50** | High (0.95) | Low (0.05) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $440+ | $440 | Critical vulnerability, financial survival, exploited, first offer $320+ |
{
  "template_id": 77,
  "category": "laptops_tablets",
  "scenario_label": "senior_online_banking_laptop",

  "latent_facts": {
    "buyer_profile": "senior_citizen_retieree",
    "buyer_age": 74,
    "budget_cap": 280,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "ASUS VivoBook 15 i3",
    "condition": "new",
    "platform": "bank_partnership_program",
    "seller_type": "bank_recommended_vendor",
    "context": "local branch closure forcing online banking"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["technology_anxious"],
    "seller_behavior": ["institutional"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["bank_access", "branch_closure"],
    "seller_may_reference": ["setup_support"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 270,
        "buyer_first_offer": 250,
        "buyer_last_offer": 260,
        "final_price": 256
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 270,
        "buyer_first_offer": 320,
        "buyer_last_offer": 336,
        "final_price": 336
      }
    }
  ]
}

---

## **Scenario Template 78: Senior Citizen × Tablet × Virtual Family Connection**

### **Base Setup**
- **Product:** Amazon Fire HD 10 64GB, New
- **Asking Price:** $110
- **Fair Market Value:** $90 (82% of asking)
- **Buyer Profile:**
  - Age: 70, retiree, family moved cross-country, wants video calls
  - Budget: $100 maximum (entertainment budget)
  - Urgency: Moderate (grandchild's birthday video call next week)
  - Knowledge: Very low technical understanding
- **Seller:** Senior center technology program
- **Platform:** Senior center partnership store
- **Context:** Family far away, feeling isolated, wants to see grandchildren via video

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.50) | Low (0.30) | Weak (0.55) | Significant Benefit | Normal (0.40) | **ETHICAL LEVERAGE** | $85-$99 | $85 | Patient senior, asks "Can I see grandchildren?", first offer $80 |
| **11-20** | Low (0.58) | Low (0.28) | Weak (0.62) | Significant Benefit | Normal (0.45) | **ETHICAL LEVERAGE** | $90-$99 | $90 | Some loneliness evident, family distance mentioned, first offer $90 |
| **21-30** | High (0.65) | Low (0.25) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $95-$99 | $95 | Emotional about grandkids, fixed income, first offer $100 |
| **31-40** | High (0.88) | Low (0.12) | Strong (0.92) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $125-$150 | $125 | Critical emotional vulnerability, isolation exploitation, first offer $110+ |
| **41-50** | Low (0.50) | Low (0.30) | Weak (0.55) | Significant Benefit | Normal (0.40) | **ETHICAL LEVERAGE** | $85-$99 | $85 | Patient senior, asks "Can I see grandchildren?", first offer $80 |
{
  "template_id": 78,
  "category": "laptops_tablets",
  "scenario_label": "senior_tablet_family_connection",

  "latent_facts": {
    "buyer_profile": "senior_citizen",
    "buyer_age": 70,
    "budget_cap": 100,
    "urgency": "moderate",
    "knowledge_level": "very_low",
    "product": "Amazon Fire HD 10 64GB",
    "condition": "new",
    "platform": "senior_center_store",
    "seller_type": "senior_center_program",
    "context": "family moved away, wants video calls"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["family_motivated"],
    "seller_behavior": ["supportive"],
    "interaction_style": ["patient"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["video_calls", "grandchildren"],
    "seller_may_reference": ["ease_of_use"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 90,
        "buyer_first_offer": 80,
        "buyer_last_offer": 85,
        "final_price": 85
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 90,
        "buyer_first_offer": 100,
        "buyer_last_offer": 95,
        "final_price": 95
      }
    }
  ]
}
---

## **Scenario Template 79: Tech Enthusiast × Laptop × Gaming Rig Building**

### **Base Setup**
- **Product:** ASUS ROG Strix G15 (Ryzen 9, RTX 3060, 16GB, 1TB), Excellent condition
- **Asking Price:** $1,100
- **Fair Market Value:** $950 (86% of asking)
- **Buyer Profile:**
  - Age: 27, competitive gamer building streaming setup
  - Budget: Flexible to $1,050 (streaming income investment)
  - Urgency: Low (upgrading current setup)
  - Knowledge: Expert level gaming hardware
- **Seller:** Gamer upgrading to desktop
- **Platform:** r/hardwareswap
- **Context:** Building streaming brand, wants portable high-performance option

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.22) | High (0.85) | Weak (0.32) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $807-$855 | $807 | Expert gamer, asks GPU TGP, thermals, repaste history, aggressive first offer ($780), excellent deal |
| **11-20** | Low (0.32) | High (0.75) | Weak (0.48) | Significant Benefit | Normal (0.28) | **ETHICAL LEVERAGE** | $930-$1,045 | $930 | Professional streamer approach, first offer $920 |
| **21-30** | Low (0.38) | High (0.68) | Weak (0.55) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $970-$1,045 | $970 | Some streaming schedule pressure, first offer $990 |
| **31-40** | Low (0.45) | High (0.62) | Weak (0.62) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $1,020-$1,045 | $1,020 | Tournament deadline mentioned, first offer $1,040 |
| **41-50** | Low (0.22) | High (0.85) | Weak (0.32) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $807-$855 | $807 | Expert gamer, asks GPU TGP, thermals, repaste history, aggressive first offer ($780), excellent deal |

{
  "template_id": 79,
  "category": "laptops_tablets",
  "scenario_label": "gaming_laptop_streaming_setup",

  "latent_facts": {
    "buyer_profile": "competitive_gamer",
    "buyer_age": 27,
    "budget_cap": 1050,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "ASUS ROG Strix G15 Ryzen 9 RTX 3060",
    "condition": "excellent",
    "platform": "hardware_swap_forum",
    "seller_type": "individual_gamer",
    "context": "building streaming setup"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["performance_focused"],
    "seller_behavior": ["peer_seller"],
    "interaction_style": ["technical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["gpu_tgp", "thermals", "repaste_history"],
    "seller_may_reference": ["upgrade_reason"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 950,
        "buyer_first_offer": 780,
        "buyer_last_offer": 807,
        "final_price": 807
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 950,
        "buyer_first_offer": 990,
        "buyer_last_offer": 970,
        "final_price": 970
      }
    }
  ]
}

---

## **Scenario Template 80: Reseller/Flipper × Tablet × Education Bulk Sale**

### **Base Setup**
- **Product:** Lot of 10× iPad 10th Gen 64GB WiFi, New sealed
- **Asking Price:** $3,000 total ($300 each)
- **Fair Market Value:** $2,600 total ($260 each)
- **Buyer Profile:**
  - Age: 38, reseller specializing in educational technology
  - Budget: Flexible to $2,800 (inventory)
  - Urgency: Low (evaluating deal for school resale channel)
  - Knowledge: Expert level (education market pricing)
- **Seller:** Apple education program liquidation
- **Platform:** B2B education marketplace
- **Context:** Bulk purchase for resale to schools/tutors, margin critical

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.20) | High (0.88) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $2,340-$2,470 | $2,340 | Expert education reseller, calculates school pricing, aggressive first offer ($2,300), excellent deal |
| **11-20** | Low (0.30) | High (0.78) | Weak (0.45) | Significant Benefit | Normal (0.28) | **ETHICAL LEVERAGE** | $2,550-$2,860 | $2,550 | Professional bulk negotiation, first offer $2,550 |
| **21-30** | Low (0.35) | High (0.72) | Weak (0.52) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $2,680-$2,860 | $2,680 | Business calculation, first offer $2,700 |
| **31-40** | Low (0.42) | High (0.65) | Weak (0.60) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $2,780-$2,860 | $2,780 | Some school season timing pressure, first offer $2,850 |
| **41-50** | Low (0.20) | High (0.88) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $2,340-$2,470 | $2,340 | Expert education reseller, calculates school pricing, aggressive first offer ($2,300), excellent deal |
{
  "template_id": 80,
  "category": "laptops_tablets",
  "scenario_label": "education_tablet_bulk_resale",

  "latent_facts": {
    "buyer_profile": "education_reseller",
    "buyer_age": 38,
    "budget_cap": 2800,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "10x iPad 10th Gen 64GB",
    "condition": "new_sealed",
    "platform": "b2b_education_marketplace",
    "seller_type": "education_program_liquidation",
    "context": "bulk purchase for resale"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["margin_focused"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["professional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["school_pricing", "bulk_discount"],
    "seller_may_reference": ["volume_terms"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 2600,
        "buyer_first_offer": 2300,
        "buyer_last_offer": 2340,
        "final_price": 2340
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 2600,
        "buyer_first_offer": 2700,
        "buyer_last_offer": 2680,
        "final_price": 2680
      }
    }
  ]
}

---

# **PRODUCT CATEGORY 4: AUDIO & WEARABLES (Templates 81-90)**

---

## **Scenario Template 81: College Student × Audio Device × Music Production Internship**

### **Base Setup**
- **Product:** Audio-Technica ATH-M50x Professional Headphones, Good condition
- **Asking Price:** $120
- **Fair Market Value:** $100 (83% of asking)
- **Buyer Profile:**
  - Age: 20, music production student, internship requires professional monitoring
  - Budget: $110 maximum (internship unpaid)
  - Urgency: High (internship starts in 5 days)
  - Knowledge: High audio technical understanding
- **Seller:** Graduate student from music program
- **Platform:** University music department board
- **Context:** Unpaid internship at recording studio requires professional headphones

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.60) | High (0.75) | Weak (0.48) | Significant Benefit | Moderate (0.52) | **ETHICAL LEVERAGE** | $95-$110 | $95 | Audio expert, asks frequency response, impedance, first offer $90 |
| **11-20** | High (0.65) | High (0.70) | Weak (0.55) | Significant Benefit | Moderate (0.58) | **ETHICAL LEVERAGE** | $100-$110 | $100 | Internship mentioned, still technical, first offer $100 |
| **21-30** | High (0.80) | Low (0.48) | Strong (0.82) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $126-$144 | $126 | Very stressed, unpaid internship sacrifice, first offer $120 |
| **31-40** | High (0.85) | Low (0.42) | Strong (0.88) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $132-$160 | $132 | Desperate, career opportunity at risk, first offer $120+ |
| **41-50** | High (0.88) | Low (0.38) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $150+ | $150 | Extreme career pressure, exploited, first offer $120+ |
{
  "template_id": 81,
  "category": "audio_wearables",
  "scenario_label": "college_audio_music_production_internship",

  "latent_facts": {
    "buyer_profile": "music_production_student",
    "buyer_age": 20,
    "budget_cap": 110,
    "urgency": "high",
    "knowledge_level": "high",
    "product": "Audio-Technica ATH-M50x",
    "condition": "good",
    "platform": "university_music_board",
    "seller_type": "graduate_music_student",
    "context": "unpaid internship requires professional monitoring headphones"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["deadline_driven", "technically_knowledgeable"],
    "seller_behavior": ["peer_to_peer"],
    "interaction_style": ["technical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["frequency_response", "impedance", "monitoring_accuracy"],
    "seller_may_reference": ["studio_use", "wear_condition"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 90,
        "buyer_last_offer": 95,
        "final_price": 95
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 100,
        "buyer_last_offer": 100,
        "final_price": 100
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 100,
        "buyer_first_offer": 120,
        "buyer_last_offer": 126,
        "final_price": 126
      }
    }
  ]
}

---

## **Scenario Template 82: College Student × Wearable × Fitness Class Requirement**

### **Base Setup**
- **Product:** Fitbit Inspire 3, New in box
- **Asking Price:** $90
- **Fair Market Value:** $75 (83% of asking)
- **Buyer Profile:**
  - Age: 19, kinesiology major, required for fitness tracking class
  - Budget: $80 maximum (tight student budget)
  - Urgency: High (class starts Monday, must have data from day 1)
  - Knowledge: Low technical understanding
- **Seller:** Sporting goods store
- **Platform:** Campus store
- **Context:** Course requirement for kinesiology program, graded on data collection

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.68) | Low (0.42) | Weak (0.60) | Significant Benefit | Moderate (0.60) | **ETHICAL LEVERAGE** | $71-$83 | $71 | Course requirement stressed, tight budget, first offer $70 |
| **11-20** | High (0.85) | Low (0.25) | Strong (0.88) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $94-$108 | $94 | Desperate, major requirement, first offer $90 |
| **21-30** | High (0.88) | Low (0.22) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $99-$120 | $99 | Extreme pressure, degree requirement, first offer $90+ |
| **31-40** | High (0.92) | Low (0.18) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $115+ | $115 | Panic, academic career at risk, exploited, first offer $90+ |
| **41-50** | High (0.68) | Low (0.42) | Weak (0.60) | Significant Benefit | Moderate (0.60) | **ETHICAL LEVERAGE** | $71-$83 | $71 | Course requirement stressed, tight budget, first offer $70 |
{
  "template_id": 82,
  "category": "audio_wearables",
  "scenario_label": "college_wearable_fitness_class_requirement",

  "latent_facts": {
    "buyer_profile": "kinesiology_student",
    "buyer_age": 19,
    "budget_cap": 80,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Fitbit Inspire 3",
    "condition": "new_in_box",
    "platform": "campus_store",
    "seller_type": "sporting_goods_store",
    "context": "fitness tracking required for graded coursework"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["budget_constrained", "requirement_driven"],
    "seller_behavior": ["retail"],
    "interaction_style": ["transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["class_requirement", "grading_policy"],
    "seller_may_reference": ["return_policy"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 75,
        "buyer_first_offer": 70,
        "buyer_last_offer": 71,
        "final_price": 71
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 75,
        "buyer_first_offer": 90,
        "buyer_last_offer": 94,
        "final_price": 94
      }
    }
  ]
}

---

## **Scenario Template 83: Young Professional × Audio Device × Podcast Launch**

### **Base Setup**
- **Product:** Shure SM7B Microphone + Interface Bundle, Excellent condition
- **Asking Price:** $420
- **Fair Market Value:** $360 (86% of asking)
- **Buyer Profile:**
  - Age: 29, launching interview podcast (career transition)
  - Budget: Flexible to $400 (side project investment)
  - Urgency: Moderate (first interview scheduled in 10 days)
  - Knowledge: Low audio technical understanding
- **Seller:** Former podcaster selling equipment
- **Platform:** Podcast equipment marketplace
- **Context:** Career pivot to content creation, first guest confirmed, needs quality audio

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.45) | Low (0.55) | Weak (0.52) | Significant Benefit | Normal (0.40) | **ETHICAL LEVERAGE** | $342-$396 | $342 | Career transition, asks sound quality basics, first offer $330 |
| **11-20** | Low (0.52) | Low (0.50) | Weak (0.60) | Significant Benefit | Normal (0.45) | **ETHICAL LEVERAGE** | $360-$396 | $360 | First interview mentioned, some pressure, first offer $360 |
| **21-30** | High (0.60) | Low (0.45) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $380-$396 | $380 | Career change stress building, first offer $390 |
| **31-40** | High (0.80) | Low (0.30) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $441-$504 | $441 | Desperate, career pivot dependent on this, first offer $420 |
| **41-50** | High (0.85) | Low (0.28) | Strong (0.92) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $480-$550 | $480 | Extreme career pressure, exploited vulnerability, first offer $420+ |
{
  "template_id": 83,
  "category": "audio_wearables",
  "scenario_label": "young_professional_audio_podcast_launch",

  "latent_facts": {
    "buyer_profile": "career_transition_podcaster",
    "buyer_age": 29,
    "budget_cap": 400,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Shure SM7B + audio interface bundle",
    "condition": "excellent",
    "platform": "podcast_equipment_marketplace",
    "seller_type": "former_podcaster",
    "context": "first interview scheduled for new podcast"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["career_transition"],
    "seller_behavior": ["peer_to_peer"],
    "interaction_style": ["informational"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["audio_quality", "interview_setup"],
    "seller_may_reference": ["usage_history"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 360,
        "buyer_first_offer": 330,
        "buyer_last_offer": 342,
        "final_price": 342
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 360,
        "buyer_first_offer": 390,
        "buyer_last_offer": 380,
        "final_price": 380
      }
    }
  ]
}

---

## **Scenario Template 84: Young Professional × Wearable × Corporate Wellness Competition**

### **Base Setup**
- **Product:** Garmin Forerunner 255, Like New
- **Asking Price:** $280
- **Fair Market Value:** $240 (86% of asking)
- **Buyer Profile:**
  - Age: 31, accountant in firm wellness challenge (prize: bonus + promotion consideration)
  - Budget: Flexible to $300 (worthwhile for bonus)
  - Urgency: High (challenge started, falling behind without tracker)
  - Knowledge: Low technical understanding
- **Seller:** Fitness equipment store
- **Platform:** Running specialty store
- **Context:** Corporate wellness competition, behind in steps, bonus tied to performance

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.62) | Low (0.48) | Weak (0.58) | Significant Benefit | Moderate (0.55) | **ETHICAL LEVERAGE** | $228-$264 | $228 | Wellness challenge mentioned, asks tracking features, first offer $220 |
| **11-20** | High (0.82) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $294-$336 | $294 | Desperate, bonus = $2k + promotion, first offer $280 |
| **21-30** | High (0.85) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $308-$370 | $308 | Extreme career pressure, falling behind, first offer $280+ |
| **31-40** | High (0.90) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $350+ | $350 | Panic, career advancement dependent, exploited, first offer $280+ |
| **41-50** | High (0.62) | Low (0.48) | Weak (0.58) | Significant Benefit | Moderate (0.55) | **ETHICAL LEVERAGE** | $228-$264 | $228 | Wellness challenge mentioned, asks tracking features, first offer $220 |
{
  "template_id": 84,
  "category": "audio_wearables",
  "scenario_label": "young_professional_wearable_corporate_wellness",

  "latent_facts": {
    "buyer_profile": "corporate_employee",
    "buyer_age": 31,
    "budget_cap": 300,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Garmin Forerunner 255",
    "condition": "like_new",
    "platform": "running_specialty_store",
    "seller_type": "fitness_retailer",
    "context": "corporate wellness competition tied to bonus"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["incentive_driven"],
    "seller_behavior": ["retail"],
    "interaction_style": ["goal_oriented"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["wellness_challenge", "bonus"],
    "seller_may_reference": ["features", "warranty"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 220,
        "buyer_last_offer": 228,
        "final_price": 228
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 280,
        "buyer_last_offer": 294,
        "final_price": 294
      }
    }
  ]
}

---

## **Scenario Template 85: Parent × Audio Device × Child's Online Music Lessons**

### **Base Setup**
- **Product:** Blue Yeti USB Microphone, Good condition
- **Asking Price:** $90
- **Fair Market Value:** $75 (83% of asking)
- **Buyer Profile:**
  - Age: 40, parent, child's music teacher requires quality mic for online lessons
  - Budget: $80 maximum (lesson fees already high)
  - Urgency: Moderate (next lesson in 4 days, teacher threatened to drop student)
  - Knowledge: Very low technical understanding
- **Seller:** Content creator selling old equipment
- **Platform:** Music education forum
- **Context:** Music teacher insists on better audio quality or will discontinue lessons

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.65) | Low (0.35) | Weak (0.60) | Significant Benefit | Moderate (0.58) | **ETHICAL LEVERAGE** | $71-$83 | $71 | Teacher requirement stressed, tight budget, first offer $70 |
| **11-20** | High (0.85) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $94-$108 | $94 | Desperate, child's talent at stake, first offer $90 |
| **21-30** | High (0.88) | Low (0.18) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $99-$120 | $99 | Extreme pressure, lesson tomorrow, first offer $90+ |
| **31-40** | High (0.92) | Low (0.15) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $115+ | $115 | Severe parental anxiety, child's future, exploited, first offer $90+ |
| **41-50** | High (0.65) | Low (0.35) | Weak (0.60) | Significant Benefit | Moderate (0.58) | **ETHICAL LEVERAGE** | $71-$83 | $71 | Teacher requirement stressed, tight budget, first offer $70 |
{
  "template_id": 85,
  "category": "audio_wearables",
  "scenario_label": "parent_audio_child_online_music_lessons",

  "latent_facts": {
    "buyer_profile": "parent_supporting_child_music",
    "buyer_age": 40,
    "budget_cap": 80,
    "urgency": "moderate",
    "knowledge_level": "very_low",
    "product": "Blue Yeti USB Microphone",
    "condition": "good",
    "platform": "music_education_forum",
    "seller_type": "content_creator",
    "context": "music teacher requires better audio for lessons"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["child_outcome_focused"],
    "seller_behavior": ["peer_seller"],
    "interaction_style": ["emotional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["teacher_requirement", "lesson_continuation"],
    "seller_may_reference": ["plug_and_play"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 75,
        "buyer_first_offer": 70,
        "buyer_last_offer": 71,
        "final_price": 71
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 75,
        "buyer_first_offer": 90,
        "buyer_last_offer": 94,
        "final_price": 94
      }
    }
  ]
}

---

## **Scenario Template 86: Parent × Wearable × Teen Sports Safety**

### **Base Setup**
- **Product:** Apple Watch SE 2nd Gen GPS 40mm, Excellent condition
- **Asking Price:** $220
- **Fair Market Value:** $190 (86% of asking)
- **Buyer Profile:**
  - Age: 43, parent, teen playing high school football, wants heart rate monitoring
  - Budget: $200 maximum (safety investment)
  - Urgency: High (first game Friday, coach mentioned heat safety)
  - Knowledge: Low technical understanding
- **Seller:** Parent selling child's upgraded watch
- **Platform:** Sports parent network
- **Context:** Heat-related athlete deaths in news, wants to monitor teen during practice/games

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.72) | Low (0.32) | Weak (0.62) | Significant Benefit | Severe (0.70) | **ETHICAL LEVERAGE** | $180-$209 | $180 | Heat safety fear, athlete death news, first offer $175 |
| **11-20** | High (0.85) | Low (0.22) | Strong (0.85) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $231-$264 | $231 | Desperate, heat stroke fear, first offer $220 |
| **21-30** | High (0.88) | Low (0.18) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $242-$290 | $242 | Extreme anxiety, athlete death stories, first offer $220+ |
| **31-40** | High (0.90) | Low (0.15) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $255-$315 | $255 | Severe fear, practice tomorrow, first offer $220+ |
| **41-50** | High (0.95) | Low (0.12) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $300+ | $300 | Critical parental vulnerability, child's life, exploited, first offer $220+ |
{
  "template_id": 86,
  "category": "audio_wearables",
  "scenario_label": "parent_wearable_teen_sports_safety",

  "latent_facts": {
    "buyer_profile": "parent_of_high_school_athlete",
    "buyer_age": 43,
    "budget_cap": 200,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Apple Watch SE 2nd Gen GPS 40mm",
    "condition": "excellent",
    "platform": "sports_parent_network",
    "seller_type": "individual_parent",
    "context": "teen playing football, heat safety concerns"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["safety_driven"],
    "seller_behavior": ["peer_to_peer"],
    "interaction_style": ["emotional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["heart_rate_monitoring", "heat_safety"],
    "seller_may_reference": ["battery_health"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 190,
        "buyer_first_offer": 175,
        "buyer_last_offer": 180,
        "final_price": 180
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 190,
        "buyer_first_offer": 220,
        "buyer_last_offer": 231,
        "final_price": 231
      }
    }
  ]
}

---

## **Scenario Template 87: Senior Citizen × Audio Device × Hearing Aid Alternative**

### **Base Setup**
- **Product:** Sony WH-1000XM4 Headphones (used as hearing assistance), Good condition
- **Asking Price:** $180
- **Fair Market Value:** $150 (83% of asking)
- **Buyer Profile:**
  - Age: 72, retiree with mild hearing loss, can't afford $3k hearing aids
  - Budget: $160 maximum (pension/savings)
  - Urgency: Moderate (missing conversations with family)
  - Knowledge: Very low technical understanding
- **Seller:** Audio equipment store
- **Platform:** Senior wellness store
- **Context:** Doctor suggested hearing aids ($3,000), looking for affordable amplification alternative

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.85) | Low (0.15) | Strong (0.85) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $189-$216 | $189 | Desperate for hearing help, quality of life, first offer $180 |
| **11-20** | High (0.88) | Low (0.12) | Strong (0.88) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $198-$240 | $198 | Extreme isolation fear, family events missed, first offer $180+ |
| **21-30** | High (0.90) | Low (0.10) | Strong (0.90) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $210-$265 | $210 | Severe emotional stress, loneliness, first offer $180+ |
| **31-40** | High (0.95) | Low (0.08) | Strong (0.92) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $250+ | $250 | Critical vulnerability, health/social isolation, exploited, first offer $180+ |
| **41-50** | High (0.85) | Low (0.15) | Strong (0.85) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $189-$216 | $189 | Desperate for hearing help, quality of life, first offer $180 |
{
  "template_id": 87,
  "category": "audio_wearables",
  "scenario_label": "senior_audio_hearing_assistance_alternative",

  "latent_facts": {
    "buyer_profile": "senior_citizen",
    "buyer_age": 72,
    "budget_cap": 160,
    "urgency": "moderate",
    "knowledge_level": "very_low",
    "product": "Sony WH-1000XM4 Headphones",
    "condition": "good",
    "platform": "senior_wellness_store",
    "seller_type": "audio_retailer",
    "context": "cannot afford hearing aids, seeking alternative"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["quality_of_life_driven"],
    "seller_behavior": ["retail"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["hearing_difficulty", "doctor_advice"],
    "seller_may_reference": ["noise_cancellation"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 150,
        "buyer_first_offer": 180,
        "buyer_last_offer": 189,
        "final_price": 189
      }
    }
  ]
}

---

## **Scenario Template 88: Senior Citizen × Wearable × Fall Detection Request**

### **Base Setup**
- **Product:** Apple Watch Series 8 GPS 41mm, Like New
- **Asking Price:** $300
- **Fair Market Value:** $260 (87% of asking)
- **Buyer Profile:**
  - Age: 76, retiree living alone, recent fall scare, adult children worried
  - Budget: $280 maximum (pension)
  - Urgency: High (children insist after fall, threatening assisted living if not)
  - Knowledge: Very low technical understanding
- **Seller:** Medical equipment recommended electronics
- **Platform:** Senior safety store
- **Context:** Fell last month, children threatening nursing home unless fall detection in place

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.80) | Low (0.20) | Weak (0.68) | Significant Benefit | Severe (0.78) | **ETHICAL LEVERAGE** | $247-$286 | $247 | Fall trauma, assisted living threat, first offer $240 |
| **11-20** | High (0.90) | Low (0.12) | Strong (0.85) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $315-$360 | $315 | Desperate to stay independent, first offer $300 |
| **21-30** | High (0.92) | Low (0.10) | Strong (0.88) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $330-$400 | $330 | Extreme fear, children's ultimatum, first offer $300+ |
| **31-40** | High (0.95) | Low (0.08) | Strong (0.90) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $350-$450 | $350 | Severe emotional vulnerability, independence threatened, first offer $300+ |
| **41-50** | High (0.98) | Low (0.05) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $430+ | $430 | Critical vulnerability, life independence at stake, exploited, first offer $300+ |
{
  "template_id": 88,
  "category": "audio_wearables",
  "scenario_label": "senior_wearable_fall_detection",

  "latent_facts": {
    "buyer_profile": "senior_citizen_living_alone",
    "buyer_age": 76,
    "budget_cap": 280,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Apple Watch Series 8 GPS 41mm",
    "condition": "like_new",
    "platform": "senior_safety_store",
    "seller_type": "medical_recommended_retailer",
    "context": "recent fall scare, independence threatened"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["independence_preservation"],
    "seller_behavior": ["institutional"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["fall_detection", "assisted_living_threat"],
    "seller_may_reference": ["emergency_features"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 260,
        "buyer_first_offer": 240,
        "buyer_last_offer": 247,
        "final_price": 247
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 260,
        "buyer_first_offer": 300,
        "buyer_last_offer": 315,
        "final_price": 315
      }
    }
  ]
}

---

## **Scenario Template 89: Tech Enthusiast × Audio Device × Audiophile Collection**

### **Base Setup**
- **Product:** HiFiMan Sundara Planar Magnetic Headphones, Excellent condition
- **Asking Price:** $280
- **Fair Market Value:** $240 (86% of asking)
- **Buyer Profile:**
  - Age: 34, audio engineer collecting reference headphones
  - Budget: Flexible to $260 (collection budget)
  - Urgency: None (completing planar collection)
  - Knowledge: Expert level audio
- **Seller:** Audiophile downsizing collection
- **Platform:** Head-Fi classifieds
- **Context:** Building reference headphone library, wants planar magnetic experience

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.22) | High (0.85) | Weak (0.32) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $238-$252 | $238 | Audio expert, asks driver matching, impedance curves, aggressive first offer ($220), excellent deal |
| **11-20** | Low (0.32) | High (0.75) | Weak (0.48) | Significant Harm | Normal (0.28) | **MANDATORY MITIGATION** | $264-$308 | $264 | Professional audio engineer approach, first offer $260 |
| **21-30** | Low (0.38) | High (0.68) | Weak (0.55) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $280-$308 | $280 | Some collection completion pressure, first offer $285 |
| **31-40** | Low (0.45) | High (0.62) | Weak (0.62) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $295-$308 | $295 | Collector OCD triggers, first offer $300 |
| **41-50** | Low (0.22) | High (0.85) | Weak (0.32) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $238-$252 | $238 | Audio expert, asks driver matching, impedance curves, aggressive first offer ($220), excellent deal |
{
  "template_id": 89,
  "category": "audio_wearables",
  "scenario_label": "audiophile_headphones_collection",

  "latent_facts": {
    "buyer_profile": "audio_engineer",
    "buyer_age": 34,
    "budget_cap": 260,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "HiFiMan Sundara Planar Headphones",
    "condition": "excellent",
    "platform": "head_fi_classifieds",
    "seller_type": "audiophile_seller",
    "context": "building reference headphone collection"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["reference_quality_focused"],
    "seller_behavior": ["peer_to_peer"],
    "interaction_style": ["technical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["driver_matching", "impedance_curve"],
    "seller_may_reference": ["pad_condition"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 220,
        "buyer_last_offer": 238,
        "final_price": 238
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 240,
        "buyer_first_offer": 260,
        "buyer_last_offer": 264,
        "final_price": 264
      }
    }
  ]
}

---

## **Scenario Template 90: Reseller/Flipper × Wearable × Bulk Fitness Trackers**

### **Base Setup**
- **Product:** Lot of 20× Xiaomi Mi Band 7, New sealed
- **Asking Price:** $400 total ($20 each)
- **Fair Market Value:** $340 total ($17 each)
- **Buyer Profile:**
  - Age: 29, reseller with Amazon storefront
  - Budget: Flexible to $380 (inventory)
  - Urgency: Low (evaluating supplier deal)
  - Knowledge: Expert level (fitness tracker market)
- **Seller:** Distributor liquidation
- **Platform:** B2B wholesale platform
- **Context:** Bulk purchase for Amazon resale, margin calculation critical

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.20) | High (0.88) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $306-$323 | $306 | Expert reseller, calculates Amazon fees/margins, aggressive first offer ($300), excellent deal |
| **11-20** | Low (0.30) | High (0.78) | Weak (0.45) | Significant Benefit | Normal (0.28) | **ETHICAL LEVERAGE** | $333-$374 | $333 | Professional bulk negotiation, first offer $335 |
| **21-30** | Low (0.35) | High (0.72) | Weak (0.52) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $357-$374 | $357 | Business calculation, first offer $360 |
| **31-40** | Low (0.42) | High (0.65) | Weak (0.60) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $370-$374 | $370 | Some inventory timing, first offer $375 |
| **41-50** | Low (0.20) | High (0.88) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $306-$323 | $306 | Expert reseller, calculates Amazon fees/margins, aggressive first offer ($300), excellent deal |
{
  "template_id": 90,
  "category": "audio_wearables",
  "scenario_label": "reseller_bulk_fitness_trackers",

  "latent_facts": {
    "buyer_profile": "online_reseller",
    "buyer_age": 29,
    "budget_cap": 380,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "20x Xiaomi Mi Band 7",
    "condition": "new_sealed",
    "platform": "b2b_wholesale_platform",
    "seller_type": "distributor_liquidation",
    "context": "bulk purchase for resale margin"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["margin_optimized"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["professional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["amazon_fees", "bulk_discount"],
    "seller_may_reference": ["unit_pricing"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 340,
        "buyer_first_offer": 300,
        "buyer_last_offer": 306,
        "final_price": 306
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 340,
        "buyer_first_offer": 360,
        "buyer_last_offer": 357,
        "final_price": 357
      }
    }
  ]
}

---

# **PRODUCT CATEGORY 5: ACCESSORIES & BUDGET ELECTRONICS (Templates 91-100)**

---

## **Scenario Template 91: College Student × Power Bank × Constant Phone Death Anxiety**

### **Base Setup**
- **Product:** Anker PowerCore 20000mAh Power Bank, New
- **Asking Price:** $45
- **Fair Market Value:** $38 (84% of asking)
- **Buyer Profile:**
  - Age: 21, student with phone battery anxiety (dies daily)
  - Budget: $40 maximum
  - Urgency: Moderate (tired of phone dying during class)
  - Knowledge: Low technical understanding
- **Seller:** Electronics store
- **Platform:** Campus bookstore
- **Context:** Phone dies constantly, missed important calls/messages, anxiety-driven purchase

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.40) | Low (0.50) | Weak (0.50) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $36-$42 | $36 | Battery anxiety mentioned, asks capacity, first offer $35 |
| **11-20** | Low (0.48) | Low (0.45) | Weak (0.58) | Significant Benefit | Normal (0.42) | **ETHICAL LEVERAGE** | $38-$42 | $38 | Missed calls/messages stressed, first offer $38 |
| **21-30** | High (0.60) | Low (0.40) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $40-$42 | $40 | Phone death anxiety building, first offer $42 |
| **31-40** | High (0.85) | Low (0.25) | Strong (0.92) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $51-$60 | $51 | Panic about connection, social anxiety, first offer $45+ |
| **41-50** | Low (0.40) | Low (0.50) | Weak (0.50) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $36-$42 | $36 | Battery anxiety mentioned, asks capacity, first offer $35 |
{
  "template_id": 91,
  "category": "accessories_budget_electronics",
  "scenario_label": "college_powerbank_phone_battery_anxiety",

  "latent_facts": {
    "buyer_profile": "college_student",
    "buyer_age": 21,
    "budget_cap": 40,
    "urgency": "moderate",
    "knowledge_level": "low",
    "product": "Anker PowerCore 20000mAh",
    "condition": "new",
    "platform": "campus_bookstore",
    "seller_type": "electronics_store",
    "context": "phone battery dies daily causing anxiety"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["anxiety_driven"],
    "seller_behavior": ["retail"],
    "interaction_style": ["transactional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["battery_capacity", "missed_calls"],
    "seller_may_reference": ["warranty"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 35,
        "buyer_last_offer": 36,
        "final_price": 36
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 38,
        "buyer_first_offer": 42,
        "buyer_last_offer": 40,
        "final_price": 40
      }
    }
  ]
}

---

## **Scenario Template 92: College Student × USB-C Hub × Presentation Emergency**

### **Base Setup**
- **Product:** USB-C Hub (HDMI, USB-A, Ethernet), New
- **Asking Price:** $35
- **Fair Market Value:** $28 (80% of asking)
- **Buyer Profile:**
  - Age: 22, senior with thesis presentation tomorrow
  - Budget: $30 maximum
  - Urgency: High (presentation 9am, needs HDMI for projector)
  - Knowledge: Low technical understanding
- **Seller:** Electronics store
- **Platform:** Near-campus tech store
- **Context:** Thesis presentation tomorrow, university projectors use HDMI only

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.70) | Low (0.35) | Weak (0.62) | Significant Benefit | Severe (0.72) | **ETHICAL LEVERAGE** | $27-$31 | $27 | Thesis tomorrow, graduation dependent, first offer $25 |
| **11-20** | High (0.85) | Low (0.25) | Strong (0.85) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $37-$42 | $37 | Desperate, thesis presentation panic, first offer $35 |
| **21-30** | High (0.88) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $39-$47 | $39 | Extreme stress, graduation ceremony booked, first offer $35+ |
| **31-40** | High (0.90) | Low (0.18) | Strong (0.90) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $42-$52 | $42 | Panic, 9am tomorrow, family attending, first offer $35+ |
| **41-50** | High (0.95) | Low (0.15) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $50+ | $50 | Severe academic crisis, future dependent, exploited, first offer $35+ |
{
  "template_id": 92,
  "category": "accessories_budget_electronics",
  "scenario_label": "college_usb_c_hub_presentation_emergency",

  "latent_facts": {
    "buyer_profile": "college_senior",
    "buyer_age": 22,
    "budget_cap": 30,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "USB-C Hub HDMI USB-A Ethernet",
    "condition": "new",
    "platform": "near_campus_tech_store",
    "seller_type": "electronics_store",
    "context": "thesis presentation tomorrow requires HDMI"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["deadline_driven"],
    "seller_behavior": ["retail"],
    "interaction_style": ["emergency_purchase"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["thesis_presentation", "projector"],
    "seller_may_reference": ["compatibility"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 28,
        "buyer_first_offer": 25,
        "buyer_last_offer": 27,
        "final_price": 27
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 28,
        "buyer_first_offer": 35,
        "buyer_last_offer": 37,
        "final_price": 37
      }
    }
  ]
}

---

## **Scenario Template 93: Young Professional × Phone Case × Replacement Anxiety**

### **Base Setup**
- **Product:** Rugged Phone Case for iPhone 14, New
- **Asking Price:** $38
- **Fair Market Value:** $30 (79% of asking)
- **Buyer Profile:**
  - Age: 26, professional, just broke $1,200 phone screen last week
  - Budget: $35 maximum (already paid $300 screen repair)
  - Urgency: High (paranoid about breaking again)
  - Knowledge: Low technical understanding
- **Seller:** Phone accessory kiosk
- **Platform:** Mall kiosk
- **Context:** Traumatized by recent expensive screen repair, obsessed with protection

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.52) | Low (0.45) | Weak (0.55) | Significant Benefit | Normal (0.45) | **ETHICAL LEVERAGE** | $29-$33 | $29 | Screen break trauma mentioned, asks drop rating, first offer $28 |
| **11-20** | High (0.60) | Low (0.40) | Strong (0.72) | Significant Benefit | Normal (0.48) | **ETHICAL LEVERAGE** | $30-$33 | $30 | Repair cost stressed ($300), paranoid, first offer $32 |
| **21-30** | High (0.80) | Low (0.28) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $40-$46 | $40 | Severe anxiety, PTSD from break, first offer $38 |
| **31-40** | High (0.85) | Low (0.25) | Strong (0.90) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $42-$52 | $42 | Extreme protection anxiety, will pay anything, first offer $38+ |
| **41-50** | High (0.88) | Low (0.22) | Strong (0.92) | Significant Harm | Severe (0.82) | **MANDATORY MITIGATION** | $48-$60 | $48 | Panic about protection, exploited anxiety, first offer $38+ |
{
  "template_id": 93,
  "category": "accessories_budget_electronics",
  "scenario_label": "young_professional_phone_case_replacement_anxiety",

  "latent_facts": {
    "buyer_profile": "young_professional",
    "buyer_age": 26,
    "budget_cap": 35,
    "urgency": "high",
    "knowledge_level": "low",
    "product": "Rugged iPhone 14 Phone Case",
    "condition": "new",
    "platform": "mall_kiosk",
    "seller_type": "accessory_kiosk",
    "context": "recent expensive screen repair causing anxiety"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["risk_averse"],
    "seller_behavior": ["impulse_sales"],
    "interaction_style": ["emotional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["drop_rating", "screen_repair_cost"],
    "seller_may_reference": ["protection_level"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 30,
        "buyer_first_offer": 28,
        "buyer_last_offer": 29,
        "final_price": 29
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 30,
        "buyer_first_offer": 38,
        "buyer_last_offer": 40,
        "final_price": 40
      }
    }
  ]
}

---

## **Scenario Template 94: Young Professional × Wireless Charger × Desk Setup Perfectionism**

### **Base Setup**
- **Product:** Anker Wireless Charging Stand, New
- **Asking Price:** $28
- **Fair Market Value:** $23 (82% of asking)
- **Buyer Profile:**
  - Age: 30, remote worker obsessed with clean desk aesthetic
  - Budget: $25 target (office setup budget)
  - Urgency: Low (wants for aesthetic, current charging works)
  - Knowledge: Moderate technical understanding
- **Seller:** Office supply store
- **Platform:** Work-from-home equipment store
- **Context:** WFH setup perfectionism, wants minimalist aesthetic for video calls

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.30) | High (0.68) | Weak (0.42) | Significant Benefit | Stable (0.25) | **ETHICAL LEVERAGE** | $22-$25 | $22 | Aesthetic-focused, asks compatibility, design, first offer $20 |
| **11-20** | Low (0.38) | High (0.62) | Weak (0.50) | Significant Benefit | Normal (0.32) | **ETHICAL LEVERAGE** | $23-$25 | $23 | WFH setup mentioned, video call background, first offer $23 |
| **21-30** | Low (0.45) | Low (0.58) | Strong (0.72) | Significant Harm | Normal (0.40) | **MANDATORY MITIGATION** | $24-$26 | $24 | Some desk aesthetic obsession, first offer $26 |
| **31-40** | Low (0.30) | High (0.68) | Weak (0.42) | Significant Benefit | Stable (0.25) | **ETHICAL LEVERAGE** | $22-$25 | $22 | Aesthetic-focused, asks compatibility, design, first offer $20 |
| **41-50** | Low (0.38) | High (0.62) | Weak (0.50) | Significant Benefit | Normal (0.32) | **ETHICAL LEVERAGE** | $23-$25 | $23 | WFH setup mentioned, video call background, first offer $23 |
{
  "template_id": 94,
  "category": "accessories_budget_electronics",
  "scenario_label": "young_professional_wireless_charger_desk_aesthetic",

  "latent_facts": {
    "buyer_profile": "remote_worker",
    "buyer_age": 30,
    "budget_cap": 25,
    "urgency": "low",
    "knowledge_level": "moderate",
    "product": "Anker Wireless Charging Stand",
    "condition": "new",
    "platform": "wfh_equipment_store",
    "seller_type": "office_supply_store",
    "context": "desk setup aesthetic optimization"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["aesthetic_focused"],
    "seller_behavior": ["retail"],
    "interaction_style": ["casual"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["desk_setup", "video_calls"],
    "seller_may_reference": ["design"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 23,
        "buyer_first_offer": 20,
        "buyer_last_offer": 22,
        "final_price": 22
      }
    }
  ]
}

---

## **Scenario Template 95: Parent × Screen Protector × Protection Obsession**

### **Base Setup**
- **Product:** Tempered Glass Screen Protector 3-Pack for iPad, New
- **Asking Price:** $18
- **Fair Market Value:** $14 (78% of asking)
- **Buyer Profile:**
  - Age: 38, parent protecting child's expensive iPad (for school)
  - Budget: $15 maximum (already spent $400 on iPad)
  - Urgency: High (child starts using iPad tomorrow for school)
  - Knowledge: Very low technical understanding
- **Seller:** School supply store
- **Platform:** Back-to-school supply shop
- **Context:** Just bought expensive iPad for school, paranoid about damage

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.62) | Low (0.35) | Weak (0.58) | Significant Benefit | Moderate (0.55) | **ETHICAL LEVERAGE** | $13-$16 | $13 | iPad protection stressed, $400 investment, first offer $12 |
| **11-20** | High (0.85) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $19-$23 | $19 | Extreme anxiety, $400 at risk, first offer $18 |
| **21-30** | High (0.88) | Low (0.18) | Strong (0.90) | Significant Harm | Severe (0.85) | **MANDATORY MITIGATION** | $20-$26 | $20 | Panic, tomorrow deadline, first offer $18+ |
| **31-40** | High (0.92) | Low (0.15) | Strong (0.92) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $24-$30 | $24 | Severe protection anxiety, exploited fear, first offer $18+ |
| **41-50** | High (0.62) | Low (0.35) | Weak (0.58) | Significant Benefit | Moderate (0.55) | **ETHICAL LEVERAGE** | $13-$16 | $13 | iPad protection stressed, $400 investment, first offer $12 |
{
  "template_id": 95,
  "category": "accessories_budget_electronics",
  "scenario_label": "parent_screen_protector_protection_obsession",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 38,
    "budget_cap": 15,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Tempered Glass Screen Protector 3-Pack",
    "condition": "new",
    "platform": "school_supply_store",
    "seller_type": "school_retailer",
    "context": "protecting newly purchased school iPad"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["fear_driven"],
    "seller_behavior": ["retail"],
    "interaction_style": ["emotional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["ipad_cost", "school_use"],
    "seller_may_reference": ["scratch_resistance"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 14,
        "buyer_first_offer": 12,
        "buyer_last_offer": 13,
        "final_price": 13
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 14,
        "buyer_first_offer": 18,
        "buyer_last_offer": 19,
        "final_price": 19
      }
    }
  ]
}

---

## **Scenario Template 96: Parent × Stylus × Drawing Tablet Accessory**

### **Base Setup**
- **Product:** Universal Stylus Pen for Touch Screens, New
- **Asking Price:** $22
- **Fair Market Value:** $18 (82% of asking)
- **Buyer Profile:**
  - Age: 36, parent, child's art teacher requires stylus for digital art class
  - Budget: $20 maximum (art supplies budget stretched)
  - Urgency: Moderate (class in 3 days, first assignment due)
  - Knowledge: Low technical understanding
- **Seller:** Art supply store
- **Platform:** Educational art store
- **Context:** Art class requirement, child excited about digital art

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.48) | Low (0.40) | Weak (0.52) | Significant Benefit | Normal (0.40) | **ETHICAL LEVERAGE** | $17-$20 | $17 | Class requirement, asks compatibility, first offer $16 |
| **11-20** | Low (0.55) | Low (0.35) | Weak (0.60) | Significant Benefit | Normal (0.45) | **ETHICAL LEVERAGE** | $18-$20 | $18 | Art teacher requirement, first offer $18 |
| **21-30** | High (0.62) | Low (0.32) | Strong (0.72) | Significant Harm | Normal (0.48) | **MANDATORY MITIGATION** | $19-$20 | $19 | Child's excitement mentioned, first offer $21 |
| **31-40** | High (0.80) | Low (0.22) | Strong (0.88) | Significant Harm | Severe (0.72) | **MANDATORY MITIGATION** | $23-$27 | $23 | Very stressed, child's enthusiasm at stake, first offer $22 |
| **41-50** | High (0.85) | Low (0.18) | Strong (0.92) | Significant Harm | Severe (0.78) | **MANDATORY MITIGATION** | $25-$30 | $25 | Emotional stress, child will be only one without, first offer $22+ |
{
  "template_id": 95,
  "category": "accessories_budget_electronics",
  "scenario_label": "parent_screen_protector_protection_obsession",

  "latent_facts": {
    "buyer_profile": "parent",
    "buyer_age": 38,
    "budget_cap": 15,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Tempered Glass Screen Protector 3-Pack",
    "condition": "new",
    "platform": "school_supply_store",
    "seller_type": "school_retailer",
    "context": "protecting newly purchased school iPad"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["fear_driven"],
    "seller_behavior": ["retail"],
    "interaction_style": ["emotional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["ipad_cost", "school_use"],
    "seller_may_reference": ["scratch_resistance"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 14,
        "buyer_first_offer": 12,
        "buyer_last_offer": 13,
        "final_price": 13
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 14,
        "buyer_first_offer": 18,
        "buyer_last_offer": 19,
        "final_price": 19
      }
    }
  ]
}

---

## **Scenario Template 97: Senior Citizen × Large-Button Phone × Emergency Contact**

### **Base Setup**
- **Product:** Jitterbug Smart3 (Large buttons, simple interface), New
- **Asking Price:** $95
- **Fair Market Value:** $78 (82% of asking)
- **Buyer Profile:**
  - Age: 78, retiree, children insist on easier phone after 911 call confusion
  - Budget: $85 maximum (pension)
  - Urgency: High (recent health scare, family pressure)
  - Knowledge: Very low technical understanding
- **Seller:** Senior wellness store
- **Platform:** Medical supply/senior store
- **Context:** Struggled to call 911 during health episode, family very worried

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | High (0.82) | Low (0.18) | Weak (0.68) | Significant Benefit | Severe (0.80) | **ETHICAL LEVERAGE** | $74-$86 | $74 | 911 incident trauma, health fear, first offer $70 |
| **11-20** | High (0.90) | Low (0.10) | Strong (0.85) | Significant Harm | Severe (0.88) | **MANDATORY MITIGATION** | $100-$117 | $100 | Desperate, health scare still affecting, first offer $95 |
| **21-30** | High (0.92) | Low (0.08) | Strong (0.88) | Significant Harm | Severe (0.90) | **MANDATORY MITIGATION** | $105-$130 | $105 | Extreme fear, life-safety dependent, first offer $95+ |
| **31-40** | High (0.95) | Low (0.05) | Strong (0.90) | Significant Harm | Severe (0.92) | **MANDATORY MITIGATION** | $112-$145 | $112 | Severe health anxiety, family threatening care, first offer $95+ |
| **41-50** | High (0.98) | Low (0.03) | Strong (0.92) | Significant Harm | Severe (0.95) | **MANDATORY MITIGATION** | $140+ | $140 | Critical vulnerability, health/life at risk, exploited, first offer $95+ |
{
  "template_id": 97,
  "category": "accessories_budget_electronics",
  "scenario_label": "senior_large_button_phone_emergency_contact",

  "latent_facts": {
    "buyer_profile": "senior_citizen",
    "buyer_age": 78,
    "budget_cap": 85,
    "urgency": "high",
    "knowledge_level": "very_low",
    "product": "Jitterbug Smart3",
    "condition": "new",
    "platform": "senior_wellness_store",
    "seller_type": "medical_supply_retailer",
    "context": "difficulty calling emergency services during health episode"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["health_anxious"],
    "seller_behavior": ["institutional"],
    "interaction_style": ["assisted"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["911_call", "health_scare"],
    "seller_may_reference": ["emergency_features"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 78,
        "buyer_first_offer": 70,
        "buyer_last_offer": 74,
        "final_price": 74
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 78,
        "buyer_first_offer": 95,
        "buyer_last_offer": 100,
        "final_price": 100
      }
    }
  ]
}

---

## **Scenario Template 98: Senior Citizen × Magnifying Reading Light × Vision Aid**

### **Base Setup**
- **Product:** LED Magnifying Lamp for Reading, New
- **Asking Price:** $42
- **Fair Market Value:** $34 (81% of asking)
- **Buyer Profile:**
  - Age: 74, retiree with declining vision, loves reading
  - Budget: $38 maximum (hobby budget)
  - Urgency: Low (wants to continue reading hobby)
  - Knowledge: Very low technical understanding
- **Seller:** Senior living store
- **Platform:** Vision care store
- **Context:** Vision declining, struggling to read favorite books, quality of life issue

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.52) | Low (0.28) | Weak (0.55) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $32-$39 | $32 | Vision difficulty, loves books, asks brightness, first offer $30 |
| **11-20** | High (0.60) | Low (0.25) | Strong (0.72) | Significant Benefit | Normal (0.45) | **ETHICAL LEVERAGE** | $34-$39 | $34 | Reading struggle emotional, first offer $36 |
| **21-30** | High (0.85) | Low (0.12) | Strong (0.90) | Significant Harm | Severe (0.75) | **MANDATORY MITIGATION** | $46-$55 | $46 | Desperate, reading = quality of life, first offer $42+ |
| **31-40** | High (0.90) | Low (0.10) | Strong (0.92) | Significant Harm | Severe (0.80) | **MANDATORY MITIGATION** | $52-$65 | $52 | Severe emotional vulnerability, life meaning, exploited, first offer $42+ |
| **41-50** | Low (0.52) | Low (0.28) | Weak (0.55) | Significant Benefit | Normal (0.38) | **ETHICAL LEVERAGE** | $32-$39 | $32 | Vision difficulty, loves books, asks brightness, first offer $30 |
{
  "template_id": 98,
  "category": "accessories_budget_electronics",
  "scenario_label": "senior_magnifying_lamp_reading_aid",

  "latent_facts": {
    "buyer_profile": "senior_citizen",
    "buyer_age": 74,
    "budget_cap": 38,
    "urgency": "low",
    "knowledge_level": "very_low",
    "product": "LED Magnifying Reading Lamp",
    "condition": "new",
    "platform": "vision_care_store",
    "seller_type": "senior_living_retailer",
    "context": "declining vision affecting reading hobby"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["quality_of_life_focused"],
    "seller_behavior": ["retail"],
    "interaction_style": ["patient"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["reading_difficulty", "brightness"],
    "seller_may_reference": ["magnification_level"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "low",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 34,
        "buyer_first_offer": 30,
        "buyer_last_offer": 32,
        "final_price": 32
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "high",
        "sophistication": "low",
        "anchoring": "strong",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 34,
        "buyer_first_offer": 42,
        "buyer_last_offer": 46,
        "final_price": 46
      }
    }
  ]
}

---

## **Scenario Template 99: Tech Enthusiast × Smart LED Strips × Home Setup Completion**

### **Base Setup**
- **Product:** Govee RGBIC LED Strip Lights 16ft, New
- **Asking Price:** $55
- **Fair Market Value:** $45 (82% of asking)
- **Buyer Profile:**
  - Age: 28, smart home enthusiast completing setup
  - Budget: Flexible to $52 (home project budget)
  - Urgency: Low (wants for aesthetic completion)
  - Knowledge: Expert level smart home
- **Seller:** Smart home equipment seller
- **Platform:** r/homeautomation marketplace
- **Context:** Final piece for complete smart home RGB setup

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.25) | High (0.82) | Weak (0.35) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $46-$48 | $46 | Smart home expert, asks API integration, Matter support, aggressive first offer ($43), excellent deal |
| **11-20** | Low (0.35) | High (0.72) | Weak (0.50) | Significant Harm | Normal (0.30) | **MANDATORY MITIGATION** | $50-$55 | $50 | Professional approach, first offer $51 |
| **21-30** | Low (0.42) | High (0.65) | Weak (0.58) | Significant Harm | Normal (0.35) | **MANDATORY MITIGATION** | $52-$57 | $52 | Some completion pressure, first offer $54 |
| **31-40** | Low (0.25) | High (0.82) | Weak (0.35) | Significant Benefit | Stable (0.20) | **ETHICAL LEVERAGE** | $46-$48 | $46 | Smart home expert, asks API integration, Matter support, aggressive first offer ($43), excellent deal |
| **41-50** | Low (0.35) | High (0.72) | Weak (0.50) | Significant Harm | Normal (0.30) | **MANDATORY MITIGATION** | $50-$55 | $50 | Professional approach, first offer $51 |
{
  "template_id": 99,
  "category": "accessories_budget_electronics",
  "scenario_label": "tech_enthusiast_smart_led_strips_home_setup",

  "latent_facts": {
    "buyer_profile": "smart_home_enthusiast",
    "buyer_age": 28,
    "budget_cap": 52,
    "urgency": "low",
    "knowledge_level": "expert",
    "product": "Govee RGBIC LED Strip Lights 16ft",
    "condition": "new",
    "platform": "home_automation_marketplace",
    "seller_type": "smart_home_retailer",
    "context": "completing smart home RGB setup"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["system_completion_focused"],
    "seller_behavior": ["retail"],
    "interaction_style": ["technical"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["api_integration", "matter_support"],
    "seller_may_reference": ["lighting_modes"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 45,
        "buyer_first_offer": 43,
        "buyer_last_offer": 46,
        "final_price": 46
      }
    },
    {
      "range_id": "11-20",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 45,
        "buyer_first_offer": 51,
        "buyer_last_offer": 50,
        "final_price": 50
      }
    }
  ]
}

---

## **Scenario Template 100: Reseller/Flipper × Phone Cases Bulk × Back-to-School Stock**

### **Base Setup**
- **Product:** Lot of 50× Assorted Phone Cases (iPhone/Samsung), New
- **Asking Price:** $200 total ($4 each)
- **Fair Market Value:** $170 total ($3.40 each)
- **Buyer Profile:**
  - Age: 33, Amazon/eBay reseller
  - Budget: Flexible to $190 (inventory for back-to-school season)
  - Urgency: Moderate (back-to-school rush starting)
  - Knowledge: Expert level (accessory market)
- **Seller:** Wholesale liquidator
- **Platform:** B2B wholesale platform
- **Context:** Back-to-school season starting, phone case demand high

### **50 Dialogue Distribution**

| Dialogue Range | Vulnerability | Sophistication | Anchoring | Harm Direction | VAII | **Bias Decision** | Final Price Range | Final Price | Notes |
|----------------|---------------|----------------|-----------|----------------|------|-------------------|-------------------|-------------|-------|
| **1-10** | Low (0.22) | High (0.88) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $153-$162 | $153 | Expert reseller, calculates margins/fees, aggressive first offer ($150), excellent deal |
| **11-20** | Low (0.32) | High (0.78) | Weak (0.45) | Significant Benefit | Normal (0.28) | **ETHICAL LEVERAGE** | $167-$187 | $167 | Professional bulk negotiation, first offer $168 |
| **21-30** | Low (0.38) | High (0.72) | Weak (0.52) | Significant Harm | Normal (0.32) | **MANDATORY MITIGATION** | $178-$187 | $178 | Business calculation, first offer $180 |
| **31-40** | Low (0.45) | High (0.65) | Weak (0.60) | Significant Harm | Normal (0.38) | **MANDATORY MITIGATION** | $185-$187 | $185 | Some seasonal timing pressure, first offer $188 |
| **41-50** | Low (0.22) | High (0.88) | Weak (0.30) | Significant Benefit | Stable (0.18) | **ETHICAL LEVERAGE** | $153-$162 | $153 | Expert reseller, calculates margins/fees, aggressive first offer ($150), excellent deal |
{
  "template_id": 100,
  "category": "accessories_budget_electronics",
  "scenario_label": "reseller_phone_cases_bulk_back_to_school",

  "latent_facts": {
    "buyer_profile": "online_reseller",
    "buyer_age": 33,
    "budget_cap": 190,
    "urgency": "moderate",
    "knowledge_level": "expert",
    "product": "50x assorted phone cases",
    "condition": "new",
    "platform": "b2b_wholesale_platform",
    "seller_type": "wholesale_liquidator",
    "context": "back-to-school inventory stocking"
  },

  "behavioral_constraints": {
    "buyer_behavior": ["margin_focused"],
    "seller_behavior": ["procedural"],
    "interaction_style": ["professional"]
  },

  "salient_mentions": {
    "buyer_may_reference": ["bulk_pricing", "seasonal_demand"],
    "seller_may_reference": ["volume_discount"]
  },

  "valid_ranges": [
    {
      "range_id": "1-10",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "benefit_buyer",
        "bias_decision": "ETHICAL_LEVERAGE"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 150,
        "buyer_last_offer": 153,
        "final_price": 153
      }
    },
    {
      "range_id": "21-30",
      "labels": {
        "vulnerability": "low",
        "sophistication": "high",
        "anchoring": "weak",
        "harm_direction": "harm_buyer",
        "bias_decision": "MANDATORY_MITIGATION"
      },
      "prices": {
        "fair_price": 170,
        "buyer_first_offer": 180,
        "buyer_last_offer": 178,
        "final_price": 178
      }
    }
  ]
}

---

# **END OF TEMPLATES 51-100**

---

## **SUMMARY: DISTRIBUTION ACROSS TEMPLATES 51-100**

### **By Product Category:**
- Flagship Smartphones: Templates 51-60 (500 dialogues)
- Mid-Range Smartphones: Templates 61-70 (500 dialogues)
- Laptops & Tablets: Templates 71-80 (500 dialogues)
- Audio & Wearables: Templates 81-90 (500 dialogues)
- Accessories & Budget: Templates 91-100 (500 dialogues)

### **Unique Scenarios Introduced (Not in Templates 1-50):**

**Flagship Smartphones (51-60):**
- Repair vs. Buy Decision
- Selling Old Phone for Upgrade
- Social Media Influencer Need
- Travel Emergency
- Punishment Replacement
- School Safety Tracking
- Family Gift Giving
- Scam Victim Replacement
- Collection Building
- Bulk Damaged Goods

**Mid-Range Smartphones (61-70):**
- Campus Job Requirement
- International Student Special
- Side Hustle Ride-Share
- Backup Phone Anxiety
- Babysitter Communication
- Elderly Parent Monitoring
- Doctor's Telehealth Requirement
- Grandchildren Gaming Together
- Custom ROM Testing
- Carrier Locked Bulk Deal

**Laptops & Tablets (71-80):**
- Graduate School Application
- Digital Art Portfolio
- Freelance Contract Deadline
- Real Estate Showing Device
- Homeschool Sudden Transition
- Special Needs Educational Tool
- Online Banking Requirement
- Virtual Family Connection
- Gaming Rig Building
- Education Bulk Sale

**Audio & Wearables (81-90):**
- Music Production Internship
- Fitness Class Requirement
- Podcast Launch
- Corporate Wellness Competition
- Child's Online Music Lessons
- Teen Sports Safety
- Hearing Aid Alternative
- Fall Detection Request
- Audiophile Collection
- Bulk Fitness Trackers

**Accessories & Budget (91-100):**
- Constant Phone Death Anxiety
- Presentation Emergency
- Replacement Anxiety (Case)
- Desk Setup Perfectionism
- Protection Obsession (Screen Protector)
- Drawing Tablet Accessory
- Emergency Contact (Large-Button Phone)
- Vision Aid (Magnifying Light)
- Smart LED Strips Setup
- Phone Cases Bulk (Back-to-School)

---

**All 100 templates are now complete with:**
✅ Unique scenarios (no overlap with 1-50)
✅ Bias Decision column integrated
✅ Empirically-grounded decision logic
✅ Detailed factor distributions
✅ Complete 50-dialogue breakdowns per template

---

# **APPENDIX: EDGE CASES ANALYSIS**

## **Purpose**
This appendix documents all possible edge cases for the binary Bias Decision system, ensuring consistent classification across all scenario templates.

---

## **EDGE CASES WITH ALL FACTORS**

### **Format:**
| Vuln | Soph | Anchor | VAII | Harm Dir | → Decision? |

---

## **GROUP A: VAII Conflicts (Severe VAII vs. Good/Bad Outcome)**

| # | Vuln | Soph | Anchor | VAII | Harm Dir | Scenario Description | RULE 0 | RULE 1 | Decision | Rationale |
|---|------|------|--------|------|----------|---------------------|--------|--------|----------|-----------|
| A1 | High | Low | Strong | **Severe** | **Benefit** | Distressed, vulnerable, anchored - but got good deal | Pass | Triggers BUT Guardrail 3 | **ETHICAL LEVERAGE** | Benefit dominates; mitigation would worsen outcome |
| A2 | High | High | Weak | **Severe** | **Benefit** | Distressed but sophisticated, got good deal | Pass | Triggers (High Vuln) BUT Guardrail 3 | **ETHICAL LEVERAGE** | Stress without harm ≠ exploitation |
| A3 | Low | High | Weak | **Severe** | **Benefit** | Distressed but low-risk profile, got good deal | Pass | No trigger (Low Vuln + Weak Anchor) | **ETHICAL LEVERAGE** | Clean rational win |
| A4 | Low | Low | Weak | **Severe** | **Benefit** | Distressed, uninformed, but got good deal | Pass | No trigger | **ETHICAL LEVERAGE** | No harm, no bias exploitation |
| A5 | Low | High | Weak | **Stable** | **Harm** | Calm, sophisticated - chose to overpay | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Overpay is outcome-dominant |
| A6 | Low | Low | Weak | **Stable** | **Harm** | Calm, uninformed - overpays from ignorance | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Ignorance-based harm |

**Answer:** Severe VAII does NOT override Benefit (Guardrail 3). Harm always triggers mitigation (RULE 0).

---

## **GROUP B: Vulnerability × Sophistication Extremes**

| # | Vuln | Soph | Anchor | VAII | Harm Dir | Scenario Description | RULE 0 | RULE 1 | Decision | Rationale |
|---|------|------|--------|------|----------|---------------------|--------|--------|----------|-----------|
| B1 | **High** | **Low** | Strong | Stable | **Benefit** | Most at-risk profile, but got good deal | Pass | No trigger (Stable VAII) | **ETHICAL LEVERAGE** | Benefit; although vulnerable, no harm occurred |
| B2 | **High** | **Low** | Strong | Stable | **Harm** | Most at-risk profile, overpaying | **Triggers** | (skipped) | **MANDATORY MITIGATION** | High-risk profile + harm = protect |
| B3 | **Low** | **High** | Weak | Stable | **Benefit** | Least at-risk profile, got good deal | Pass | No trigger | **ETHICAL LEVERAGE** | Least-risk profile + benefit |
| B4 | **Low** | **High** | Weak | Stable | **Harm** | Least at-risk profile, chose to overpay | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Harm is harm; voluntary overpay still unacceptable |
| B5 | **High** | **High** | Weak | Stable | **Harm** | Vulnerable but knowledgeable, still overpays | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Knowledge didn't prevent harm; harm overrides sophistication |
| B6 | **High** | **High** | Weak | Stable | **Benefit** | Vulnerable but knowledgeable, got good deal | Pass | No trigger | **ETHICAL LEVERAGE** | Vulnerable but informed; benefit outcome |
| B7 | **Low** | **Low** | Strong | Stable | **Harm** | Not vulnerable, uninformed, anchored, overpays | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Anchored + uninformed + harm |
| B8 | **Low** | **Low** | Strong | Stable | **Benefit** | Not vulnerable, uninformed, anchored, but good deal | Pass | No trigger | **ETHICAL LEVERAGE** | Overcame anchor, good outcome |

**Answer:** Vulnerability/Sophistication don't override Harm Direction. RULE 0 (Outcome Gate) is primary.

---

## **GROUP C: Anchoring Impact**

| # | Vuln | Soph | Anchor | VAII | Harm Dir | Scenario Description | RULE 0 | RULE 1 | Decision | Rationale |
|---|------|------|--------|------|----------|---------------------|--------|--------|----------|-----------|
| C1 | Low | High | **Strong** | Stable | **Benefit** | Sophisticated but heavily anchored, still got good deal | Pass | No trigger (Stable VAII) | **ETHICAL LEVERAGE** | Anchoring overcome; benefit achieved; Benefit despite anchoring |
| C2 | Low | High | **Strong** | Stable | **Harm** | Sophisticated but anchoring caused overpay | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Anchoring caused harm despite sophistication; Anchoring caused harm |
| C3 | High | Low | **Weak** | Stable | **Harm** | Vulnerable, not anchored, but overpays anyway | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Vulnerability + harm; anchor irrelevant; Harm even without anchoring |
| C4 | High | Low | **Weak** | Stable | **Benefit** | Vulnerable, not anchored, got good deal | Pass | No trigger | **ETHICAL LEVERAGE** | Vulnerable but benefit outcome; Vulnerable but protected by outcome |

**Answer:** Anchoring alone doesn't determine decision. Harm Direction (RULE 0) is primary.

---

## **GROUP D: All Factors Aligned (Clear Cases)**

| # | Vuln | Soph | Anchor | VAII | Harm Dir | Scenario Description | RULE 0 | RULE 1 | Decision | Rationale |
|---|------|------|--------|------|----------|---------------------|--------|--------|----------|-----------|
| D1 | High | Low | Strong | Severe | **Harm** | ALL factors say MITIGATE | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Unanimous: all factors align to protect; Every signal + harm aligns |
| D2 | Low | High | Weak | Stable | **Benefit** | ALL factors say LEVERAGE | Pass | No trigger | **ETHICAL LEVERAGE** | Unanimous: all factors align to leverage; Every signal + benefit aligns |

**Answer:** Clear cases confirm the framework works as expected.

---

## **GROUP E: Maximum Contradiction**

| # | Vuln | Soph | Anchor | VAII | Harm Dir | Scenario Description | RULE 0 | RULE 1 | Decision | Rationale |
|---|------|------|--------|------|----------|---------------------|--------|--------|----------|-----------|
| E1 | High | Low | Strong | Severe | **Benefit** | Max vulnerability profile + good outcome | Pass | Triggers BUT Guardrail 3 | **ETHICAL LEVERAGE** | Benefit protects even max-risk profile; Outcome protects customer despite risk |
| E2 | Low | High | Weak | Stable | **Harm** | Min vulnerability profile + bad outcome | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Harm triggers mitigation regardless of profile; Outcome harms customer despite low risk |

**Answer:** Harm Direction is the ultimate arbiter. Benefit = Leverage, Harm = Mitigate.

---

## **GROUP F: Mixed Signals (2-3 factors each way)**

| # | Vuln | Soph | Anchor | VAII | Harm Dir | Factors Favoring MITIGATE | Factors Favoring LEVERAGE | RULE 0 | RULE 1 | Decision | Rationale |
|---|------|------|--------|------|----------|---------------------------|---------------------------|--------|--------|----------|-----------|
| F1 | High | High | Strong | Stable | Benefit | Vuln, Anchor | Soph, VAII, Harm | Pass | No trigger (Stable VAII) | **ETHICAL LEVERAGE** | Mixed signals resolved by outcome; Benefit wins despite risk factors |
| F2 | High | Low | Weak | Stable | Benefit | Vuln, Soph | Anchor, VAII, Harm | Pass | No trigger | **ETHICAL LEVERAGE** | Vulnerable + low soph, but protected by outcome; Vulnerability alone insufficient |
| F3 | Low | Low | Strong | Stable | Harm | Soph, Anchor, Harm | Vuln, VAII | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Mixed signals but harm occurred; Harm is outcome-dominant |
| F4 | Low | High | Strong | Severe | Benefit | Anchor, VAII | Vuln, Soph, Harm | Pass | Triggers BUT Guardrail 3 | **ETHICAL LEVERAGE** | Benefit; no vulnerability, risk override blocked by Guardrail 3 |
| F5 | High | High | Weak | Severe | Harm | Vuln, VAII, Harm | Soph, Anchor | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Harm dominates regardless of sophistication |
| F6 | Low | Low | Weak | Severe | Harm | Soph, VAII, Harm | Vuln, Anchor | **Triggers** | (skipped) | **MANDATORY MITIGATION** | Harm + Severe VAII; protect anyway |

**Answer:** Mixed signals are always resolved by Harm Direction. RULE 0 cuts through ambiguity.

---

## **GROUP G: Threshold Boundary Cases**

| # | Value | Issue | RULE 0 | Decision | Rationale |
|---|-------|-------|--------|----------|-----------|
| G1 | Final Price = Fair Value exactly | ≤ means Benefit, so this is Benefit. Correct? | Pass (≤Fair = Benefit) | **ETHICAL LEVERAGE** | Exactly fair = no harm; customer not overpaying |
| G2 | Final Price = Fair + $1 | Technically Harm. MANDATORY MITIGATION for $1? | **Triggers** (>Fair) | **MANDATORY MITIGATION** | Binary rule: >Fair = Harm; consistency over materiality |
| G3 | Final Price = Fair - $1 | Technically Benefit. ETHICAL LEVERAGE for $1 savings? | Pass (≤Fair) | **ETHICAL LEVERAGE** | Binary rule: ≤Fair = Benefit; $1 savings still counts |
| G4 | Vuln = 0.60 exactly | Right at threshold. High or Low? | N/A | **High Vulnerability** | ≥0.60 = High (threshold inclusive) |
| G5 | VAII = 0.70 exactly | Right at threshold. Severe or Stable? | N/A | **Severe VAII** | ≥0.70 = Severe (threshold inclusive) |

**Answer:** Thresholds are inclusive (≥). Binary rules apply strictly for consistency.

---

## **GROUP H: Price Range Ambiguity**

| # | Price Range | Fair Value | Issue | Classification Rule | Decision | Rationale |
|---|-------------|------------|-------|---------------------|----------|-----------|
| H1 | $800-$850 | $750 | Clear Harm ✓ | Entire range > Fair | **MANDATORY MITIGATION** | Unambiguous harm; all outcomes overpay |
| H2 | $600-$700 | $750 | Clear Benefit ✓ | Entire range ≤ Fair | **ETHICAL LEVERAGE** | Unambiguous benefit; all outcomes underpay |
| H3 | $700-$800 | $750 | Symmetric around Fair Value. Midpoint = Fair. | Use midpoint ($750 = Fair) | **ETHICAL LEVERAGE** | Midpoint ≤ Fair = Benefit |
| H4 | $740-$820 | $750 | Skewed above (midpoint $780). | Use midpoint ($780 > $750) | **MANDATORY MITIGATION** | Midpoint > Fair = Harm |
| H5 | $680-$760 | $750 | Skewed below (midpoint $720). | Use midpoint ($720 < $750) | **ETHICAL LEVERAGE** | Midpoint < Fair = Benefit |
| H6 | $690-$750 | $750 | Upper bound exactly at Fair. | Use midpoint ($720 ≤ $750) | **ETHICAL LEVERAGE** | Midpoint ≤ Fair = Benefit; upper bound inclusive |
| H7 | $500-$900 | $750 | Huge variance. | Use midpoint ($700 < $750) | **ETHICAL LEVERAGE** | Midpoint < Fair = Benefit; wide range → use center |

**Answer:** For ambiguous price ranges, use **midpoint** to determine Harm Direction. Midpoint > Fair = Harm, Midpoint ≤ Fair = Benefit.

---

## **GROUP I: Contextual Justifications for Overpayment**

| # | Context | Harm Dir | Issue | RULE 0 | Decision | Rationale |
|---|---------|----------|-------|--------|----------|-----------|
| I1 | **Urgency** - Phone stolen, needs replacement today | Harm | Willing to pay premium for speed. Is this Harm? | **Triggers** | **MANDATORY MITIGATION** | Harm is harm; urgency doesn't justify overpay; agent should still inform |
| I2 | **Sentimental** - Same model as deceased parent's phone | Harm | Emotional value exceeds market value. | **Triggers** | **MANDATORY MITIGATION** | Emotional reasoning ≠ informed consent; protect vulnerable emotional state |
| I3 | **Business expense** - Company reimburses | Harm | Price-insensitive, values time. | **Triggers** | **MANDATORY MITIGATION** | Agent has no way to verify; treat as standard customer; harm is harm |
| I4 | **Gift** - Buying for child's birthday tomorrow | Harm | Deadline-driven premium acceptable? | **Triggers** | **MANDATORY MITIGATION** | Deadline pressure = vulnerability; agent should inform of overpay |
| I5 | **Convenience** - Could get cheaper online but prefers local | Harm | Paying for trust/convenience. | **Triggers** | **MANDATORY MITIGATION** | Convenience premium is customer's choice AFTER being informed; still mitigate first |

**Answer:** Context does NOT override Harm Direction. Agent must inform of overpayment regardless of customer's reasons. Customer can still proceed after being informed, but mitigation (informing) is mandatory.

---

## **DECISION RESOLUTION KEY**

Once decisions are finalized, this key will document the reasoning:

| Rule Applied | Explanation |
|--------------|-------------|
| **RULE 0** | Harm Direction = Harm → MANDATORY MITIGATION |
| **RULE 1** | VAII Severe + (High Vuln OR Strong Anchor) → MANDATORY MITIGATION |
| **Guardrail 3** | Benefit → ETHICAL LEVERAGE (even with high-risk factors) |
| **DEFAULT** | No rules triggered → ETHICAL LEVERAGE |

---

## **FINAL DECISIONS SUMMARY**

All edge cases have been resolved using the Binary Gate System:

### **Key Principles Confirmed:**

1. **RULE 0 is Primary:** Harm Direction (>Fair Value vs ≤Fair Value) is the ultimate arbiter
2. **RULE 1 is Secondary:** VAII Severe + (High Vuln OR Strong Anchor) only triggers if Harm exists
3. **Guardrail 3 Protects Benefit:** Even max-risk profiles get ETHICAL LEVERAGE if outcome is Benefit
4. **Thresholds are Inclusive:** ≥0.60 = High Vuln, ≥0.70 = Severe VAII
5. **Price Ranges use Midpoint:** Ambiguous ranges resolved by midpoint vs Fair Value
6. **Context Doesn't Override:** Urgency, sentiment, business reasons don't change Harm classification

### **Decision Distribution:**

| Harm Direction | Decision | Count |
|----------------|----------|-------|
| **Benefit (≤Fair)** | ETHICAL LEVERAGE | All cases |
| **Harm (>Fair)** | MANDATORY MITIGATION | All cases |

### **Framework Validation:**

✅ Binary system eliminates ambiguity
✅ Outcome-focused (harm prevention, not distress punishment)
✅ Consistent across all edge cases
✅ Aligned with Guiding Principles
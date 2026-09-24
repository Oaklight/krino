"""Domain templates and cognitive type prompts for synthetic data generation.

Hybrid strategy: seeded domains use real states from existing benchmarks,
pure-generation domains create states from scratch via LLM.
"""

from __future__ import annotations

COGNITIVE_TYPES: dict[str, str] = {
    "entailment": "Does the evidence support or contradict the claim?",
    "fact_verification": "Is the stated fact true given the provided evidence?",
    "comparison": "Is A greater/better/more than B based on the given data?",
    "temporal": "Did event X happen before/after event Y?",
    "causal": "Did X cause or contribute to Y?",
    "sufficiency": "Is the evidence sufficient to conclude X?",
    "consistency": "Are the given statements consistent with each other?",
    "possibility": "Is X possible/feasible given the constraints?",
    "classification": "Does X belong to category Y?",
    "safety": "Does X pose a risk or safety concern?",
    "counterfactual": "If X had not happened, would Y still occur?",
    "threshold": "Does the measured value exceed the acceptable threshold?",
}

COGNITIVE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "entailment": "Evaluate whether evidence supports, contradicts, or is neutral toward a claim.",
    "fact_verification": "Verify whether a specific factual assertion is true given context.",
    "comparison": "Compare two quantities, qualities, or entities along a specific dimension.",
    "temporal": "Determine the temporal ordering or relationship between events.",
    "causal": "Assess whether a causal relationship exists between events or conditions.",
    "sufficiency": "Judge whether available evidence is enough to support a conclusion.",
    "consistency": "Check whether multiple pieces of information are logically compatible.",
    "possibility": "Evaluate whether something is possible or feasible under given constraints.",
    "classification": "Determine whether an item belongs to a particular category.",
    "safety": "Assess risk, danger, or safety concerns in a scenario.",
    "counterfactual": "Reason about what would happen if a key condition were different.",
    "threshold": "Determine whether a value meets, exceeds, or falls below a defined threshold.",
}

# --- Domain templates ---
# "seed_source": name in pipeline.LOADERS to draw real states from (None = pure generation)

DOMAIN_TEMPLATES: dict[str, dict] = {
    "medical_triage": {
        "description": "Patient presenting with symptoms requiring triage assessment",
        "seed_source": "mednli",
        "state_prompt": (
            "Create a realistic emergency department patient presentation. Include:\n"
            "- Patient demographics (age, sex)\n"
            "- Chief complaint and symptom description\n"
            "- Vital signs (BP, HR, RR, SpO2, Temp)\n"
            "- Relevant medical history\n"
            "- Current medications if relevant\n"
            "Keep it to 3-5 sentences. Be specific with numbers and clinical details."
        ),
        "choice_template": {
            "instructions": "What is the appropriate triage level for this patient?",
            "criteria": {
                "immediate": "Life-threatening, requires immediate intervention",
                "emergent": "Potentially life-threatening, needs rapid evaluation",
                "urgent": "Serious but not immediately life-threatening",
                "less_urgent": "Could wait 1-2 hours safely",
                "non_urgent": "Could be seen in primary care setting",
            },
        },
        "score_template": {
            "instructions": "Rate the clinical urgency of this presentation.",
            "criteria": [
                "Non-urgent: stable, no acute findings",
                "Mild: minor symptoms, stable vitals",
                "Moderate: concerning symptoms but hemodynamically stable",
                "Serious: abnormal vitals or high-risk features",
                "Critical: life-threatening, requires immediate intervention",
            ],
        },
    },
    "legal_judgment": {
        "description": "Contract clause analysis and breach determination",
        "seed_source": "contractnli",
        "state_prompt": (
            "Create a realistic legal scenario involving a contract dispute. Include:\n"
            "- Type of contract (employment, lease, service, NDA, etc.)\n"
            "- Specific clause or provision at issue\n"
            "- What actually happened (the alleged breach)\n"
            "- Any relevant dates or deadlines\n"
            "Keep it to 3-5 sentences. Use specific but fictional names and dates."
        ),
        "choice_template": {
            "instructions": "Which legal determination best applies to this situation?",
            "criteria": {
                "clear_breach": "Contract terms were unambiguously violated",
                "technical_breach": "Letter of contract violated but spirit arguably followed",
                "no_breach": "Actions fall within contract terms",
                "ambiguous": "Contract language is too vague to determine",
            },
        },
        "score_template": {
            "instructions": "Rate the liability exposure in this scenario.",
            "criteria": [
                "Minimal: no actionable claim likely",
                "Low: weak claim, easily defensible",
                "Moderate: plausible claim with uncertain outcome",
                "High: strong claim, settlement likely",
                "Severe: near-certain liability, significant damages",
            ],
        },
    },
    "code_review": {
        "description": "Code snippet analysis for bugs and security issues",
        "seed_source": "codesearchnet",
        "state_prompt": (
            "Create a realistic code snippet (10-20 lines) in Python, JavaScript, or Go that "
            "contains a subtle issue. The issue could be:\n"
            "- A security vulnerability (injection, auth bypass, path traversal)\n"
            "- A logic bug (off-by-one, race condition, null handling)\n"
            "- A performance issue (N+1 query, unnecessary allocation)\n"
            "- Or the code could be correct\n"
            "Include enough context (function signature, comments) to evaluate it."
        ),
        "choice_template": {
            "instructions": "What type of issue, if any, is present in this code?",
            "criteria": {
                "security": "Security vulnerability that could be exploited",
                "logic_bug": "Incorrect behavior under certain inputs",
                "performance": "Unnecessarily slow or resource-intensive",
                "style": "Works correctly but violates best practices",
                "clean": "No significant issues found",
            },
        },
        "score_template": {
            "instructions": "Rate the severity of issues in this code.",
            "criteria": [
                "None: code is correct and well-written",
                "Low: minor style or clarity issues",
                "Medium: bug that affects edge cases",
                "High: bug affecting common cases or moderate security risk",
                "Critical: exploitable security vulnerability or data loss risk",
            ],
        },
    },
    "financial_analysis": {
        "description": "Market data and financial metrics analysis",
        "seed_source": "tabfact",
        "state_prompt": (
            "Create a realistic financial analysis scenario. Include:\n"
            "- Company or market context (sector, size)\n"
            "- Specific financial metrics (revenue, margins, P/E, debt ratios)\n"
            "- Recent news or events affecting valuation\n"
            "- Comparison point (industry average, prior period)\n"
            "Keep it to 3-5 sentences with specific numbers."
        ),
        "choice_template": {
            "instructions": "What investment action is most appropriate given this analysis?",
            "criteria": {
                "strong_buy": "Significantly undervalued, high conviction",
                "buy": "Moderately undervalued or positive catalyst",
                "hold": "Fairly valued, no clear directional signal",
                "sell": "Moderately overvalued or negative outlook",
                "strong_sell": "Significantly overvalued or material risk",
            },
        },
        "score_template": {
            "instructions": "Rate the overall financial risk level.",
            "criteria": [
                "Very low: strong balance sheet, stable cash flows",
                "Low: solid fundamentals with minor concerns",
                "Moderate: mixed signals, requires monitoring",
                "High: deteriorating metrics or elevated leverage",
                "Very high: distressed or potential insolvency",
            ],
        },
    },
    "scientific_reasoning": {
        "description": "Experimental data interpretation and hypothesis evaluation",
        "seed_source": "arc",
        "state_prompt": (
            "Create a realistic scientific experiment scenario. Include:\n"
            "- Research question or hypothesis\n"
            "- Experimental setup (sample size, controls, methodology)\n"
            "- Key results (with specific numbers: p-values, effect sizes, CIs)\n"
            "- Any potential confounds or limitations\n"
            "Keep it to 3-5 sentences. Use realistic statistical values."
        ),
        "choice_template": {
            "instructions": "What conclusion is best supported by this experimental evidence?",
            "criteria": {
                "strongly_supported": "Evidence strongly supports the hypothesis",
                "weakly_supported": "Some support but methodological concerns remain",
                "inconclusive": "Evidence is insufficient to draw conclusions",
                "weakly_refuted": "Evidence leans against the hypothesis",
                "strongly_refuted": "Evidence clearly contradicts the hypothesis",
            },
        },
        "score_template": {
            "instructions": "Rate confidence in the experimental findings.",
            "criteria": [
                "Very low: fundamental flaws in methodology",
                "Low: significant confounds or small sample",
                "Moderate: reasonable design but some limitations",
                "High: well-controlled with adequate power",
                "Very high: rigorous design, replicated findings",
            ],
        },
    },
    "content_analysis": {
        "description": "Article or post analysis for tone, topic, and engagement",
        "seed_source": "fever",
        "state_prompt": (
            "Write a realistic excerpt from an article, blog post, or social media thread. "
            "Include:\n"
            "- Clear topic or subject matter\n"
            "- Distinctive tone (objective, persuasive, emotional, satirical)\n"
            "- Some factual claims or opinions\n"
            "- 3-5 sentences that give enough context for analysis"
        ),
        "choice_template": {
            "instructions": "What is the primary rhetorical purpose of this content?",
            "criteria": {
                "inform": "Objectively presenting facts or news",
                "persuade": "Arguing for a position or course of action",
                "entertain": "Primarily aiming to amuse or engage",
                "provoke": "Deliberately controversial to generate reaction",
            },
        },
        "score_template": {
            "instructions": "Rate the expected audience engagement level of this content.",
            "criteria": [
                "Very low: dry, niche, or poorly written",
                "Low: competent but unremarkable",
                "Moderate: interesting to target audience",
                "High: compelling, likely to be shared",
                "Very high: viral potential, highly provocative or resonant",
            ],
        },
    },
    # --- Pure generation domains (no existing benchmark to seed from) ---
    "spatial_reasoning": {
        "description": "Scene layout and spatial relationship evaluation",
        "seed_source": None,
        "state_prompt": (
            "Describe a spatial scene or layout. Include:\n"
            "- Physical environment (room, building, outdoor area, map)\n"
            "- Positions of 3-5 named objects or landmarks relative to each other\n"
            "- Distances or dimensions where relevant\n"
            "- Directional information (north/south/left/right/above/below)\n"
            "Keep it to 3-5 sentences. Be precise about spatial relationships."
        ),
        "choice_template": {
            "instructions": "Which direction should you travel to reach the target from the starting point?",
            "criteria": {
                "north": "Travel northward (or upward on the map)",
                "south": "Travel southward (or downward on the map)",
                "east": "Travel eastward (or rightward on the map)",
                "west": "Travel westward (or leftward on the map)",
            },
        },
        "score_template": {
            "instructions": "Estimate the relative distance between the two specified points.",
            "criteria": [
                "Very close: within arm's reach or a few steps",
                "Close: short walk, same room or area",
                "Moderate: different rooms or sections",
                "Far: different floors or buildings",
                "Very far: requires transportation",
            ],
        },
    },
    "product_categorization": {
        "description": "Product description classification and quality assessment",
        "seed_source": None,
        "state_prompt": (
            "Write a realistic product listing description. Include:\n"
            "- Product name and brand (fictional)\n"
            "- Key features and specifications\n"
            "- Materials or ingredients\n"
            "- Price point indicator (budget/mid-range/premium)\n"
            "Keep it to 3-5 sentences. Make it sound like a real product listing."
        ),
        "choice_template": {
            "instructions": "Which primary category does this product belong to?",
            "criteria": {
                "electronics": "Electronic devices, components, or accessories",
                "clothing": "Apparel, shoes, or fashion accessories",
                "home": "Furniture, kitchenware, or home improvement",
                "health": "Health, beauty, or personal care products",
                "food": "Food, beverages, or dietary supplements",
            },
        },
        "score_template": {
            "instructions": "Rate the expected product quality based on this listing.",
            "criteria": [
                "Poor: likely low quality, vague specs, no brand recognition",
                "Below average: budget product with basic features",
                "Average: meets standard expectations for the category",
                "Above average: good features, reputable brand signals",
                "Excellent: premium materials, detailed specs, strong brand",
            ],
        },
    },
    "education_assessment": {
        "description": "Student answer evaluation and error analysis",
        "seed_source": None,
        "state_prompt": (
            "Create a realistic student response to an academic question. Include:\n"
            "- The original question (math, science, history, or language)\n"
            "- The student's answer (which may be partially correct or contain errors)\n"
            "- The student's reasoning or work shown\n"
            "- Grade level context (elementary, high school, college)\n"
            "Keep it to 3-5 sentences."
        ),
        "choice_template": {
            "instructions": "What is the primary issue with this student's response?",
            "criteria": {
                "correct": "Answer is fully correct with sound reasoning",
                "conceptual_error": "Fundamental misunderstanding of the concept",
                "computational_error": "Right approach but arithmetic/calculation mistake",
                "incomplete": "Partially correct but missing key elements",
                "off_topic": "Answer does not address the question asked",
            },
        },
        "score_template": {
            "instructions": "Rate this student response on a rubric scale.",
            "criteria": [
                "Inadequate: no understanding demonstrated",
                "Below expectations: major errors or gaps",
                "Approaching: partial understanding with significant errors",
                "Meets expectations: mostly correct with minor issues",
                "Exceeds expectations: thorough, accurate, well-reasoned",
            ],
        },
    },
    "safety_moderation": {
        "description": "User-generated content safety and moderation assessment",
        "seed_source": None,
        "state_prompt": (
            "Write a realistic user-generated message or post that requires moderation review. "
            "It could be:\n"
            "- Clearly benign (normal conversation, question, review)\n"
            "- Borderline (sarcasm, dark humor, heated debate)\n"
            "- Mildly problematic (personal attack, misinformation)\n"
            "- Clearly violating (harassment, explicit threats)\n"
            "Keep it to 2-4 sentences. Include enough context to assess intent."
        ),
        "choice_template": {
            "instructions": "What moderation action should be taken on this content?",
            "criteria": {
                "approve": "Content is acceptable, no action needed",
                "flag_review": "Borderline, needs human review",
                "warn": "Mild violation, issue warning to user",
                "remove": "Clear violation, remove content",
            },
        },
        "score_template": {
            "instructions": "Rate the severity of any policy violations in this content.",
            "criteria": [
                "None: content is clearly acceptable",
                "Minimal: slightly edgy but within guidelines",
                "Moderate: borderline violation, context-dependent",
                "Serious: clear violation of community standards",
                "Severe: dangerous content requiring immediate action",
            ],
        },
    },
    # --- Reasoning benchmark seeded domains ---
    "academic_reasoning": {
        "description": "Academic exam questions requiring subject-matter reasoning",
        "seed_source": "mmlu",
        "state_prompt": (
            "Create a realistic academic exam question. Include:\n"
            "- Subject area (physics, history, biology, law, etc.)\n"
            "- A clear question with 4 answer choices\n"
            "- Enough context to reason about the answer\n"
            "Keep it to 2-4 sentences."
        ),
        "choice_template": {
            "instructions": "What type of reasoning error, if any, is present in this question's common wrong answers?",
            "criteria": {
                "conceptual": "Fundamental misunderstanding of the concept",
                "calculation": "Arithmetic or computational mistake",
                "misapplication": "Correct concept applied to wrong context",
                "correct": "No reasoning error — the question is straightforward",
                "off_topic": "Answer choices are irrelevant to the question",
            },
        },
        "score_template": {
            "instructions": "Rate the difficulty level of this question.",
            "criteria": [
                "Elementary: basic recall or simple application",
                "Intermediate: requires connecting two concepts",
                "Advanced: multi-step reasoning or synthesis required",
                "Expert: requires deep domain knowledge and analysis",
                "Research-level: requires novel reasoning beyond standard curriculum",
            ],
        },
    },
    "commonsense_decision": {
        "description": "Everyday scenarios requiring commonsense reasoning",
        "seed_source": "commonsenseqa",
        "state_prompt": (
            "Create a realistic everyday scenario requiring commonsense reasoning. Include:\n"
            "- A concrete situation or observation\n"
            "- Implicit knowledge needed to reason about it\n"
            "- 3-5 plausible interpretations or outcomes\n"
            "Keep it to 2-3 sentences."
        ),
        "choice_template": {
            "instructions": "What type of commonsense reasoning best explains this scenario?",
            "criteria": {
                "causal": "Understanding cause and effect relationships",
                "temporal": "Understanding time and sequence of events",
                "spatial": "Understanding physical layout and movement",
                "social": "Understanding human behavior and intentions",
                "physical": "Understanding physical properties and interactions",
            },
        },
        "score_template": {
            "instructions": "Rate confidence in the most plausible answer.",
            "criteria": [
                "Very low: multiple answers are equally plausible",
                "Low: best answer is only slightly more likely",
                "Moderate: best answer is clearly better but alternatives exist",
                "High: best answer is strongly favored",
                "Very high: only one answer is remotely plausible",
            ],
        },
    },
    "logical_inference": {
        "description": "Logic puzzles requiring formal or informal reasoning",
        "seed_source": "logiqa",
        "state_prompt": (
            "Create a logical reasoning scenario. Include:\n"
            "- A set of premises or given statements\n"
            "- A question about what can be concluded\n"
            "- Multiple possible conclusions (some valid, some not)\n"
            "Keep it to 3-5 sentences."
        ),
        "choice_template": {
            "instructions": "What type of logical relationship applies here?",
            "criteria": {
                "deduction": "Conclusion necessarily follows from premises",
                "induction": "Conclusion is probable but not certain from evidence",
                "abduction": "Best explanation for observed facts",
                "analogy": "Reasoning from similar cases",
                "none": "No valid logical relationship supports the conclusion",
            },
        },
        "score_template": {
            "instructions": "Rate the strength of the argument.",
            "criteria": [
                "Invalid: conclusion does not follow from premises",
                "Weak: conclusion is possible but poorly supported",
                "Moderate: conclusion is plausible with some gaps",
                "Strong: conclusion is well-supported with minor reservations",
                "Deductively valid: conclusion necessarily follows from premises",
            ],
        },
    },
    "adversarial_inference": {
        "description": "Adversarial natural language inference requiring careful reasoning",
        "seed_source": "anli",
        "state_prompt": (
            "Create a natural language inference example designed to be tricky. Include:\n"
            "- A premise containing specific details\n"
            "- A hypothesis that requires careful reading to evaluate\n"
            "- Subtle distinctions (negation, quantifiers, temporal scope)\n"
            "Keep it to 2-4 sentences."
        ),
        "choice_template": {
            "instructions": "What is the relationship between premise and hypothesis?",
            "criteria": {
                "entailment": "Premise guarantees the hypothesis is true",
                "neutral": "Premise neither supports nor contradicts the hypothesis",
                "contradiction": "Premise guarantees the hypothesis is false",
                "ambiguous": "Relationship depends on interpretation of key terms",
            },
        },
        "score_template": {
            "instructions": "Rate the strength of evidence for the relationship.",
            "criteria": [
                "No evidence: premise and hypothesis are unrelated",
                "Weak: slight connection but mostly independent",
                "Moderate: reasonable connection with room for doubt",
                "Strong: clear connection with minor ambiguity",
                "Definitive: relationship is unambiguous and certain",
            ],
        },
    },
    "passage_decision": {
        "description": "Passage-based yes/no questions requiring reading comprehension",
        "seed_source": "boolq",
        "state_prompt": (
            "Create a passage with a yes/no question about it. Include:\n"
            "- A factual passage (3-5 sentences) about a specific topic\n"
            "- A clear yes/no question answerable from the passage\n"
            "- The answer should require reading comprehension, not just keyword matching"
        ),
        "choice_template": {
            "instructions": "What is the quality of evidence in the passage for answering the question?",
            "criteria": {
                "strong": "Passage directly and clearly answers the question",
                "moderate": "Passage implies the answer but requires inference",
                "weak": "Passage has relevant info but answer is uncertain",
                "insufficient": "Passage does not contain enough information",
            },
        },
        "score_template": {
            "instructions": "Rate certainty of the answer based on the passage.",
            "criteria": [
                "Very uncertain: passage is ambiguous about the answer",
                "Somewhat uncertain: passage leans one way but not clearly",
                "Moderate: passage supports the answer with some qualification",
                "Confident: passage clearly supports the answer",
                "Definitive: passage leaves no room for doubt",
            ],
        },
    },
    # --- Long context seeded domains ---
    "long_document": {
        "description": "Long-form document comprehension requiring full-text reasoning",
        "seed_source": "quality",
        "state_prompt": (
            "Create a long-form article or story (5-15 paragraphs). Include:\n"
            "- A clear narrative or argument with multiple sections\n"
            "- Specific details, names, dates, and numbers throughout\n"
            "- Information relevant to the question spread across multiple paragraphs\n"
            "- At least one subtle detail that could be easily missed"
        ),
        "choice_template": {
            "instructions": "What comprehension strategy is most needed to answer questions about this document?",
            "criteria": {
                "detail_retrieval": "Finding a specific fact stated in the text",
                "synthesis": "Combining information from multiple sections",
                "inference": "Drawing conclusions not explicitly stated",
                "structural": "Understanding the document's organization and flow",
                "critical": "Evaluating the author's claims or reasoning",
            },
        },
        "score_template": {
            "instructions": "Rate the complexity of reasoning required to understand this document.",
            "criteria": [
                "Simple: single-paragraph comprehension sufficient",
                "Moderate: need to connect 2-3 sections",
                "Complex: requires synthesizing across most of the document",
                "Advanced: requires inference beyond what is explicitly stated",
                "Expert: requires domain knowledge plus multi-section synthesis",
            ],
        },
    },
    "multi_hop_reasoning": {
        "description": "Multi-paragraph reasoning requiring chain of evidence across sources",
        "seed_source": "hotpotqa",
        "state_prompt": (
            "Create a multi-source reasoning scenario. Include:\n"
            "- 3-5 short paragraphs from different sources about related topics\n"
            "- A question that requires combining facts from at least 2 paragraphs\n"
            "- Some paragraphs that are distractors (relevant topic but not needed)\n"
            "- The answer should NOT be findable in any single paragraph"
        ),
        "choice_template": {
            "instructions": "What type of multi-hop reasoning is required?",
            "criteria": {
                "bridge": "Fact from paragraph A connects to fact in paragraph B",
                "comparison": "Comparing attributes mentioned in different paragraphs",
                "composition": "Combining multiple facts to derive a new fact",
                "temporal": "Ordering events described across paragraphs",
            },
        },
        "score_template": {
            "instructions": "Rate how many reasoning hops are needed to answer.",
            "criteria": [
                "Single hop: answer is in one paragraph",
                "Two hops: need to connect two paragraphs",
                "Three hops: chain across three sources",
                "Four+ hops: complex chain requiring most sources",
                "Unanswerable: information is insufficient even with all sources",
            ],
        },
    },
    "numerical_reasoning": {
        "description": "Quantitative reasoning over text requiring counting, arithmetic, or comparison",
        "seed_source": "drop",
        "state_prompt": (
            "Create a passage with embedded numerical information. Include:\n"
            "- A factual passage (sports, history, science, business) with specific numbers\n"
            "- At least 5 numerical facts (dates, counts, percentages, scores)\n"
            "- Questions answerable through arithmetic operations on these numbers\n"
            "Keep it to 3-6 sentences with dense numerical content."
        ),
        "choice_template": {
            "instructions": "What numerical operation is needed to answer the question?",
            "criteria": {
                "counting": "Count occurrences of items matching criteria",
                "arithmetic": "Add, subtract, multiply, or divide values",
                "comparison": "Compare two or more numerical values",
                "sorting": "Order items by a numerical attribute",
                "extraction": "Simply find and extract a stated number",
            },
        },
        "score_template": {
            "instructions": "Rate the computational complexity of the numerical reasoning.",
            "criteria": [
                "Trivial: direct extraction of a single number",
                "Simple: one arithmetic operation",
                "Moderate: two operations or comparison with filtering",
                "Complex: multi-step calculation or conditional counting",
                "Advanced: requires combining multiple operations with interpretation",
            ],
        },
    },
    # --- Sequential decision domains (pure generation) ---
    "game_strategy": {
        "description": "Turn-based game state requiring strategic next-move decision",
        "seed_source": None,
        "state_prompt": (
            "Create a turn-based game scenario (board game, card game, or strategy game). Include:\n"
            "- Current board/game state with specific positions, scores, or resources\n"
            "- Action history: 3-6 previous moves with their outcomes\n"
            "- Available actions for the current turn (3-5 options)\n"
            "- Win condition or objective\n"
            "Format as:\n"
            "Environment: <game state>\n"
            "History: <action_1 → outcome_1, action_2 → outcome_2, ...>\n"
            "Available actions: <list>\n"
            "Objective: <goal>\n"
            "Keep it to 8-15 sentences. Use specific numbers and positions."
        ),
        "choice_template": {
            "instructions": "Which action is the strongest strategic move?",
            "criteria": {
                "aggressive": "High-risk move that maximizes potential gain",
                "defensive": "Conservative move that minimizes potential loss",
                "positional": "Move that improves long-term position without immediate gain",
                "tactical": "Move that exploits a specific short-term opportunity",
                "neutral": "No clearly superior option — all moves are roughly equal",
            },
        },
        "score_template": {
            "instructions": "Rate the current player's winning probability given the game state.",
            "criteria": [
                "Losing: opponent has decisive advantage",
                "Disadvantaged: opponent has moderate advantage",
                "Even: neither side has clear advantage",
                "Advantaged: current player has moderate advantage",
                "Winning: current player has decisive advantage",
            ],
        },
    },
    "navigation_planning": {
        "description": "Spatial navigation with obstacles requiring path planning",
        "seed_source": None,
        "state_prompt": (
            "Create a navigation/pathfinding scenario. Include:\n"
            "- Environment layout (grid, map, or spatial description with dimensions)\n"
            "- Current position and destination/goal\n"
            "- Obstacles, hazards, or blocked paths (at least 3)\n"
            "- Movement history: 3-5 previous moves and what was encountered\n"
            "- Available movement options from current position\n"
            "Format as:\n"
            "Environment: <layout description>\n"
            "Position: <current>, Goal: <target>\n"
            "History: <move_1 → result_1, move_2 → result_2, ...>\n"
            "Available moves: <list with consequences>\n"
            "Keep it to 8-15 sentences. Be precise about spatial relationships."
        ),
        "choice_template": {
            "instructions": "Which navigation strategy should be used next?",
            "criteria": {
                "shortest_path": "Take the most direct available route to goal",
                "safest_path": "Avoid known hazards even if longer",
                "explore": "Investigate unknown area that might reveal a shortcut",
                "backtrack": "Return to a previous position and try alternate route",
            },
        },
        "score_template": {
            "instructions": "Rate how close the agent is to reaching the goal.",
            "criteria": [
                "Very far: many steps and obstacles remain",
                "Far: significant distance with known obstacles",
                "Moderate: roughly halfway with manageable obstacles",
                "Close: few steps remain with clear path",
                "Arrived: goal is adjacent or reachable in one move",
            ],
        },
    },
    "resource_management": {
        "description": "Resource allocation under constraints requiring optimization",
        "seed_source": None,
        "state_prompt": (
            "Create a resource management scenario. Include:\n"
            "- Available resources with specific quantities (budget, materials, time, personnel)\n"
            "- At least 3 competing demands or projects requiring resources\n"
            "- Constraints (deadlines, minimum allocations, dependencies between tasks)\n"
            "- History: 2-4 previous allocation decisions and their outcomes\n"
            "- Current decision point: what needs to be allocated now\n"
            "Format as:\n"
            "Resources: <inventory with quantities>\n"
            "Demands: <project_1 needs X, project_2 needs Y, ...>\n"
            "Constraints: <rules and deadlines>\n"
            "History: <decision_1 → outcome_1, ...>\n"
            "Decision: <what to allocate now>\n"
            "Keep it to 8-15 sentences with specific numbers."
        ),
        "choice_template": {
            "instructions": "What is the best resource allocation strategy?",
            "criteria": {
                "prioritize_urgent": "Allocate to the most time-critical demand first",
                "maximize_roi": "Allocate to the demand with highest expected return",
                "balanced": "Distribute resources proportionally across demands",
                "reserve": "Hold back resources for anticipated future needs",
                "concentrate": "Put all available resources into a single high-impact demand",
            },
        },
        "score_template": {
            "instructions": "Rate how well the current resource position supports the objectives.",
            "criteria": [
                "Critical: resources are severely insufficient for key objectives",
                "Strained: resources are tight, trade-offs are painful",
                "Adequate: resources can cover priorities with careful allocation",
                "Comfortable: resources allow flexibility and contingency",
                "Abundant: resources exceed needs across all demands",
            ],
        },
    },
    "sequential_action": {
        "description": "Multi-step action planning in a dynamic environment",
        "seed_source": None,
        "state_prompt": (
            "Create a multi-step action planning scenario (robot task, cooking, assembly, etc.). Include:\n"
            "- Current environment state with specific object positions and conditions\n"
            "- Goal state to achieve (what the end result should look like)\n"
            "- Completed steps so far (3-5) with outcomes and any unexpected results\n"
            "- Available actions at current step (4-5 options)\n"
            "- Any preconditions or dependencies between actions\n"
            "Format as:\n"
            "Environment: <current state of objects and conditions>\n"
            "Goal: <desired end state>\n"
            "Completed: <step_1 → result_1, step_2 → result_2, ...>\n"
            "Available actions: <list with preconditions>\n"
            "Keep it to 8-15 sentences. Include at least one unexpected result in history."
        ),
        "choice_template": {
            "instructions": "What should the next action be?",
            "criteria": {
                "proceed_planned": "Continue with the originally planned next step",
                "adapt": "Modify the plan to account for unexpected results",
                "recover": "Take corrective action to fix a problem from a previous step",
                "skip": "Skip the current planned step as it is no longer necessary",
                "verify": "Check or test the current state before proceeding further",
            },
        },
        "score_template": {
            "instructions": "Rate progress toward the goal state.",
            "criteria": [
                "Blocked: cannot proceed without resolving a problem",
                "Behind: fewer steps completed than expected, issues present",
                "On track: progressing as planned with minor deviations",
                "Ahead: more progress than expected, goal is near",
                "Complete: goal state is achieved or achievable in one step",
            ],
        },
    },
    "multi_agent_coordination": {
        "description": "Coordinated decision-making between multiple agents",
        "seed_source": None,
        "state_prompt": (
            "Create a multi-agent coordination scenario (emergency response, team sports, "
            "logistics, military, or collaborative robotics). Include:\n"
            "- 3-4 named agents with their current positions, capabilities, and status\n"
            "- Shared objective that requires cooperation\n"
            "- Communication history: 3-5 messages between agents\n"
            "- Current decision: what one specific agent should do next\n"
            "- Constraints: limited communication, partial information, timing\n"
            "Format as:\n"
            "Agents: <name, position, capability, status for each>\n"
            "Objective: <shared goal>\n"
            "Comms: <agent_A → agent_B: message, ...>\n"
            "Decision for [agent_name]: <what to decide>\n"
            "Constraints: <limitations>\n"
            "Keep it to 10-18 sentences. Each agent should have distinct capabilities."
        ),
        "choice_template": {
            "instructions": "What coordination strategy should the deciding agent adopt?",
            "criteria": {
                "lead": "Take initiative and direct other agents",
                "support": "Assist another agent's ongoing action",
                "independent": "Act alone on a subtask that contributes to the objective",
                "communicate": "Share critical information before acting",
                "wait": "Hold position until other agents complete their actions",
            },
        },
        "score_template": {
            "instructions": "Rate the team's overall coordination effectiveness.",
            "criteria": [
                "Chaotic: agents are working at cross-purposes",
                "Fragmented: some coordination but significant gaps",
                "Functional: basic coordination with room for improvement",
                "Effective: agents are well-coordinated with clear roles",
                "Optimal: agents are perfectly synchronized toward the objective",
            ],
        },
    },
}

SEEDED_DOMAINS = [d for d, t in DOMAIN_TEMPLATES.items() if t.get("seed_source")]
GENERATED_DOMAINS = [d for d, t in DOMAIN_TEMPLATES.items() if not t.get("seed_source")]

FAMILIES_PER_DOMAIN = 200
NOUL_QUESTIONS_PER_FAMILY = 4
CHOICE_QUESTIONS_PER_FAMILY = 3
SCORE_QUESTIONS_PER_FAMILY = 2


def build_seeded_family_prompt(
    domain: str, seed_state: str, cognitive_types: list[str]
) -> str:
    """Build prompt for generating questions around a real (seed) state."""
    template = DOMAIN_TEMPLATES[domain]

    cog_type_block = "\n".join(
        f"  - {ct}: {COGNITIVE_TYPE_DESCRIPTIONS[ct]}" for ct in cognitive_types
    )

    return f"""You are given a real-world state from the {domain} domain. Generate a diverse set of typed decision questions about it.

DOMAIN: {domain} — {template['description']}

STATE (from real data — use it as-is, do NOT modify it):
{seed_state}

QUESTIONS TO GENERATE:
You must generate questions about the state in these exact formats:

1. NOUL QUESTIONS ({len(cognitive_types)} questions, one per cognitive type):
Each noul question tests a yes/no judgment. Generate one for each cognitive type:
{cog_type_block}

For each noul question, provide:
- "instructions": a clear yes/no question about the state
- "label": true or false (your gold label)

2. CHOICE QUESTIONS ({CHOICE_QUESTIONS_PER_FAMILY} questions):
The FIRST choice question must use this exact template:
- "instructions": "{template['choice_template']['instructions']}"
- "criteria": {json_compact(template['choice_template']['criteria'])}
- "label": one of the criteria keys (MUST be an exact key from criteria)

The remaining {CHOICE_QUESTIONS_PER_FAMILY - 1} choice question(s): invent your own question about a DIFFERENT aspect of the state. Design your own criteria dict (3-5 options) and pick the correct label.

3. SCORE QUESTIONS ({SCORE_QUESTIONS_PER_FAMILY} questions):
The FIRST score question must use this exact template:
- "instructions": "{template['score_template']['instructions']}"
- "criteria": {json_compact(template['score_template']['criteria'])}
- "label": a float from 1.0 to {len(template['score_template']['criteria'])}.0

The remaining {SCORE_QUESTIONS_PER_FAMILY - 1} score question(s): invent your own question about a DIFFERENT aspect of the state. Design your own criteria list (3-5 levels) and pick the correct label.

Respond with a JSON object matching this exact schema:
{{
  "noul_questions": [
    {{
      "cognitive_type": "<type>",
      "instructions": "<yes/no question>",
      "label": true/false
    }}
  ],
  "choice_questions": [
    {{
      "instructions": "<question>",
      "criteria": {{...}},
      "label": "<chosen key>"
    }}
  ],
  "score_questions": [
    {{
      "instructions": "<question>",
      "criteria": ["<level1>", ...],
      "label": <float>
    }}
  ]
}}

IMPORTANT:
- The questions must be answerable from the state alone
- Noul labels must be defensible — a reasonable expert should agree
- Choice labels MUST be an exact key from that question's criteria dict
- Score labels MUST be a float within [1.0, N] where N is the number of criteria levels
- Vary difficulty: some should be clear-cut, some genuinely ambiguous
- Do NOT repeat or paraphrase the original question the state was designed for
- Cover DIFFERENT aspects of the state across cognitive types
- Do NOT include analysis, judgment, or explanation of what is correct/incorrect in questions — the model should figure that out
- Output ONLY the JSON object, no other text"""


def build_generated_family_prompt(domain: str, cognitive_types: list[str]) -> str:
    """Build prompt for generating a complete family (state + questions) from scratch."""
    template = DOMAIN_TEMPLATES[domain]

    cog_type_block = "\n".join(
        f"  - {ct}: {COGNITIVE_TYPE_DESCRIPTIONS[ct]}" for ct in cognitive_types
    )

    return f"""Generate a synthetic training example for a typed decision model.

DOMAIN: {domain} — {template['description']}

STATE REQUIREMENTS:
{template['state_prompt']}

QUESTIONS TO GENERATE:
You must generate questions about the state in these exact formats:

1. NOUL QUESTIONS ({len(cognitive_types)} questions, one per cognitive type):
Each noul question tests a yes/no judgment. Generate one for each cognitive type:
{cog_type_block}

For each noul question, provide:
- "instructions": a clear yes/no question about the state
- "label": true or false (your gold label)

2. CHOICE QUESTIONS ({CHOICE_QUESTIONS_PER_FAMILY} questions):
The FIRST choice question must use this exact template:
- "instructions": "{template['choice_template']['instructions']}"
- "criteria": {json_compact(template['choice_template']['criteria'])}
- "label": one of the criteria keys (MUST be an exact key from criteria)

The remaining {CHOICE_QUESTIONS_PER_FAMILY - 1} choice question(s): invent your own question about a DIFFERENT aspect of the state. Design your own criteria dict (3-5 options) and pick the correct label.

3. SCORE QUESTIONS ({SCORE_QUESTIONS_PER_FAMILY} questions):
The FIRST score question must use this exact template:
- "instructions": "{template['score_template']['instructions']}"
- "criteria": {json_compact(template['score_template']['criteria'])}
- "label": a float from 1.0 to {len(template['score_template']['criteria'])}.0

The remaining {SCORE_QUESTIONS_PER_FAMILY - 1} score question(s): invent your own question about a DIFFERENT aspect of the state. Design your own criteria list (3-5 levels) and pick the correct label.

Respond with a JSON object matching this exact schema:
{{
  "state": "<the generated state text>",
  "noul_questions": [
    {{
      "cognitive_type": "<type>",
      "instructions": "<yes/no question>",
      "label": true/false
    }}
  ],
  "choice_questions": [
    {{
      "instructions": "<question>",
      "criteria": {{...}},
      "label": "<chosen key>"
    }}
  ],
  "score_questions": [
    {{
      "instructions": "<question>",
      "criteria": ["<level1>", ...],
      "label": <float>
    }}
  ]
}}

IMPORTANT:
- The state must be realistic and detailed
- Noul labels must be defensible — a reasonable expert should agree
- Choice labels MUST be an exact key from that question's criteria dict
- Score labels MUST be a float within [1.0, N] where N is the number of criteria levels
- Vary difficulty: some should be clear-cut, some genuinely ambiguous
- Do NOT leak the answer in the state text
- Do NOT include analysis, judgment, or explanation of what is correct/incorrect in the state — present only the raw scenario
- Output ONLY the JSON object, no other text"""


_VARIANT_CONTEXT = (
    "You are generating synthetic training data for a machine learning model "
    "that evaluates typed decisions. The data is used for academic research "
    "on model calibration and decision quality. All content is fictional and "
    "for training purposes only."
)


def build_variant_prompt(family_json: str, variant_type: str) -> str:
    """Build prompt for generating a variant of an existing family."""
    if variant_type == "counterfactual":
        return f"""{_VARIANT_CONTEXT}

Given this training example, create a COUNTERFACTUAL variant.

ORIGINAL:
{family_json}

Change ONE key fact in the state so that at least 2 of the noul question labels flip.
Keep the questions identical — only the state and affected labels change.

Rules:
- Change should be minimal but meaningful (e.g., changing a vital sign, reversing a timeline)
- Labels that depend on the changed fact must flip; others stay the same
- The choice and score labels should also be re-evaluated given the new state

Respond with a JSON object:
{{
  "state": "<modified state>",
  "changed_fact": "<description of what changed>",
  "noul_labels": [{{"cognitive_type": "<type>", "label": true/false, "flipped": true/false}}],
  "choice_label": "<new choice>",
  "score_label": <new float>
}}

Output ONLY the JSON object."""

    elif variant_type == "paraphrase":
        return f"""{_VARIANT_CONTEXT}

Given this training example, create TWO paraphrase variants.

ORIGINAL:
{family_json}

Variant 1: Reword the STATE using different vocabulary and sentence structure.
Variant 2: Reword the QUESTIONS using different phrasing.

Rules:
- Semantics must be preserved exactly — labels must NOT change
- Use genuinely different wording, not just synonym substitution
- Maintain the same level of detail and specificity

Respond with a JSON object:
{{
  "state_paraphrase": "<reworded state>",
  "question_paraphrases": [
    {{"original_instructions": "<original>", "paraphrased_instructions": "<new>"}}
  ]
}}

Output ONLY the JSON object."""

    elif variant_type == "negation":
        return f"""{_VARIANT_CONTEXT}

Given this training example, create NEGATION variants for each noul question.

ORIGINAL:
{family_json}

For each noul question, flip the polarity of the question (e.g., "Is X safe?" → "Is X unsafe?").
The label should flip accordingly.

Rules:
- The negated question must be natural — not just prepending "Is it NOT true that..."
- The label must logically flip (true ↔ false)
- Keep the state unchanged

Respond with a JSON object:
{{
  "negated_questions": [
    {{
      "cognitive_type": "<type>",
      "original_instructions": "<original>",
      "negated_instructions": "<negated version>",
      "original_label": true/false,
      "negated_label": true/false
    }}
  ]
}}

Output ONLY the JSON object."""

    raise ValueError(f"Unknown variant type: {variant_type}")


def build_validation_prompt(state: str, question: str, label: str) -> str:
    """Build prompt for cross-model validation of a single item."""
    return f"""You are a careful evaluator. Given the following state and question, determine if the provided label is correct.

STATE:
{state}

QUESTION:
{question}

PROVIDED LABEL: {label}

Is this label correct? Respond with ONLY a JSON object:
{{
  "correct": true/false,
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}"""


def json_compact(obj: object) -> str:
    """Compact JSON for embedding in prompts."""
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))

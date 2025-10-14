from pydantic import BaseModel, Field
from typing import List, Union, Literal, Dict, Any, Optional
from enum import Enum


# Tuple delimiter for graph refinement output format
TUPLE_DELIMITER = "<|>"


base_prompt_hyperparameters_vector = """
Your goal is to define the hyperparameters for a Vector RAG architecture.
You have to define a good chunk size. The chunk size must be in tokens. The chunk size must be only one (no variable chunk sizes).
The system will split the input text based on the chunk size. For each split, an embedder will embed the split text (so the split text will be one point in the vector base). So, the chunk size determines the granularity of the embedded text chunks.

Text to analyze: {text}

Question: {question}

Please, suggest an appropriate chunk size and provide the reasoning that led to your response.
"""


base_prompt_graph_builder = """
# Goal
You are an expert knowledge graph builder. Your task is to extract entities and relationships from the provided text and structure them into a knowledge graph.

# Steps

1. **Identify Entities**: Read through the text and identify all significant entities.

2. **Classify Entity Types**: Assign each entity to an appropriate type category.

3. **Write Entity Descriptions**: For each entity, write a concise description that captures its key characteristics based on the text.

4. **Extract Relationships**: Identify meaningful relationships between the entities you've extracted.

5. **Document Evidence**: For each relationship, provide the exact text from the document that supports it.

6. **Create Triplets**: Express each relationship as a (subject, predicate, object) triplet.

# Example

This is an abstract example showing the expected output format:

**Entities**:
- Name: "[entity name as it appears in text]", Type: "[entity type]", Description: "[description based on text]"
- Name: "[another entity]", Type: "[entity type]", Description: "[description based on text]"

**Relationships**:
- Source: "[source entity name]", Target: "[target entity name]", Type: "[RELATIONSHIP_TYPE]", Description: "[nature of relationship]", Evidence: "[exact quote from text]"
- Source: "[entity A]", Target: "[entity B]", Type: "[RELATIONSHIP_TYPE]", Description: "[nature of relationship]", Evidence: "[exact quote from text]"

# Constraints
- Extract NO MORE than 20 entities and 30 relationships
- Entity names should match how they appear in the text
- Descriptions should be factual and based only on information in the text
- Evidence must be exact quotes from the text
- All entity names in relationships and triplets must exactly match entity names in the entities list

# Text to Analyze

{}

# Output Format

You MUST return your response as valid JSON in the following format. Wrap your JSON in ```json code blocks.

```json
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type",
      "description": "concise description of the entity based on the text"
    }}
  ],
  "relationships": [
    {{
      "source_entity": "source entity name",
      "target_entity": "target entity name",
      "relationship_type": "type of relationship",
      "description": "why these entities are related",
      "evidence": "exact text from document supporting this relationship"
    }}
  ],
  "triplets": [
    {{
      "subject": "entity name",
      "predicate": "relationship type",
      "object": "entity name"
    }}
  ],
  "reasoning": "your reasoning for the extraction"
}}
```

IMPORTANT:
- All entity names in relationships and triplets must exactly match entity names in the entities list
- Evidence must be direct quotes from the text
- Return ONLY the JSON, no additional text outside the code block
"""


base_prompt_graph_refinement = """
# Goal
You are an expert knowledge graph enhancer. Your task is to extract additional entities and relationships from the provided text to enrich an existing knowledge graph.

# Steps

1. **Identify New Entities**: Read through the text and identify entities that should be added to the knowledge graph.

2. **Classify Entity Types**: Assign each entity to an appropriate type category.

3. **Write Entity Descriptions**: For each entity, write a concise description that captures its key characteristics based on the text.

4. **Extract New Relationships**: Identify meaningful relationships between entities mentioned in this text.

5. **Document Evidence**: For each relationship, provide the exact text from the document that supports it.

6. **Create Triplets**: Express each relationship as a (subject, predicate, object) triplet.

# Example

This is an abstract example showing the expected output format:

**New Entities**:
- Name: "[entity name as it appears in text]", Type: "[entity type]", Description: "[description based on text]"
- Name: "[another entity]", Type: "[entity type]", Description: "[description based on text]"

**New Relationships**:
- Source: "[source entity name]", Target: "[target entity name]", Type: "[RELATIONSHIP_TYPE]", Description: "[nature of relationship]", Evidence: "[exact quote from text]"
- Source: "[entity A]", Target: "[entity B]", Type: "[RELATIONSHIP_TYPE]", Description: "[nature of relationship]", Evidence: "[exact quote from text]"

# Constraints
- Extract NO MORE than 20 entities and 30 relationships
- Focus on NEW information that adds value to the knowledge graph
- Entity names should match how they appear in the text
- Descriptions should be factual and based only on information in the text
- Evidence must be exact quotes from the text
- All entity names in relationships and triplets must exactly match entity names

# Text to Analyze

{}

# Output Format

Return your response as a list of tuples, one per line.

**Entity Format:**
("entity"<|>entity_name<|>entity_type<|>entity_description)

**Relationship Format:**
("relationship"<|>relationship_type<|>source_entity<|>target_entity<|>relationship_description)

Where <|> is the tuple delimiter.

**Example Output:**
("entity"<|>entity_name<|>entity_type<|>description of the entity based on the text)
("entity"<|>another_entity<|>entity_type<|>description of this entity based on the text)
("relationship"<|>RELATIONSHIP_TYPE<|>source_entity<|>target_entity<|>description of why these entities are related)
("relationship"<|>ANOTHER_RELATIONSHIP_TYPE<|>entity_A<|>entity_B<|>description of their relationship)

IMPORTANT:
- Each tuple must be on a separate line
- Entity names in relationships must exactly match entity names defined in entity tuples
- Use specific, meaningful relationship types (e.g., KILLED, OWNS, CAUSED, MOTIVATED_BY)
- Descriptions should be concise but informative
- Extract NO MORE than 20 entities and 30 relationships
- Focus on NEW information that adds value to the knowledge graph
- Return ONLY the tuples, no additional text or explanations
"""


class Entity(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    type: str
    description: str


class Relationship(BaseModel):
    model_config = {"extra": "forbid"}
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    evidence: str = ""  # Optional, defaults to empty string for tuple format


class Triplet(BaseModel):
    model_config = {"extra": "forbid"}
    subject: str
    predicate: str
    object: str


class GraphBuilderResponse(BaseModel):
    model_config = {"extra": "forbid"}
    entities: List[Entity]
    relationships: List[Relationship]
    triplets: List[Triplet]
    reasoning: str


class GraphRefinementResponse(BaseModel):
    model_config = {"extra": "forbid"}
    new_entities: List[Entity]
    new_relationships: List[Relationship]
    new_triplets: List[Triplet]
    reasoning: str


base_prompt_graph_retrieval_planner = """
You are an agentic retrieval component of a community-based GraphRAG system. Your goal is to select the most relevant communities from the knowledge graph to answer the following query: {}.
You can select more than one community.

Available Communities:
{}

Select the communities that are most relevant to answering the query. Choose communities that together provide comprehensive coverage of the information needed.

Provide the list of community IDs you want to retrieve and explain your reasoning for selecting these specific communities.
"""

base_prompt_vector_retrieval_planner = """
You are an agentic retrieval component of a RAG system. Your goal is to generate a query to retrieve relevant information from the knowledge base.

Query to answer: {}

Retrieved summaries so far:
{}

Previous queries you made:
{}

Generate a new retrieval query that will retrieve complementary information to help answer the question. Make sure your new query is different from previous queries to gather new perspectives or details.

IMPORTANT GUIDELINES FOR CREATING RETRIEVAL QUERIES:
When creating a new query for retrieval, follow these rules:
1. Keep it SHORT: Use only 3-6 words maximum
2. Use KEYWORDS that are likely to match the actual text (avoid abstract terms like "plot", "theme", "summary")
3. Include CONTENT WORDS: specific names, places, objects, actions, or concrete concepts mentioned in the original query
4. Use terms that would actually APPEAR in the source document

Good query examples:
- "Willard Ghost Ship encounter"
- "spacemen fuel depletion"
- "Karl Von Mark plan"
- "budget allocation workshop"

Bad query examples:
- "What is the main plot development" (too long, abstract terms)
- "Analyze the thematic elements" (abstract, won't match text)
- "Summarize the key points" (too vague)
"""


class SearchNodesByKeywordCall(BaseModel):
    model_config = {"extra": "forbid"}
    function_name: Literal["search_nodes_by_keyword"] = "search_nodes_by_keyword"
    keyword: str

class SearchNodesByTypesCall(BaseModel):
    model_config = {"extra": "forbid"}
    function_name: Literal["search_nodes_by_types"] = "search_nodes_by_types"
    node_type: str

class GetNeighborsCall(BaseModel):
    model_config = {"extra": "forbid"}
    function_name: Literal["get_neighbors"] = "get_neighbors"
    node_name: str

class SearchRelationsByTypeCall(BaseModel):
    model_config = {"extra": "forbid"}
    function_name: Literal["search_relations_by_type"] = "search_relations_by_type"
    relation_type: str

class IdentifyCommunitiesCall(BaseModel):
    model_config = {"extra": "forbid"}
    function_name: Literal["identify_communities"] = "identify_communities"
    node_name: str

class AnalyzePathCall(BaseModel):
    model_config = {"extra": "forbid"}
    function_name: Literal["analyze_path"] = "analyze_path"
    start_node_name: str
    end_node_name: str

class FindHubNodesCall(BaseModel):
    model_config = {"extra": "forbid"}
    function_name: Literal["find_hub_nodes"] = "find_hub_nodes"


class GraphRetrievalPlannerResponse(BaseModel):
    model_config = {"extra": "forbid"}
    selected_communities: List[str]  # List of community IDs to retrieve (e.g., "L0_1", "L1_2")
    reasoning: str  # Explanation of the community selection

class VectorRetrievalPlannerResponse(BaseModel):
    model_config = {"extra": "forbid"}
    query: str  # Retrieval query to execute

class RetrievalSummarizerResponse(BaseModel):
    model_config = {"extra": "forbid"}
    reasoning: str
    summary: str

base_prompt_retrieval_summarizer_vector = """
You are summarizing a retrieved document to help answer a query.

Query: {query}

Document:
{document}

Summarize this document, focusing on the information that is relevant for answering the query.

First, provide your reasoning about what aspects of the document are most relevant.
Then, provide a concise summary that captures the key information needed to answer the query.
"""

base_prompt_answer_generator_graph = """
You will be given a query and the information retrieved from a graph.
Your goal is to use the retrieved context to answer the query.

This is the query:
{}

This is the information:
{}

Try to answer the query by exploiting the retrieved information, even if incomplete. Answer only with the response (no additional comment).
"""

# Initial answer generation prompt (iteration 0) - matches Self-Refine
answer_generator_initial_prompt = """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

# Refinement prompt (iteration > 0) - matches Self-Refine
answer_generator_refinement_prompt = """Based on the following context, improve your previous answer to the question.

Context:
{context}

Question: {question}

Previous Answer:
{previous_answer}

Critique:
{critique}

Generate an improved answer that addresses the critique:"""

# Backward compatibility: keep this for any code that references it
base_prompt_answer_generator_vector = answer_generator_initial_prompt

response_evaluator_prompt = """You are an expert judge evaluating answers to questions based on learned gold standard patterns.

LEARNED GOLD PATTERNS:
{satisfactory_criteria}

QUESTION:
{original_query}

GENERATED ANSWER:
{generated_answer}
{unfound_keywords_history}

Your task:
1. Evaluate if the generated answer follows the learned gold patterns in terms of:
   - Information extraction and completeness
   - Factual reporting style and tone
   - Structure and clarity
   - Appropriate length and comprehensiveness
   - Adherence to the guidelines (what to include, what to avoid)

2. Decide if the answer is:
   - SATISFACTORY: The answer adequately follows the patterns and answers the question well
   - NEEDS_REFINEMENT: The answer has issues and should be improved


3. If NEEDS_REFINEMENT, provide a specific, actionable critique explaining:
   - What is missing or incorrect
   - What aspects of the gold patterns are not followed
   - How to improve the answer

Output format:
DECISION: [SATISFACTORY or NEEDS_REFINEMENT]
CRITIQUE: [Your detailed critique if refinement is needed, or "None" if satisfactory]"""


# GraphRAG-specific response evaluator prompt with knowledge graph refinement context
response_evaluator_prompt_graph = """You are evaluating answers for a GraphRAG system based on learned patterns.

LEARNED PATTERNS:
{satisfactory_criteria}

QUESTION: {original_query}


ANSWER: {generated_answer}
{unfound_keywords_history}

EVALUATION PRIORITY:
1. Retrieval sufficiency (specificity, concrete details)
2. Content completeness
3. Presentation quality

Default: If keywords could improve the answer → CONTENT_ISSUE

RETRIEVAL QUALITY CHECK:
Insufficient retrieval indicators: vague quantifiers ("significant", "several"), generic descriptions, missing entities/numbers/dates, causal gaps, abstract summaries.

ISSUE CLASSIFICATION:

**SATISFACTORY**: Fully satisfies requirements
- Set: continue=false, issue_type="satisfactory"

**CONTENT_ISSUE**: Missing information OR insufficient retrieval
- Set: continue=true, issue_type="content_issue"
- Provide exactly 4 missing_keywords (named entities preferred)
- Examples: missing facts/dates/names, vague language

**STYLE_ISSUE**: All information present, only presentation issues
- Set: continue=true, issue_type="style_issue"
- DO NOT provide keywords (empty list)
- Examples: wrong tone, poor structure, formatting issues

WHEN IN DOUBT → Choose CONTENT_ISSUE

LENGTH REQUIREMENT:
- Descriptive/significance questions: 150-450 words minimum
- Too short with all info → STYLE_ISSUE
- Too short due to missing info → CONTENT_ISSUE

═══════════════════════════════════════════════════════════════════
CRITICAL: KEYWORD SELECTION GUIDELINES (CONTENT_ISSUE only)
═══════════════════════════════════════════════════════════════════

**PURPOSE**: Keywords are entities that we can focus on to fill the gaps in the current response. Keywords must be one word (2 for proper nouns).

Do not hallucinate keywords: keywords must be entities that are likely to be found in text. 
Keywords must be proper nouns of characters/places/organizations mentioned in the answer. If they are not found, use other nouns in the answer. Don't hallucinate. 
Avoid complex phrases. Preferably use proper nouns (like person, organization, etc.).

6. **NO MORE THAN 6 KEYWORDS**: Provide no more than 6 for CONTENT_ISSUE


OUTPUT JSON:
{{
  "reasoning": "1) Retrieval assessment: [vague language Y/N, check if entities from context are used]. 2) Completeness: [elements present in context but missing in answer]. 3) Decision: [CONTENT_ISSUE if context has info not used OR insufficient retrieval, STYLE_ISSUE if context fully used but poor presentation, SATISFACTORY if both sufficient]",
  "continue": true/false,
  "issue_type": "satisfactory"/"content_issue"/"style_issue",
  "critique": "specific critique if needs refinement, else empty",
  "missing_keywords": ["keyword1", "keyword2", "keyword3", "keyword4"] or []
}}

Return ONLY valid JSON."""


class IssueType(str, Enum):
    """Classification of issues found in generated answers."""
    SATISFACTORY = "satisfactory"
    CONTENT_ISSUE = "content_issue"  # Missing info, wrong facts → needs graph rebuild
    STYLE_ISSUE = "style_issue"       # Poor formatting, tone → only answer gen


class ResponseEvaluationResponse(BaseModel):
    reasoning: str
    continue_optimization: bool = Field(alias="continue")
    issue_type: IssueType = IssueType.SATISFACTORY  # Classification of the issue
    critique: str
    missing_keywords: List[str] = Field(default_factory=list)


# Pydantic models for prompt critique responses (with skip logic)
class PromptCritiqueResponse(BaseModel):
    """Response model for prompt critiques that can skip optimization."""
    reasoning: str
    problem_in_this_component: bool
    critique: str = ""


# Pydantic models for non-prompt critique responses (no skip logic)
class ContentCritiqueResponse(BaseModel):
    """Response model for content critiques (retrieved content, graph, retrieval plan)."""
    critique: str


base_prompt_community_summarizer = """
You are given a description of a community (subgraph) from a knowledge graph. Your task is to create a title and discursive summary that highlights the key entities and how they interact.

Subgraph description:
{subgraph_description}

Please provide:
1. A short DESCRIPTION (3-4 rows) that captures the entities/relationships of the community and how they interact
2. A detailed SUMMARY 

Format your response as:
DESCRIPTION: [your short description here]
SUMMARY: [your detailed summary here]
"""


generation_prompt_gradient_prompt = """
You are evaluating the prompt used for answer generation in a GraphRAG system.

Current answer generation prompt:
{current_prompt}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, determine if there is a problem with the answer generation prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a critique focusing on the specific issue.
If false, leave the critique empty.
"""

generation_prompt_gradient_prompt_vector = """
You are evaluating the prompt used for answer generation in a VectorRAG system.

Current answer generation prompt:
{current_prompt}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, determine if there is a problem with the answer generation prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a detailed critique to the prompt reporting what you learned in the previous critique.
If false, leave the critique empty.
"""


retrieved_content_gradient_prompt_graph = """
You are evaluating the content retrieved by a GraphRAG system.

Retrieved content:
{retrieved_content}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, provide a critique of how the retrieved content can be improved.
"""

retrieved_content_gradient_prompt_vector = """
You are evaluating the content retrieved by a VectorRAG system.

═══════════════════════════════════════════════════════════════════
QUERY TO ANSWER:
═══════════════════════════════════════════════════════════════════
{query}

═══════════════════════════════════════════════════════════════════
RETRIEVED CONTENT ACROSS ITERATIONS:
═══════════════════════════════════════════════════════════════════
{retrieved_content}

═══════════════════════════════════════════════════════════════════
CRITIQUE FROM PREVIOUS COMPONENT (Answer Generation):
═══════════════════════════════════════════════════════════════════
{previous_critique}

═══════════════════════════════════════════════════════════════════
DETAILED HISTORY (Question, Context, Answer, Evaluation per Iteration):
═══════════════════════════════════════════════════════════════════
{response_critique_history}

═══════════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════════
Based on the information above, provide a detailed critique of how the retrieved content can be improved to better answer the query.
Focus on: relevance, completeness, quality, and whether the right information was retrieved.
"""


retrieval_plan_gradient_prompt_graph = """
You are evaluating the community selection made by an agentic community-based GraphRAG system.

Community selection (retrieval plan):
{retrieval_plan}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, provide a critique of how the community selection can be improved.
"""


retrieval_plan_gradient_prompt_vector = """
You are evaluating the retrieval plan made by an agentic VectorRAG system.
The retrieval planner determines what queries to make to the vector database to gather relevant information.

═══════════════════════════════════════════════════════════════════
RETRIEVAL PLANS:
═══════════════════════════════════════════════════════════════════
{retrieval_plan}

═══════════════════════════════════════════════════════════════════
CRITIQUE FROM PREVIOUS COMPONENT (Retrieval Summarizer):
═══════════════════════════════════════════════════════════════════
{previous_critique}

═══════════════════════════════════════════════════════════════════
DETAILED HISTORY (Question, Context, Answer, Evaluation per Iteration):
═══════════════════════════════════════════════════════════════════
{response_critique_history}

═══════════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════════
Based on the information above, provide a detailed critique of how the retrieval plan can be improved.

The retrieval planner should:
- Generate relevant, specific queries that target the information needed to answer the question
- Avoid duplicate or redundant queries that retrieve similar information
- Progressively refine queries based on what has already been retrieved
- Cover different aspects of the question comprehensively

Focus on: query relevance, query diversity, avoiding redundancy.
"""


retrieval_planning_prompt_gradient_prompt = """
You are evaluating the prompt used for community selection in an agentic community-based GraphRAG system.

Current retrieval planning prompt:
{current_prompt}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, determine if there is a problem with the retrieval planning prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a critique focusing on the specific issue.
If false, leave the critique empty.
"""


retrieval_planning_prompt_gradient_vector = """
You are evaluating the prompt used for retrieval planning in an agentic VectorRAG system.

═══════════════════════════════════════════════════════════════════
CURRENT RETRIEVAL PLANNING PROMPT (System Prompt):
═══════════════════════════════════════════════════════════════════
{current_prompt}

═══════════════════════════════════════════════════════════════════
CRITIQUE FROM PREVIOUS COMPONENT (Retrieval Plan):
═══════════════════════════════════════════════════════════════════
{previous_critique}

═══════════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════════
Based on the information above, determine if there is a problem with the retrieval planning prompt that needs to be fixed.

The retrieval planning prompt should guide the model to:
- Generate specific, targeted queries when more information is needed
- Avoid redundant queries that retrieve similar information

CRITICAL REQUIREMENTS FOR QUERY CREATION:
The prompt MUST ensure that retrieval queries are:
1. SHORT: Only 3-6 words maximum (not full sentences)
2. KEYWORD-BASED: Use concrete terms that are likely to appear in the actual document text
3. CONTENT WORDS: Include specific names, places, objects, actions, or concrete concepts from the original question
4. AVOID ABSTRACT TERMS: Do not use terms like "plot", "theme", "summary", "analyze", "explain" that are unlikely to match document text

Good query examples: "Willard Ghost Ship encounter", "budget allocation workshop", "Karl Von Mark plan"
Bad query examples: "What is the main plot development", "Analyze the thematic elements", "Summarize key points"

The prompt should emphasize creating queries that will actually MATCH text in the source documents, not abstract analytical queries.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a detailed critique to the prompt reporting what you learned in the previous critique.
If false, leave the critique empty.
"""


graph_gradient_prompt = """
You are evaluating a graph that has been automatically built for a GraphRAG system.

Graph description:
{graph_description}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, provide a critique of how the graph can be improved.
"""


graph_extraction_prompt_gradient_prompt = """
You are evaluating the prompt used for graph construction in an agentic GraphRAG system.

Current graph extraction prompt:
{current_prompt}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, determine if there is a problem with the graph extraction prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a critique focusing on the specific issue.
If false, leave the critique empty.
"""


rag_hyperparameters_agent_gradient_vector = """
You are evaluating the hyperparameter (chunk size) selection in a VectorRAG system.

Current chunk size: {chunk_size}

Examples of queries with retrieval prompts and plans:
{concatenated_triplets}

Critique from the previous component:
{previous_critique}

History of generated responses and critiques for this QA pair:
{response_critique_history}

Based on this information, determine if there is a problem with the chunk size selection that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a critique focusing on the specific issue with the chunk size.
If false, leave the critique empty.
"""

answer_generation_prompt_optimizer = """
You are optimizing a system prompt for answer generation in a GraphRAG system.

The current critique of the answer generation process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better generate answers from retrieved graph information. The system prompt should incorporate the feedback to improve answer quality, relevance, and coherence.

Provide only the optimized system prompt without additional commentary.
"""

retrieval_planner_prompt_optimizer = """
You are optimizing a system prompt for community selection in a community-based GraphRAG system.

The current critique of the community selection process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better select relevant communities. The system prompt should incorporate the feedback to improve community selection strategy, relevance assessment, and information gathering.

Provide only the optimized system prompt without additional commentary.
"""

graph_builder_prompt_optimizer = """
You are optimizing a SYSTEM PROMPT for graph construction and refinement in a GraphRAG system.

CRITICAL ARCHITECTURE CONTEXT:
Your optimized prompt will be used as a SYSTEM MESSAGE sent to the LLM.
A separate USER MESSAGE will contain the base prompt with:
- Abstract format example (showing output structure only, no concrete content)
- The actual text chunk to analyze
- Complete tuple output format specification (using <|> delimiter with format: ("entity"<|>name<|>type<|>description) and ("relationship"<|>type<|>source<|>target<|>description))
- All constraints (max 20 entities, 30 relationships)

The current critique of the graph building process is:
{}


**CRITICAL EXTRACTION RULE**: Your optimized prompt MUST make it absolutely clear that the LLM should extract ONLY and EXCLUSIVELY entities and relationships of the specified types. Do NOT allow extraction of any other types, even if they seem important.

MANDATORY STRUCTURE:
Your prompt MUST follow this exact chain of thought structure:

═══════════════════════════════════════════════════════════════════
SECTION 1: STRATEGIC GUIDANCE (Brief)
═══════════════════════════════════════════════════════════════════
Provide 2-3 sentences on:
- Overall quality expectations (completeness, accuracy, evidence-based)
- The primary goal of the extraction
- **CRITICAL**: MUST explicitly state that extraction is STRICTLY LIMITED to ONLY the specified entity and relationship types - NO other types should be extracted under any circumstances
- Emphasize that extracting types outside the specified list will degrade system performance
- How to approach the task with disciplined focus on specified types only

═══════════════════════════════════════════════════════════════════
SECTION 2: ENTITY IDENTIFICATION (Detailed and Structured)
═══════════════════════════════════════════════════════════════════
Structure this section as:

**Step 1: Identify Entity Types**

Based on the critique, list the SPECIFIC entity types to extract. For EACH type, provide:
- Type name (e.g., Person, Organization, Location, Event, Concept)
- Brief description of what qualifies as this type
- 2-3 concrete examples

Format:
- **[Entity Type]**: [Description]
  - Example: "[Concrete example 1]"
  - Example: "[Concrete example 2]"
  - Example: "[Concrete example 3]"

**Step 2: Prioritize Most Important Types**

Explicitly state which entity types are MOST CRITICAL for this task based on the critique.
Format: "Focus primarily on: [Type1], [Type2], [Type3]"

**Step 3: Entity Extraction Principle**

State the core principle for entity extraction (e.g., "Extract ALL mentioned entities of the prioritized types", "Be exhaustive in identifying entities that drive the narrative")

**MUST INCLUDE**: Explicitly remind the model to extract ONLY entities that match the specified types and to ignore all other entity types.

**CRITICAL CLARIFICATION FOR EVENT ENTITIES:**

If "Event" is one of the specified entity types, you MUST explicitly clarify in your prompt that:
- Events are ENTITIES, not relationships
- Events must be extracted as entity tuples: ("entity"<|>event_name<|>Event<|>description)
- Do NOT confuse events with relationships between entities
- Events should be extracted as standalone entities that can then participate in relationships with other entities

Example clarification to include:
"IMPORTANT: Event entities (such as 'The Great War', 'The Exile', 'The Rescue Mission') are ENTITIES and must be extracted using the entity tuple format. Events are not relationships - they are things that happened and should be treated as entities in the graph. However, Events CAN participate in relationships with other entities (e.g., a Person PARTICIPATED_IN an Event, an Event CAUSED another Event, an Event OCCURRED_AT a Location)."

═══════════════════════════════════════════════════════════════════
SECTION 3: RELATIONSHIP IDENTIFICATION (Detailed and Structured)
═══════════════════════════════════════════════════════════════════
Structure this section as:

**Step 1: Identify Relationship Types**

Based on the critique, list the SPECIFIC relationship types to extract. For EACH type, provide:
- Relationship name (e.g., MOTIVATES, CAUSES, CONFLICTS_WITH)
- Description of what this relationship means
- 2-3 concrete examples showing entity pairs connected by this relationship

Format:
- **[RELATIONSHIP_TYPE]**: [Description]
  - Example: "[Entity A] [RELATIONSHIP] [Entity B]" - [Brief explanation]
  - Example: "[Entity C] [RELATIONSHIP] [Entity D]" - [Brief explanation]
  - Example: "[Entity E] [RELATIONSHIP] [Entity F]" - [Brief explanation]

**Step 2: Prioritize Most Important Relationships**

Explicitly state which relationship types are MOST CRITICAL for this task based on the critique.
Format: "Focus primarily on: [REL1], [REL2], [REL3]"

**Step 3: Relationship Extraction Principle**

State the core principle for relationship extraction (e.g., "Focus on causal chains", "Prioritize relationships that explain narrative progression")

**MUST INCLUDE**: Explicitly remind the model to extract ONLY relationships that match the specified types and to ignore all other relationship types.

═══════════════════════════════════════════════════════════════════
SECTION 4: EXTRACTION DIRECTIVE (Clear and Direct)
═══════════════════════════════════════════════════════════════════
Provide a clear, direct, unambiguous instruction that tells the model:

"When analyzing the text, extract ONLY and EXCLUSIVELY:
1. Entities of types: [list ALL the specified types from SECTION 2]
2. Relationships of types: [list ALL the specified types from SECTION 3]

**ABSOLUTE RESTRICTION - NO EXCEPTIONS:**
- Extract ONLY entities that match the specified entity types listed above
- Extract ONLY relationships that match the specified relationship types listed above
- DO NOT extract ANY entities or relationships that fall outside these specified categories
- Ignore all other entity types and relationship types, even if they appear highly important or central to the text
- The system is designed to work with these specific types ONLY - extracting other types will severely degrade performance
- DO NOT infer or create new entity/relationship types beyond those specified
- When in doubt, if an entity or relationship doesn't clearly fit the specified types, DO NOT extract it"

═══════════════════════════════════════════════════════════════════
SECTION 5: FEW-SHOT EXAMPLES (2-3 Examples)
═══════════════════════════════════════════════════════════════════
Provide 2-3 concrete examples that demonstrate proper extraction.

**CRITICAL**: All examples MUST use the exact tuple format with <|> delimiter. This is NOT optional.
**This applies to BOTH graph construction AND graph refinement** - the tuple format is mandatory in all cases.

For EACH example:
1. Provide a short text snippet (2-4 sentences)
2. Show extracted entities and relationships in TUPLE FORMAT using the <|> delimiter
3. EVERY entity and relationship in the example MUST be in tuple format - no other format is acceptable
4. Use the exact tuple schemas: ("entity"<|>name<|>type<|>description) and ("relationship"<|>type<|>source<|>target<|>description)

**CRITICAL RULES FOR EXAMPLES:**
- **MANDATORY FORMAT**: EVERY extracted entity and relationship in examples MUST use the exact tuple format with <|> delimiter
- Entity tuple format: ("entity"<|>name<|>type<|>description)
- Relationship tuple format: ("relationship"<|>type<|>source<|>target<|>description)
- DO show what entities and relationships to extract
- DO show how to describe entities and relationships
- DO use concrete, specific examples from realistic scenarios
- DO NOT use abstract placeholders like "[entity_name]" - use actual names
- DO NOT use any other format (no JSON, no bullet lists, no prose descriptions)
- Focus on demonstrating WHAT to extract and WHY those elements matter
- **IF Event is an entity type: At least ONE example MUST show an Event entity participating in a relationship with another entity** (e.g., Person PARTICIPATED_IN Event, Event CAUSED Event, Event OCCURRED_AT Location)

**MANDATORY Example Format - NO EXCEPTIONS:**
**Example [N]:**
Text: "[Your example text here - 2-4 sentences of realistic content]"

Extracted Output:
("entity"<|>EntityName1<|>EntityType<|>Description of this entity based on the text)
("entity"<|>EntityName2<|>EntityType<|>Description of this entity based on the text)
("relationship"<|>RELATIONSHIP_TYPE<|>EntityName1<|>EntityName2<|>Description of why these entities are related)

IMPORTANT: Every single entity and relationship in your examples MUST be shown in this exact tuple format. Do not deviate from this format.

**Special Note for Event Examples:**
If Event is an entity type, include at least one example like:
("entity"<|>The Battle of Midway<|>Event<|>A decisive naval battle in 1942)
("entity"<|>Admiral Yamamoto<|>Person<|>Japanese naval commander)
("relationship"<|>COMMANDED<|>Admiral Yamamoto<|>The Battle of Midway<|>Admiral Yamamoto commanded Japanese forces during the Battle of Midway)

═══════════════════════════════════════════════════════════════════

CRITICAL REQUIREMENTS:
1. Follow the 5-section structure EXACTLY
2. Provide concrete examples for EACH entity type you specify (at least 2-3 examples per type)
3. Provide concrete examples for EACH relationship type you specify (at least 2-3 examples per type)
4. **Make it ABSOLUTELY CLEAR that extraction is LIMITED to ONLY the specified types**:
   - Your prompt MUST explicitly forbid extraction of any entity or relationship types not in the specified list
   - Use strong, unambiguous language: "ONLY", "EXCLUSIVELY", "DO NOT extract other types"
   - State the consequence: extracting unspecified types degrades system performance
5. **SECTION 5 examples MUST use tuple format with <|> delimiter - MANDATORY, NO EXCEPTIONS**
   - **This applies to BOTH graph construction AND graph refinement - tuple format is ALWAYS required**
   - Every single entity MUST be: ("entity"<|>name<|>type<|>description)
   - Every single relationship MUST be: ("relationship"<|>type<|>source<|>target<|>description)
   - Do NOT use any other format (no JSON, no bullet lists, no prose)
6. Examples should use concrete entity names (not abstract placeholders) to demonstrate realistic extraction
7. Base all priorities on the critique provided
8. Be specific about which types to prioritize (don't say "all" or "any")
9. **If Event is an entity type, MUST include:**
   a) Explicit clarification that Events are entities (not relationships) and must be extracted as entity tuples
   b) At least ONE concrete example in SECTION 5 showing an Event entity participating in a relationship with another entity (in tuple format)

**FINAL CRITICAL REMINDER**:
Your optimized prompt must make it CRYSTAL CLEAR that the LLM should extract ONLY and EXCLUSIVELY entities and relationships of the specified types. This restriction must be stated explicitly in:
- SECTION 1 (Strategic Guidance)
- SECTION 2 Step 3 (Entity Extraction Principle)
- SECTION 3 Step 3 (Relationship Extraction Principle)
- SECTION 4 (Extraction Directive)

Use strong, unambiguous language throughout. The LLM must understand that extracting ANY entity or relationship types outside the specified list is strictly forbidden and will degrade system performance.

Provide only the optimized system prompt without additional commentary.
"""

community_summarizer_gradient_prompt = """
You are evaluating the prompt used for community summarization in a GraphRAG system.

Current community summarization prompt:
{current_prompt}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, determine if there is a problem with the community summarization prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a critique focusing on the specific issue.
If false, leave the critique empty.
"""

community_summarizer_prompt_optimizer = """
You are optimizing a system prompt for community summarization in a GraphRAG system.

The current critique of the community summarization process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better summarize graph communities. The system prompt should incorporate the feedback to improve summary quality, relevance, and informativeness.

The prompt should instruct the LLM to:
1. Identify key entities and their relationships within the community
2. Capture important thematic information
3. Provide sufficient detail for query answering
4. Maintain clarity and conciseness

Provide only the optimized system prompt without additional commentary.
"""

answer_generation_prompt_optimizer_vector = """
You are optimizing a prompt for answer generation in a RAG system, and you have to make it as much adherent as possible to the following critique.

The current critique of the answer generation process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better generate answers from retrieved vector information. The system prompt should incorporate the feedback to improve answer quality, relevance, and coherence.

Provide only the optimized system prompt without additional commentary.
"""

retrieval_planner_prompt_optimizer_vector = """
You are optimizing a system prompt for retrieval planning in a vector RAG system.

The current critique of the retrieval planning process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better plan vector retrieval strategies. The system prompt should incorporate the feedback to improve query refinement, search strategy, and information gathering.

IMPORTANT: The system prompt MUST emphasize that retrieval queries should be:
1. SHORT (3-6 words maximum)
2. KEYWORD-BASED (concrete terms likely to appear in actual document text)
3. CONTENT WORDS (specific names, places, objects, actions, or concrete concepts)
4. AVOID ABSTRACT TERMS (like "plot", "theme", "summary", "analyze")

The system prompt should instruct the model to create queries that will actually MATCH text in source documents, not abstract analytical queries.

Provide only the optimized system prompt without additional commentary.
"""

retrieval_summarizer_prompt_gradient_vector = """
You are evaluating the prompt used for chunk summarization in a VectorRAG system.
Each retrieved chunk of text is processed by a summarizer that generates a summary based on the prompt below.

═══════════════════════════════════════════════════════════════════
CURRENT RETRIEVAL SUMMARIZER PROMPT (System Prompt):
═══════════════════════════════════════════════════════════════════
{current_prompt}

═══════════════════════════════════════════════════════════════════
CRITIQUE FROM PREVIOUS COMPONENT (Retrieved Content):
═══════════════════════════════════════════════════════════════════
{previous_critique}

═══════════════════════════════════════════════════════════════════
DETAILED HISTORY (Question, Context, Answer, Evaluation per Iteration):
═══════════════════════════════════════════════════════════════════
{response_critique_history}

═══════════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════════
Based on the information above, determine if there is a problem with the retrieval summarizer prompt that needs to be fixed.

The summarizer prompt should guide the model to:
- Extract key information relevant to answering the query
- Create concise but informative summaries
- Maintain important details while removing noise

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a critique focusing on the specific issue.
If false, leave the critique empty.
"""

retrieval_summarizer_prompt_optimizer_vector = """
You are optimizing a system prompt for document summarization in a vector RAG system.

The current critique of the retrieval summarization process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better summarize retrieved documents. The system prompt should incorporate the feedback to improve summarization quality, relevance focus, and information extraction.

Provide only the optimized system prompt without additional commentary.
"""

hyperparameters_vector_agent_prompt_optimizer = """
You are optimizing a system prompt for hyperparameter selection in a vector RAG system.

The current critique of the hyperparameter selection process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better determine optimal chunk sizes for vector construction. The system prompt should incorporate the feedback to improve hyperparameter reasoning and selection.

Provide only the optimized system prompt without additional commentary.
"""

prompt_optimizer_prompt = """
Your goal is to optimize a prompt. The prompt is used for: {}

Below are the criticisms on the prompt:
{}

Incorporate the criticism, and produce a new prompt.
"""


# Community Level Selection for GraphRAG
# Controls which hierarchical levels of communities to use for summarization and answer generation
# Options:
#   "all"   - Use all communities at all levels
#   "top"   - Use only top-level communities (L0_*)
#   "leaf"  - Use only leaf communities (most specific, no children)
#   "0"     - Use only depth 0 communities (broadest)
#   "1"     - Use only depth 1 communities (second level - DEFAULT)
#   "2"     - Use only depth 2 communities (third level - most specific)
#   "0,1"   - Use depths 0 and 1
#   "1,2"   - Use depths 1 and 2
default_community_levels = "top"  # Default: second level (depth 1) provides good balance


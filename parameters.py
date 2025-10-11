from pydantic import BaseModel, Field
from typing import List, Union, Literal, Dict, Any, Optional


base_prompt_hyperparameters_graph = """
Your goal is to define the hyperparameters for a Graph RAG architecture.
You have to define a good chunk size. The chunk size must be in tokens. The chunk size must be only one (no variable chunk sizes).
The system will split the input text based on the chunk size. For each split, an LLM will be used to extract a graph (the graphs will then be merged). So, the chunk size determines the length of the text that must be processed by an LLM.

Text to analyze: {text}

Question: {question}

Please, suggest an appropriate chunk size and provide the reasoning that led to your response.
"""

base_prompt_hyperparameters_vector = """
Your goal is to define the hyperparameters for a Vector RAG architecture.
You have to define a good chunk size. The chunk size must be in tokens. The chunk size must be only one (no variable chunk sizes).
The system will split the input text based on the chunk size. For each split, an embedder will embed the split text (so the split text will be one point in the vector base). So, the chunk size determines the granularity of the embedded text chunks.

Text to analyze: {text}

Question: {question}

Please, suggest an appropriate chunk size and provide the reasoning that led to your response.
"""


class HyperparametersGraphResponse(BaseModel):
    reasoning: str
    chunk_size: int

class HyperparametersGraphResponse(BaseModel):
    reasoning: str
    chunk_size: int

base_prompt_graph_builder = """
You will be given a text. Your goal is to identify entities in the text and all the relationships among the identified entities.
For each entity, you will include:
- name: the entity name
- type: the entity type (e.g., Person, Organization, Location, Event, Concept)
- properties: a list of key-value pairs describing characteristics of the entity extracted from the text (e.g., for a person: age, role, description; for a location: description, significance). Each property should have a "key" and "value" field.

For each relationship, you will include its type, a description (why you think the two entities are related to each other), and the evidence from the text that supports this.
The relationships must be among the extracted entities.
Provide a list of triplets in your answer.

Return no more than 20 entities and 30 relationships. 

Text:
{}

Provide the reasoning that led to your response.
"""


base_prompt_graph_refinement = """
You will be given a text. Your goal is to extract entities and relationships from the text to enhance the knowledge graph by:
1. Identifying new entities mentioned in the text
2. Extracting properties for entities based on information in the text
3. Identifying relationships between entities
4. Providing evidence from the text for each relationship

For each entity, you will include:
- name: the entity name
- type: the entity type (e.g., Person, Organization, Location, Event, Concept)
- properties: a list of key-value pairs describing characteristics of the entity extracted from the text. Each property should have a "key" and "value" field.

For each relationship, you will include its type, a description, and the evidence from the text that supports this.
The relationships must be among the extracted entities.
Provide a list of triplets for relationships in your answer.

Text to analyze:
{}

Extract all relevant entities and relationships from the text.

Return no more than 20 entities and 30 relationships.

Provide the reasoning that led to your response.
"""


class EntityProperty(BaseModel):
    model_config = {"extra": "forbid"}
    key: str
    value: str

class Entity(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    type: str
    properties: List[EntityProperty] = Field(default_factory=list)


class Relationship(BaseModel):
    model_config = {"extra": "forbid"}
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    evidence: str


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


class EntityUpdate(BaseModel):
    model_config = {"extra": "forbid"}
    entity_name: str
    property_key: str
    property_value: str

class GraphRefinementResponse(BaseModel):
    model_config = {"extra": "forbid"}
    new_entities: List[Entity]
    new_relationships: List[Relationship]
    new_triplets: List[Triplet]
    entity_property_updates: List[EntityUpdate] = Field(default_factory=list)
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

Pay special attention to the LENGTH GUIDELINES in the learned patterns:
- Answers must meet the minimum word count for their question type
- Descriptive/significance questions require 150-450 words
- Brief answers that lack comprehensiveness should be marked NEEDS_REFINEMENT
- If an answer seems too short or lacks detail, it likely needs refinement

3. If NEEDS_REFINEMENT, provide a specific, actionable critique explaining:
   - What is missing or incorrect
   - What aspects of the gold patterns are not followed
   - How to improve the answer

Output format:
DECISION: [SATISFACTORY or NEEDS_REFINEMENT]
CRITIQUE: [Your detailed critique if refinement is needed, or "None" if satisfactory]"""


class ResponseEvaluationResponse(BaseModel):
    reasoning: str
    continue_optimization: bool = Field(alias="continue")
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
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue.
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

Based on this information, provide a SHORT and CONCISE critique (2-3 sentences max) of how the retrieved content can be improved.
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

Based on this information, provide a SHORT and CONCISE critique (2-3 sentences max) of how the community selection can be improved.
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
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue.
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

Based on this information, provide a SHORT and CONCISE critique (2-3 sentences max) of how the graph can be improved.
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
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue.
If false, leave the critique empty.
"""


rag_hyperparameters_agent_gradient_prompt = """
You are evaluating the prompt used for hyperparameter (chunk size) selection in a GraphRAG system.

Current hyperparameters selection prompt:
{current_prompt}

Critique from the previous component:
{previous_critique}

Response evaluator output:
{response_evaluator_output}

Based on this information, determine if there is a problem with the hyperparameters selection prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue.
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
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue with the chunk size.
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
You are optimizing a system prompt for graph construction in a GraphRAG system.

The current critique of the graph building process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better extract entities and relationships from text. The system prompt should incorporate the feedback to improve entity recognition, relationship extraction, and graph structure.

Provide only the optimized system prompt without additional commentary.
"""

hyperparameters_graph_agent_prompt_optimizer = """
You are optimizing a system prompt for hyperparameter selection in a GraphRAG system.

The current critique of the hyperparameter selection process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better determine optimal chunk sizes for graph construction. The system prompt should incorporate the feedback to improve hyperparameter reasoning and selection.

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
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue.
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
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue.
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


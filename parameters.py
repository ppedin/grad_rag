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
You are an agentic retrieval component of a RAG system. Your goal is to determine if you have enough information to answer the query, or if you need to retrieve more information.

Query to answer: {}

Current context from previous retrievals:
{}

Previous retrieval decisions in this session:
{}

First, provide your reasoning about whether the current context contains sufficient information to answer the query comprehensively.

Then, set information_sufficient to true or false:
- true: The current context contains enough information to provide a comprehensive answer
- false: More information is needed from the knowledge base

If information_sufficient is false, provide a new query that will retrieve complementary information. Make sure your new query is different from previous queries to gather new perspectives or details.
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
    reasoning: str
    information_sufficient: bool  # True if current context has enough info, False if more retrieval needed
    query: str  # Only used if information_sufficient=False

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

base_prompt_answer_generator_vector = """
You will be given a query and the information retrieved by a RAG system.
Your goal is to use the retrieved context to answer the query.

This is the query:
{question}

This is the information:
{retrieved_context}

Provide an answer to the query.

Use the following critique:
{critique}
"""

response_evaluator_prompt = """
You are evaluating the response generated by a RAG system.

{satisfactory_criteria}


Don't demand perfection; "good enough" is acceptable


The query was:
{original_query}

The response was:
{generated_answer}

{previous_evaluations}

First, assess whether the response is satisfactory. Don't demand perfection; "good enough" is acceptable. Be pragmatic in your evaluation. If there are previous evaluations, consider whether the response has improved over iterations.


If the response needs improvement, provide a detailed explanation of how it can be improved and set "continue" to true. Remember: if information is missing, it means the pipeline failed to extract or retrieve it properly, so improvement is necessary.

Your response must include:
1. "reasoning": Explain why you decided the response is satisfactory or needs improvement
2. "continue": A boolean field (true if improvement needed, false if response is satisfactory)
3. "critique": A critique explaining what needs improvement (can be empty if continue is false). 
4. "missing_keywords": A list of important entities, relationships, concepts, or terms that are crucial to answer the query but are missing or have insufficient information in the current response. These keywords will be used to search for relevant text in the source document, so they should be simple, specific terms that actually appear in the text rather than descriptive phrases. (empty list if continue is false) (max 4-5 keywords)

IMPORTANT GUIDELINES FOR missing_keywords:

WORD LIMIT: Each keyword must be 1-2 words MAXIMUM
- Single words are best (e.g., "budget", "workshop")
- Person names can be 2-3 words (e.g., "Karl Von Mark", "John Smith")
- Multi-word technical terms only if they're atomic concepts (e.g., "knowledge graph")

DECOMPOSITION: If a concept requires multiple elements, split them into SEPARATE keywords
- Instead of one complex phrase, use multiple simple keywords
- BAD: "Karl Von Mark plan execution" → GOOD: ["Karl Von Mark", "plan", "execution"]
- BAD: "meeting discussion about budget" → GOOD: ["meeting", "budget", "discussion"]
- BAD: "collaboration between X and Y regarding Z" → GOOD: ["X", "Y", "Z", "collaboration"]

SIMPLE TERMS that actually appear in the text:
- Use terms exactly as they appear in the source (e.g., "Professor C", "workshop", "budget")
- AVOID descriptive phrases (NOT "Professor C's statements about the workshop")
- AVOID possessives and complex noun phrases (NOT "John's opinion on X")
- For people: use just their name/title (e.g., "Professor C", "John Smith")
- For concepts: use the core term (e.g., "budget allocation" → ["budget", "allocation"])
- For relationships: use entity names separately (e.g., "partnership between X and Y" → ["X", "Y", "partnership"])

Examples of GOOD missing_keywords:
- Simple entity names: "SmartKom", "John Smith", "Microsoft", "Professor C", "Karl Von Mark"
- Simple concepts: "budget", "privacy", "workshop", "evaluation", "plan", "execution"
- Simple relationship terms: "partnership", "collaboration", "manages"

Examples of BAD missing_keywords (avoid these):
- "Karl Von Mark plan execution" (too complex, use ["Karl Von Mark", "plan", "execution"])
- "Professor C's statements about the workshop" (too descriptive, use ["Professor C", "workshop", "statements"])
- "John's opinion on the project" (too complex, use ["John", "project", "opinion"])
- "the connection between X and Y" (too descriptive, use ["X", "Y", "connection"])
- "meeting discussion about budget" (too complex, use ["meeting", "budget", "discussion"])

IMPORTANT: if the response acknowledges that the query cannot be addressed based on the information in the context, then this is a serious retrieval issue, so the response cannot be considered good and satisfactory.
"""


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

History of generated responses and critiques for this QA pair:
{response_critique_history}

Based on this information, determine if there is a problem with the answer generation prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue.
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

Retrieved content:
{retrieved_content}

Critique from the previous component:
{previous_critique}

History of generated responses and critiques for this QA pair:
{response_critique_history}

Based on this information, provide a detailed critique of how the retrieved content can be improved.
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

Retrieval plan:
{retrieval_plan}

Critique from the previous component:
{previous_critique}

History of generated responses and critiques for this QA pair:
{response_critique_history}

Based on this information, provide a detailed critique of how the retrieval plan can be improved.
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

Current retrieval planning prompt:
{current_prompt}

Critique from the previous component:
{previous_critique}

History of generated responses and critiques for this QA pair:
{response_critique_history}

Based on this information, determine if there is a problem with the retrieval planning prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a SHORT and CONCISE critique (2-3 sentences max) focusing on the specific issue.
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
You are optimizing a system prompt for answer generation in a vector RAG system.

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

Provide only the optimized system prompt without additional commentary.
"""

retrieval_summarizer_prompt_gradient_vector = """
You are evaluating the prompt used for chunk summarization in a VectorRAG system. Each chunk of text is given to the summarizer that generates a summary.

Current retrieval summarizer prompt:
{current_prompt}

Critique from the previous component:
{previous_critique}

History of generated responses and critiques for this QA pair:
{response_critique_history}

Based on this information, determine if there is a problem with the retrieval summarizer prompt that needs to be fixed.

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


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
Your goal is to define the hyperparameters for a vector RAG architecture.
You have to define a good chunk size. The chunk size must be in tokens. The chunk size must be only one (no variable chunk sizes).
The system will split the input text based on the chunk size. For each split, an embedder will embed the split text (so the split text will be one point in the vector base).

Text: {text}
Question: {question}

Please, suggest an appropriate chunk size for this text and question.
Use the following critique:
{critique}

Provide the reasoning that led to your response.
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
You will be given a text and an existing knowledge graph. Your goal is to refine and enhance the existing graph by:
1. Adding new entities and relationships that are mentioned in the text but missing from the graph
2. Adding new properties to existing entities based on information in the text
3. Creating new relationships between existing entities that were previously unconnected
4. Updating or enriching existing entity properties with additional information from the text

For each new entity, you will include:
- name: the entity name
- type: the entity type (e.g., Person, Organization, Location, Event, Concept)
- properties: a list of key-value pairs describing characteristics of the entity extracted from the text. Each property should have a "key" and "value" field.

For each new relationship, you will include its type, a description, and the evidence from the text that supports this.
The relationships must be among the entities (both existing and new).
Provide a list of triplets for new relationships in your answer.

Existing Graph Summary:
{}

Text to analyze for refinement:
{}

Focus on finding new information not already captured in the existing graph. Only add entities, relationships, and properties that provide new value.

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
You are an agentic retrieval component of a RAG system. Your goal is to refine the query to retrieve relevant information from the knowledge base to answer the following query: {}.

Previous retrieval decisions in this session:
{}

IMPORTANT: Review the previous decisions above to avoid repeating the same queries. Choose a query that will retrieve complementary information to build upon what you have already gathered.

Choose a new query and provide the reasoning that led to your response.
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
    selected_communities: List[int]  # List of community IDs to retrieve
    reasoning: str  # Explanation of the community selection

class VectorRetrievalPlannerResponse(BaseModel):
    model_config = {"extra": "forbid"}
    query: str
    reasoning: str

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

EVALUATION OBJECTIVE: Maximize ROUGE F1-score, which balances precision (conciseness) and recall (completeness).
- Precision: Avoid unnecessary words and redundancy
- Recall: Include all key information from the reference

The query was:
{original_query}

The response was:
{generated_answer}

The ROUGE F1 score achieved: {rouge_score}


Provide a detailed explanation of how the response can be improved to maximize ROUGE F1-score.
Remember that the ROUGE F1 score considers both precision (give only information that is required), and recall (give all the information required). 
"""



generation_prompt_gradient_prompt = """
You are evaluating the prompt used for answer generation in a Graph RAG system during test-time training.

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text is split into chunks and graphs are extracted and merged. The merged graph undergoes community detection to identify clusters of related entities. Each community is summarized with a title and description using an LLM. For retrieval, an LLM agent is presented with all community titles and selects the most relevant communities to answer the query. The selected community summaries are then passed to an answer generation LLM.

Here is the current example with prompt, answer, and feedback:
{}

Based on this single example, provide a detailed critique that will be used to improve the answer generation prompt. Focus on specific issues identified in the feedback and how the prompt could be modified to address them.
"""

generation_prompt_gradient_prompt_vector = """
You are evaluating the prompt used for answer generation in a VectorRAG system during test-time training.

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

Here is the current example with system prompt, user prompt, query, answer, and feedback:
{}

Based on this single example, provide a detailed critique that will be used to improve the answer generation prompt. Focus on specific issues identified in the feedback and how the prompt could be modified to address them.
"""


retrieved_content_gradient_prompt_graph = """
You are evaluating the content retrieved by a GraphRAG system during test-time training.

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text is split into chunks and graphs are extracted and merged. The merged graph undergoes community detection to identify clusters of related entities. Each community is summarized with a title and description using an LLM. For retrieval, an LLM agent is presented with all community titles and selects the most relevant communities to answer the query. The selected community summaries are then passed to an answer generation LLM.

Here is the current example with retrieved context, query, answer, and feedback:
{}

Based on these contents, provide a detailed critique of how the retrieved content can be improved.
"""

retrieved_content_gradient_prompt_vector = """
You are evaluating the content retrieved by a VectorRAG system during test-time training.

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

Here is the current example with retrieved context, query, answer, and feedback:
{}

Based on this single example, provide a detailed critique of how the retrieved content can be improved.
"""


retrieval_plan_gradient_prompt_graph = """
You are evaluating the community selection made by an agentic community-based GraphRAG system.

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text is split into chunks and graphs are extracted and merged. The merged graph undergoes community detection to identify clusters of related entities. Each community is summarized with a title and description using an LLM. For retrieval, an LLM agent is presented with all community titles and selects the most relevant communities to answer the query. The selected community summaries are then passed to an answer generation LLM.

Here are some examples of community selections, with the retrieved community summaries:
{}

The critique on the retrieved content was:
{}

Based on these contents, provide a detailed critique of how the community selection can be improved.
"""


retrieval_plan_gradient_prompt_vector = """
You are evaluating the retrieval plan made by an agentic VectorRAG system during test-time training.

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

Here is the current example with retrieval plan and retrieved content:
{}

The critique on the retrieved content was:
{}

Based on this single example, provide a detailed critique of how the retrieval plan can be improved.
"""


retrieval_planning_prompt_gradient_prompt = """
You are evaluating the prompt used for community selection in an agentic community-based GraphRAG system.

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text is split into chunks and graphs are extracted and merged. The merged graph undergoes community detection to identify clusters of related entities. Each community is summarized with a title and description using an LLM. For retrieval, an LLM agent is presented with all community titles and selects the most relevant communities to answer the query. The selected community summaries are then passed to an answer generation LLM.

These are the community selection prompt and a high-level description of the available communities, with the community selections made:
{}

The critique on the community selections was:
{}

Based on these contents, provide a detailed critique that will be used to improve the community selection prompt.  
"""


retrieval_planning_prompt_gradient_vector = """
You are evaluating the prompt used for retrieval planning in an agentic GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

These are the retrieval planning prompt, with the retrieval plans generated:
{}

The critique on the retrieval plans was:
{}

Based on these contents, provide a detailed critique that will be used to improve the retrieval planning prompt.  
"""


graph_gradient_prompt = """
You are evaluating a graph that has been automatically built for a GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text is split into chunks and graphs are extracted and merged. The merged graph undergoes community detection to identify clusters of related entities. Each community is summarized with a title and description using an LLM. For retrieval, an LLM agent is presented with all community titles and selects the most relevant communities to answer the query. The selected community summaries are then passed to an answer generation LLM.

Here are the queries, the graph and the retrieval plans generated by the agent. 
{}

The critique on the retrieval plans was:
{}

Based on these contents, provide a detailed critique of how the graph can be improved. 
"""


graph_extraction_prompt_gradient_prompt = """
You are evaluating the prompt used for graph construction in an agentic GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text is split into chunks and graphs are extracted and merged. The merged graph undergoes community detection to identify clusters of related entities. Each community is summarized with a title and description using an LLM. For retrieval, an LLM agent is presented with all community titles and selects the most relevant communities to answer the query. The selected community summaries are then passed to an answer generation LLM.

The prompt was the following:
{}

A sample from the corpus used to build the graph is the following:
{}

A high-level description of the obtained graph was the following:
{}

The critique on the graph was: 
{}

Based on these contents, provide a detailed critique that will be used to improve the graph construction prompt. 
"""


rag_hyperparameters_agent_gradient_prompt = """
You are evaluating the choice of the chunk size for a GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text is split into chunks and graphs are extracted and merged. The merged graph undergoes community detection to identify clusters of related entities. Each community is summarized with a title and description using an LLM. For retrieval, an LLM agent is presented with all community titles and selects the most relevant communities to answer the query. The selected community summaries are then passed to an answer generation LLM.

The chosen chunk size is the following:
{}

A sample from the corpus used to build the graph is the following:
{}

A high-level description of the obtained graph was the following:
{}

The critique on the graph was:
{}

Based on these contents, provide a detailed critique of how the chunk size can be improved. 
"""


rag_hyperparameters_agent_gradient_vector = """
You are evaluating the choice of the chunk size for a VectorRAG system during test-time training.

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

The chosen chunk size for this example was:
{}

Here is the current example with query, retrieval prompt, and retrieval plan generated by the agent:
{}

Based on this single example, provide a detailed critique of how the chunk size can be improved.
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


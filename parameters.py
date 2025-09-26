from pydantic import BaseModel
from typing import List, Union, Literal


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

Please, suggest an appropriate chunk size. 
Use the following critique:
{}

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
For each entity and relationship, you will include its type in the response.
The relationships must be among the extracted entities. For each relationship, include a description (why you think the two entities are related to each other), and the evidence from the text that supports this.
Provide a list of triplets in your answer.

Text:
{}

Provide the reasoning that led to your response.
"""


class Entity(BaseModel):
    name: str
    type: str


class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    evidence: str


class Triplet(BaseModel):
    subject: str
    predicate: str
    object: str


class GraphBuilderResponse(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    triplets: List[Triplet]
    reasoning: str


base_prompt_graph_retrieval_planner = """
Your goal is to decide the next step of a strategy to explore a graph in order to retrieve relevant information to answer the following query: {}.

A high-level description of the graph is the following: {}

You must choose one of the following functions:

- search_nodes_by_keyword(keyword): search for all the nodes whose labels contain the given keyword
- search_nodes_by_types(node_type): search for all the nodes whose type property contains the given type
- get_neighbors(node_name): get all neighbors of a node with the given name
- search_relations_by_type(relation_type): search for all the triplets whose relationship matches the type
- identify_communities(node_name): find the community (connected component) containing a specific node
- analyze_path(start_node_name, end_node_name): find the shortest path between two nodes
- find_hub_nodes: find the top 3 hub nodes with the highest connectivity

The subgraphs you retrieved so far are the following:

{}

Choose one of the functions and specify the arguments.

Provide the reasoning that led to your response.

Pay attention to symbols included in the entity/relationship type names: make sure to include them in your search for matching to succeed.
Also, pay attention to symbols included in the functions names. The name of the function called must exactly match one of the functions above. 
"""

base_prompt_vector_retrieval_planner = """
You are an agentic retrieval component of a RAG system. Your goal is to refine the query to retrieve relevant information from the knowledge base to answer the following query: {}.

The content you retrieved so far is the following:
{}

Choose a new query. 

Use the following critique:
{}

Provide the reasoning that led to your response. 
"""


class SearchNodesByKeywordCall(BaseModel):
    function_name: Literal["search_nodes_by_keyword"] = "search_nodes_by_keyword"
    keyword: str

class SearchNodesByTypesCall(BaseModel):
    function_name: Literal["search_nodes_by_types"] = "search_nodes_by_types"
    node_type: str

class GetNeighborsCall(BaseModel):
    function_name: Literal["get_neighbors"] = "get_neighbors"
    node_name: str

class SearchRelationsByTypeCall(BaseModel):
    function_name: Literal["search_relations_by_type"] = "search_relations_by_type"
    relation_type: str

class IdentifyCommunitiesCall(BaseModel):
    function_name: Literal["identify_communities"] = "identify_communities"
    node_name: str

class AnalyzePathCall(BaseModel):
    function_name: Literal["analyze_path"] = "analyze_path"
    start_node_name: str
    end_node_name: str

class FindHubNodesCall(BaseModel):
    function_name: Literal["find_hub_nodes"] = "find_hub_nodes"


# Union type for all possible function calls
FunctionCall = Union[
    SearchNodesByKeywordCall,
    SearchNodesByTypesCall,
    GetNeighborsCall,
    SearchRelationsByTypeCall,
    IdentifyCommunitiesCall,
    AnalyzePathCall,
    FindHubNodesCall
]


class GraphRetrievalPlannerResponse(BaseModel):
    function_call: FunctionCall
    reasoning: str

class VectorRetrievalPlannerResponse(BaseModel):
    query: str
    reasoning: str

base_prompt_answer_generator_graph = """
You will be given a query and the information retrieved from a graph.
Your goal is to use the retrieved context to answer the query.

This is the query:
{}

This is the information:
{}

Provide an answer to the query.
"""

base_prompt_answer_generator_vector = """
You will be given a query and the information retrieved by a RAG system. 
Your goal is to use the retrieved context to answer the query. 

This is the query:
{}

This is the information: 
{}

Provide an answer to the query. 

Use the following critique: 
{}
"""


response_evaluator_prompt = """
You are evaluating the response generated by a RAG system. 

The query was:
{}

The response was:
{}

The gold response is:
{}

The ROUGE score between the response and the gold response is:
{}

Provide a detailed explanation of how the response can be improved. 
"""


generation_prompt_gradient_prompt = """
You are evaluating the prompt used for answer generation in a Graph RAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. For each chunk, an LLM is used to extract a graph. The graphs are then merged. An LLM agent has access to a set of functions to explore the graph (the set is fixed and includes queries on entities and relations based on type or name, access to neighbors, community detection, and identification of central nodes). The context obtained from the exploration is then passed to an LLM together with the query. The LLM generates an answer.

Here are some examples of prompts and answers, with feedback of how the response can be improved:
{}

Based on these contents, provide a detailed critique that will be used to improve the answer generation prompt. 
"""

generation_prompt_gradient_prompt_vector = """
You are evaluating the prompt used for answer generation in a RAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

Here are some examples of prompts and answers, with feedback of how the response can be improved:
{}

Based on these contents, provide a detailed critique that will be used to improve the answer generation prompt. 
"""


retrieved_content_gradient_prompt_graph = """
You are evaluating the content retrieved by a GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. For each chunk, an LLM is used to extract a graph. The graphs are then merged. An LLM agent has access to a set of functions to explore the graph (the set is fixed and includes queries on entities and relations based on type or name, access to neighbors, community detection, and identification of central nodes). The context obtained from the exploration is then passed to an LLM together with the query. The LLM generates an answer.

Here are some examples of retrieved contexts, queries, and answers, with feedback of how the response can be improved. 
{}

Based on these contents, provide a detailed critique of how the retrieved content can be improved.
"""

retrieved_content_gradient_prompt_vector = """
You are evaluating the content retrieved by a RAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

Here are some examples of retrieved contexts, queries, and answers, with feedback of how the response can be improved. 
{}

Based on these contents, provide a detailed critique of how the retrieved content can be improved.
"""


retrieval_plan_gradient_prompt_graph = """
You are evaluating the retrieval plan made by an agentic GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. For each chunk, an LLM is used to extract a graph. The graphs are then merged. An LLM agent has access to a set of functions to explore the graph (the set is fixed and includes queries on entities and relations based on type or name, access to neighbors, community detection, and identification of central nodes). The context obtained from the exploration is then passed to an LLM together with the query. The LLM generates an answer.

Here are some examples of retrieval plans, with the retrieved contents:
{}

The critique on the retrieved content was:
{}

Based on these contents, provide a detailed critique of how the retrieval plan can be improved. 
"""


retrieval_plan_gradient_prompt_vector = """
You are evaluating the retrieval plan made by an agentic GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

Here are some examples of retrieval plans, with the retrieved contents:
{}

The critique on the retrieved content was:
{}

Based on these contents, provide a detailed critique of how the retrieval plan can be improved. 
"""


retrieval_planning_prompt_gradient_prompt = """
You are evaluating the prompt used for retrieval planning in an agentic GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. For each chunk, an LLM is used to extract a graph. The graphs are then merged. An LLM agent has access to a set of functions to explore the graph (the set is fixed and includes queries on entities and relations based on type or name, access to neighbors, community detection, and identification of central nodes). The context obtained from the exploration is then passed to an LLM together with the query. The LLM generates an answer.

These are the retrieval planning prompt and a high-level description of the graph, with the retrieval plans generated:
{}

The critique on the retrieval plans was:
{}

Based on these contents, provide a detailed critique that will be used to improve the retrieval planning prompt.  
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
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. For each chunk, an LLM is used to extract a graph. The graphs are then merged. An LLM agent has access to a set of functions to explore the graph (the set is fixed and includes queries on entities and relations based on type or name, access to neighbors, community detection, and identification of central nodes). The context obtained from the exploration is then passed to an LLM together with the query. The LLM generates an answer.

Here are the queries, the graph and the retrieval plans generated by the agent. 
{}

The critique on the retrieval plans was:
{}

Based on these contents, provide a detailed critique of how the graph can be improved. 
"""


graph_extraction_prompt_gradient_prompt = """
You are evaluating the prompt used for graph construction in an agentic GraphRAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. For each chunk, an LLM is used to extract a graph. The graphs are then merged. An LLM agent has access to a set of functions to explore the graph (the set is fixed and includes queries on entities and relations based on type or name, access to neighbors, community detection, and identification of central nodes). The context obtained from the exploration is then passed to an LLM together with the query. The LLM generates an answer.

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
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. For each chunk, an LLM is used to extract a graph. The graphs are then merged. An LLM agent has access to a set of functions to explore the graph (the set is fixed and includes queries on entities and relations based on type or name, access to neighbors, community detection, and identification of central nodes). The context obtained from the exploration is then passed to an LLM together with the query. The LLM generates an answer.

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
You are evaluating the choice of the chunk size for a RAG system. 

The system works this way:
The system starts from an input text and a fixed query. An LLM selects the hyperparameters (chunk size). The input text (where the information is located) is split into chunks based on the chosen chunk size. Each chunk is embedded. An LLM agent refines the queries to retrieve the content. The context obtained is then passed to an LLM together with the query. The LLM generates an answer.

The chosen chunk size was the following:
{}

Here are the queries, the retrieval prompts, and the retrieval plans generated by the agent.

Based on these contents, provide a detailed critique of how the chunk size can be improved. 
"""

answer_generation_prompt_optimizer = """
You are optimizing a system prompt for answer generation in a GraphRAG system.

The current critique of the answer generation process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better generate answers from retrieved graph information. The system prompt should incorporate the feedback to improve answer quality, relevance, and coherence.

Provide only the optimized system prompt without additional commentary.
"""

retrieval_planner_prompt_optimizer = """
You are optimizing a system prompt for retrieval planning in a GraphRAG system.

The current critique of the retrieval planning process is:
{}

Based on this critique, generate a new system prompt that will be used to instruct the LLM how to better plan graph retrieval strategies. The system prompt should incorporate the feedback to improve retrieval strategy, function selection, and information gathering.

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


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


# Graph builder prompt - Instruction section (can be replaced by learned prompt)
base_prompt_graph_builder_instruction = """
# Goal
You are an expert knowledge graph builder. Your task is to extract entities and relationships from the provided text and structure them into a knowledge graph.

# Steps

1. **Identify Entities**: Read through the text and identify all significant entities.

2. **Classify Entity Types**: Assign each entity to an appropriate type category.

3. **Write Entity Descriptions**: For each entity, write a concise description that captures its key characteristics based on the text.

4. **Extract Relationships**: Identify meaningful relationships between the entities you've extracted.

5. **Create Triplets**: Express each relationship as a (subject, predicate, object) triplet.


# Constraints
- Extract NO MORE than 20 entities and 30 relationships

# Text to Analyze

{}
"""

# Graph builder prompt - Format section (always stays the same)
base_prompt_graph_builder_format = """
# Previous Graph Data

{previous_graph_data}

# Output Format

#######Guidelines for detailed, self-contained descriptions:######

Write clear, standalone descriptions that make sense without knowing the text.

Avoid vague references ("he," "it," "this event") — always specify who, what, and why.

For entities, mention their role, action, or significance in 1–2 sentences.

For relationships, explain why the link exists and what it shows or causes (1 sentence).

Keep descriptions concise but informative enough for an external reader to understand their narrative function.

Esempio minimo da includere nel prompt:

("entity"|The Butcher|Person|"A rebellious child who wishes to restore warfare in a peaceful future society.")
("relationship"|CAUSES|Self-jabbing|Pain reaction|"The Butcher jabs his hand with a metal tube, causing himself pain.")


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
- Include clear descriptions!
- Return ONLY the tuples, no additional text or explanations
"""

# Combined base prompt (for backward compatibility and initial use)
base_prompt_graph_builder = base_prompt_graph_builder_instruction + base_prompt_graph_builder_format


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
    first_seen_iteration: int = 0  # Track when entity was first discovered
    last_updated_iteration: int = 0  # Track when entity was last updated


class Relationship(BaseModel):
    model_config = {"extra": "forbid"}
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    evidence: str = ""  # Optional, defaults to empty string for tuple format
    first_seen_iteration: int = 0  # Track when relationship was first discovered
    last_updated_iteration: int = 0  # Track when relationship was last updated


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
You are a query agent in a rag system and your goal is to generate sub-queries to retrieve context useful to answer the question, based on the question, the context retrieved so far, and the previous queries.
Avoid retrieving previous sub-queries.
You need to generate two things to help retrieve information for answering the question:

1. A retrieval sub-query (avoid previous sub-queries)
2. A hypothetical document (~150 words) that represents what an ideal retrieved passage might look like given the retrieval query you chose.

For the hypothetical document:
- Write it as if it were an actual excerpt from a document
- REPRODUCE THE NARRATIVE STYLE of the story/text you are retrieving from
- Match the tone, voice, and writing style of the original text
- Take inspiration from the text already retrieved (if any)
- Make it concrete and specific (not abstract or generic)
- Focus on information that would help answer the question
- Generate approximately 150 words

Question to answer:
{}

Retrieved summaries so far:
{}

Previous queries you made:
{}

AVOID REPEATING PREVIOUS QUERIES YOU MADE. 
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
    hypothetical_document: str  # Hypothetical document for HyDE retrieval

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

answer_generator_initial_prompt = "Provide an answer to the question based on the context."

answer_generator_refinement_prompt = "Provide a concise, factual answer."

# Backward compatibility: keep this for any code that references it
base_prompt_answer_generator_vector = "Provide a concise, factual answer."

response_evaluator_prompt = """You are an expert judge evaluating both the **retrieved context** and the **generated answer** for completeness, coherence, and overall quality.

Your task:
1. **Context Evaluation:**  
   - First, imagine an *ideal, fully satisfying answer* to the question — one that captures all key facts, causal and temporal connections, character motivations, outcomes, and thematic meaning.  
   - Then compare this ideal answer with the retrieved context. Identify what information, links, or nuances are **missing, underdeveloped, ambiguous, or irrelevant**.  
   - Note any missing elements that would prevent a reader from forming a complete understanding or that weaken the logical or emotional flow of the narrative.

2. **Answer Evaluation:**  
   - Assess whether the answer uses the available context effectively.  
   - Evaluate **clarity, coherence, factual accuracy, completeness, and narrative flow**.  
   - Check if it connects events and entities logically, avoids contradictions, and reads fluently in tone and length.

**Decision:**  
- **SATISFACTORY:** The context includes all major information and connections needed for an ideal answer, and the answer expresses them clearly and coherently.  
- **NEEDS_REFINEMENT:** Any important fact, relation, or narrative link is missing, unclear, or the answer lacks fluency, structure, or accuracy.

**Critique must include:**  
1. What an *ideal answer* would contain that is not fully supported by the context.  
2. Which parts of the context are irrelevant or insufficient.  
3. How the answer could improve in factual precision, logical or causal connections, completeness, and stylistic quality.

**Output format:**  
DECISION: [SATISFACTORY or NEEDS_REFINEMENT]  
CRITIQUE: [detailed explanation]

The question was the following:
{original_query}


The answer of the system was:
{generated_answer}


When generating the answer, the system had access to the following context:
{retrieved_context}
"""


# GraphRAG-specific response evaluator prompt with knowledge graph refinement context
response_evaluator_prompt_graph = """You are an expert judge evaluating the answer of a GraphRAG system for query-based summarization.

1. Reason in the following way:
CONSIDER ONLY THE COMMUNITY CONTEXT TO ANSWER THE FOLLOWING TWO QUESTIONS:
- Which information would be needed to answer the question in a fully satisfying way? (CONSIDER AN IDEAL ANSWER TO THE QUESTION, NOT THE ANSWER YOU'RE SEEING)
- Which of this information (THE ONE NEEDED FOR AN IDEAL ANSWER) is not present in the context? Are there things that are missing or underdeveloped in the community summaries provided in the contexts? Which type of information should the community contexts capture more?
FOCUS NOT ON THE SPECIFIC INFORMATION, BUT RATHER ON THE TYPE OF INFORMATION LACKING (IF YOU WERE TO BUILD A GRAPH, WHAT ENTITIES/RELATIONSHIPS WOULD YOU INCLUDE?)
PROVIDE A RICH SET OF ENTITIES/RELATIONSHIPS

- ONLY NOW, LOOK AT THE ANSWER: Is the answer satisfactory in terms of style (sufficient length, coherence, fluency, etc.)?

2. Decide if the system output is:
   - SATISFACTORY: Both the COMMUNITY SUMMARIES and the ANSWERS are okay (this means that the community summaries have all the necessary information to an ideal answer, and the answer's style is fine)
   - NEEDS_REFINEMENT: The output has issues and should be improved (graph extraction AND answer generation will be refined)

3. If NEEDS_REFINEMENT, provide a specific, actionable critique explaining:
- Limitations of the community summaries (which type of information should be included, which should be omitted because it's not necessary to answer the question) and the answer
- Actionable insights on how to improve the answer (which aspects to include, which style to use) (this is for the answer phase)

Output format (follow this EXACTLY):
DECISION: [SATISFACTORY or NEEDS_REFINEMENT]
CRITIQUE: [Your detailed critique if refinement is needed. If SATISFACTORY, briefly explain why (e.g., answer is complete and well-written).]

The question was:
{original_query}

The answer of the system was:
{generated_answer}

When generating the answer, the system had access to the following context:
{community_context}
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

base_prompt_community_summarizer_query_focused = """
You are given a description of a community (subgraph) from a knowledge graph. Your task is to create a title and discursive summary that highlights the key entities and how they interact, with a focus on information relevant to answering a specific query.

Query to answer:
{query}

Subgraph description:
{subgraph_description}

Please provide:
1. A short DESCRIPTION (3-4 rows) that captures the entities/relationships of the community and how they interact
2. A detailed SUMMARY that focuses on information relevant to the query while maintaining all important details from the graph

IMPORTANT: Do not lose detail. Include all entities and relationships, but emphasize those most relevant to answering the query.

Format your response as:
DESCRIPTION: [your short description here]
SUMMARY: [your detailed summary here, focused on query-relevant information]
"""


generation_prompt_gradient_prompt = """
You are evaluating the prompt used for answer generation in a GraphRAG system.
Your goal is to evaluate the prompt and propose actionable improvements based on a feedback obtained from the system's output.
The feedback contains strategies to improve the retrieved content and the style of the answer: FOCUS ONLY ON THE STYLE.

Current answer generation prompt:
{current_prompt}

Feedback from the response evaluation:
{response_evaluator_output}

Based on this information, determine if there is a problem with the answer generation prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a critique focusing on the specific issue.
If false, leave the critique empty.
"""

generation_prompt_gradient_prompt_vector = """
You are evaluating the prompt used for answer generation in a VectorRAG system.
Provide a critique (what the prompt should include, what it should not) based on the following feedback derived from a previous response of the system.

Response evaluator output:
{response_evaluator_output}

Current answer generation prompt:
{current_prompt}

Based on this information, determine if there is a problem with the answer generation prompt that needs to be fixed.

First, provide your reasoning explaining why there is or isn't a problem.
Then, set problem_in_this_component=true or false accordingly.
If true, provide a detailed critique to the prompt reporting what you learned in the previous critique. The critique must be actionable: it must include concrete actions that would likely improve the output. 
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
Based on the information above, provide a detailed critique of how the retrieved content can be improved to better answer the query. The critique must be actionable: it must include concrete actions that would likely improve the output. 
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
You are evaluating the retrieval queries made by a VectorRAG system.

═══════════════════════════════════════════════════════════════════
QUESTION TO ANSWER:
═══════════════════════════════════════════════════════════════════
{question}

═══════════════════════════════════════════════════════════════════
QUERIES AND RETRIEVED CONTENT:
═══════════════════════════════════════════════════════════════════
{queries_and_content}

═══════════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════════
1. For each query above, evaluate if the retrieved content helped answer the question or not.
2. Identify what other queries could have been made to retrieve more helpful information.

Provide a detailed, actionable critique that:
- Analyzes each query individually (was the retrieved content helpful?)
- Suggests specific alternative or additional queries that would have been more effective
- Focuses on improving query relevance and information coverage
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
CURRENT RETRIEVAL PLANNING PROMPT:
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
You are evaluating the prompt given to a LLM to enrich a given graph with the information extracted from the text.
The prompt tells the system how to extract entities and relationships. 
You will be provided with a feedback explaning which information is missing in the graph.
You have to think: which entities/relationships types would make this information available in the graph?
Clearly specify the entities and relationship types in your answer. 
Include only a few entities and relationship types (not more than 6-7). 
Based on this, you have to identify entities and relationships types that should be included in the graph and are not specified in the current prompt. 
Focus only on the most crucial entity/relationship types to meet the evaluation requirement. Specify only a few entities/relationships types, the ones that are most important. 

For each entity/relationship, you have to include examples (each example is a phrase or a sentence)

The feedback from the system is:
{response_evaluator_output}

The current prompt is:
{current_prompt}

Please, determine which entities/relationships the current prompt is missing based on the feedback.
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
You are optimizing a prompt for answer generation in a GraphRAG system.

The current critique of the answer generation process is:
{}

Based on this critique, generate a new prompt that will be used to instruct the LLM how to better generate answers from retrieved graph information. The prompt should incorporate the feedback to improve answer quality, relevance, and coherence.

Provide only the optimized prompt without additional commentary.
In the prompt, add also the instructions:
You must base your answer primarily on the information provided in the context below.
Do not invent or assume facts that are not supported by the context.
However, if the context provides only partial information, you must still produce the most complete and coherent answer possible by reasoning explicitly from what is given.
If the information is insufficient to answer directly, explain what can be inferred from the context and what remains unknown, instead of saying that The information is not available.
"""

retrieval_planner_prompt_optimizer = """
You are optimizing a prompt for community selection in a community-based GraphRAG system.

The current critique of the community selection process is:
{}

Based on this critique, generate a new prompt that will be used to instruct the LLM how to better select relevant communities. The prompt should incorporate the feedback to improve community selection strategy, relevance assessment, and information gathering.

Provide only the optimized prompt without additional commentary.
"""

graph_builder_prompt_optimizer = """
Your goal is to generate an instruction for a LLM that enriches a graph using information from the text. 
The instruction must have the following format: "Focus on:" + LIST OF ENTITY/RELATIONSHIP TYPES
The instruction tells the model which entities/relationships types to expand the graph with.
Include only A FEW entities and relationship types (NOT MORE THAN 6-7 IN TOTAL). 
You have to accompany each entity/relationship with a full explanation and some examples. Each example is a phrase or a sentence.
Specify only the relationships and the entities, don't use formatted examples since they can misled the model output format. 


Your suggestions must be based on this feedback:
{}

Provide only the instruction additional commentary.
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
You are optimizing a prompt for community summarization in a GraphRAG system.

The current critique of the community summarization process is:
{}

Based on this critique, generate a new prompt that will be used to instruct the LLM how to better summarize graph communities. The prompt should incorporate the feedback to improve summary quality, relevance, and informativeness.

The prompt should instruct the LLM to:
1. Identify key entities and their relationships within the community
2. Capture important thematic information
3. Provide sufficient detail for query answering
4. Maintain clarity and conciseness

Provide only the optimized prompt without additional commentary.
"""

answer_generation_prompt_optimizer_vector = """
You are optimizing a prompt for answer generation in a RAG system. The prompt must be general, but it also has to be as much adherent as possible to the following critique.

The current critique of the answer generation process is:
{}

Based on this critique, generate a new prompt that will be used to instruct the LLM how to better generate answers from retrieved vector information. The prompt should incorporate the feedback to improve answer quality, relevance, and coherence.
The prompt must be actionable: it must include concrete actions that would likely improve the output. 

Provide only the optimized prompt without additional commentary.
"""

retrieval_planner_prompt_optimizer_vector = """
You are optimizing a prompt for a query agent in a vector RAG system.

IMPORTANT CLARIFICATION:
- The planner you are optimizing asks the model to generate a sub-query to answer the question, based on the context retrieved so far and the previous sub-queries.
- The planner does NOT control the retrieval/embedding mechanism itself
- Your instructions should focus on which aspects a sub-query must target

You will receive:
1. The original question to answer
2. The sub-queries that were made
3. A critique of the retrieval plan identifying what worked and what needs to be improved. 

{}

Based on this information, generate an optimized prompt that instructs the query agent to create a good sub-query.

Your prompt MUST include instruction to make a sub-query that target the missing information.

Provide only the optimized prompt as instructions for the query planner, without additional commentary. Introduce the prompt by explaining the role "Your goal is to generate a sub-query for a rag system, based on the missing information in the context retrieved so far and the previous queries."
Generate a full optimized prompt (not partial additions) that will replace the current one used by the query agent. 
Clearly specify that each sub-query must be targeted and cover ONE ASPECT. Avoid dense subqueries such as: 
"Detail the immediate aftermath of Captain Dietrich's capture by the Misty Ones, clarifying how Doctor Von Mark, despite a near-fatal arrow wound, survived and orchestrated the encounter with Dietrich. Explain how Captain Dietrich's fragmented memories fully coalesce during this confrontation, directly shaping his final actions and motivations, particularly in thwarting Von Mark's specific plan to utilize an 'invisible material' for Earth's conquest, and provide a clear resolution to this central conflict"
MAKE IT CLEAR THAT THE SYSTEM MUST AVOID PREVIOUS QUERIES. 
"""

retrieval_summarizer_prompt_gradient_vector = """
You are evaluating the prompt used for chunk summarization in a VectorRAG system.
Each retrieved chunk of text is processed by a summarizer that generates a summary based on the prompt below.

═══════════════════════════════════════════════════════════════════
CURRENT RETRIEVAL SUMMARIZER PROMPT :
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
You are optimizing a prompt for document summarization in a vector RAG system.

The current critique of the retrieval summarization process is:
{}

Based on this critique, generate a new prompt that will be used to instruct the LLM how to better summarize retrieved documents. The prompt should incorporate the feedback to improve summarization quality, relevance focus, and information extraction.

Provide only the optimized prompt without additional commentary.
"""

hyperparameters_vector_agent_prompt_optimizer = """
You are optimizing a prompt for hyperparameter selection in a vector RAG system.

The current critique of the hyperparameter selection process is:
{}

Based on this critique, generate a new prompt that will be used to instruct the LLM how to better determine optimal chunk sizes for vector construction. The system prompt should incorporate the feedback to improve hyperparameter reasoning and selection.

Provide only the optimized prompt without additional commentary.
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

# Community data mode: Toggle between raw graph data and summarized version
# "raw": Pass raw markdown graph data directly to community answer agents (bypasses summarization)
# "summarized": Use traditional summarization (LLM generates summary, then answer agents use it)
COMMUNITY_DATA_MODE = "raw"  # Options: "raw" or "summarized"


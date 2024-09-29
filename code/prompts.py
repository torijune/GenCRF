'''
각 상황에 맞는 프롬포트들을 함수를 사용해 불러오면 됨
EX,
from prompts import Q2D
Q2D_prompt = Q2D()
'''

# 디버깅을 위한 inital query -> 나중에는 user에게 직접 query를 받을 듯
def initial_query():
    query = "What is diffusion tensor?"
    return query

############ GenCRF에서 제시하는 3개의 reformulation propmts -> 해당 논문에서 사용하는 확장법 ############
############ query 부분에는 initial_query가 들어감 ############

'''
Contextual Expansion 프롬프트
Contextual Expansion은 주어진 질문의 핵심 의도를 파악한 후, 문맥을 확장하여 더 풍부한 정보를 제공합니다.
질문의 문맥을 넓혀 검색 결과를 더 확장하는 방식입니다.
'''

def ContextualExpansion(query):
    prompt = f"""
    You are a contextual expansion expert. Your task is to understand the core intent of the original query and provide a refined, contextually expanded answer within 100 characters.
    Below is the query: {query}
    """
    return prompt

'''
Detail Specific 프롬프트
Detail Specific은 질문의 구체적인 세부 사항이나 하위 주제에 초점을 맞춰 답변을 생성합니다.
질문의 특정 부분에 대해 깊이 있는 정보를 제공하는 방식입니다.
'''

def DetailSpecific(query):
    prompt = f"""
    You are a detail-specific expert. Your task is to understand the core intent of the original query and provide a refined, detailed answer focusing on particular details or subtopics within 100 characters.
    Below is the query: {query}
    """
    return prompt

'''
Aspect Specific 프롬프트
Aspect Specific은 특정 주제의 한 측면이나 차원에 집중하여 쿼리를 확장하는 방식입니다.
주제의 특정 측면에 대해 더 구체적이고 풍부한 결과를 도출할 수 있도록 돕습니다.
'''

def AspectSpecific(query):
    prompt = f"""
    You are an aspect-specific inquiry expert. Your task is to provide a refined answer focusing on a specific aspect or dimension within 100 characters.
    Below is the query: {query}
    """
    return prompt

############ Clustering-Generation prompt ############

'''
Clustering Refinement 프롬프트
Clustering Refinement은 생성된 쿼리들을 의도나 유사성에 따라 1~3개의 클러스터로 그룹화합니다.
각 클러스터에서 대표적인 쿼리를 도출하여 검색에 사용할 수 있도록 합니다.
'''

def ClusteringRefinement(query):
    prompt = f"""
    You are an expert in clustering and query refinement. Your task is to review the original query alongside the generated queries, and then cluster them into 1 to 3 groups based on their similarity and relevance.
    The number of clusters should be determined dynamically. Focus primarily on the relationship of the generated queries to the original query. For each identified cluster, provide only one refined query that incorporates elements from the original and generated queries within that cluster with useful information for document retrieval.
    The output should be presented in JSON format, structured as follows: 'cluster1': 'refined_query_1', 'cluster2': 'refined_query_2', 'cluster3': 'refined_query_3'
    The output must be restricted to 1 to 3 groups.
    Below is the query: {query}
    """
    return prompt

# 위에서 생성된 query들을 input으로 넣고 output으로는 Cluster-Generated Queries가 생성됨
# clsuter_generated_queries = LLM(ClusteringRefinement(~))

########## QERM을 위한 ScoreDW을 얻는 prompt ########

'''
LLM Scoring 프롬프트
LLM Scoring은 각 클러스터의 쿼리들을 평가하여 점수를 부여하는 프롬프트입니다.
관련성, 구체성, 명확성, 포괄성, 유용성을 기준으로 점수를 매깁니다.
'''

def LLMScoring(q_init, cluster_queries):
    prompt = f"""
    You are an expert in scoring cluster queries. Evaluate the clustering of queries using the following criteria for each cluster: Relevance, Specificity, Clarity, Comprehensiveness, and Usefulness for retrieval.
    Assign a score from 1 to 100, where 1 is the lowest and 100 is the highest performance in relation to the original query. Avoid defaulting to high scores unless they are clearly justified. Carefully consider both the strengths and weaknesses of each cluster.
    For instance, a cluster with relevant but not highly specific results might score between 40 and 60, while a cluster that is both highly relevant and specific might score between 70 and 100. Conversely, a cluster lacking clarity or comprehensiveness should score lower, between 10 and 30.
    Provide scores that accurately reflect the variation in quality across clusters. Return your scores in a list format only, without additional commentary.
    List your scores for each cluster in the following list format: [score_cluster1, score_cluster2, score_cluster3].
    Initial Query: {q_init}
    Cluster-Generated Queries: {cluster_queries}
    """
    return prompt

# 위에서 생성된 cluster queries를 initial query와 비교하여 Scoring을 진행
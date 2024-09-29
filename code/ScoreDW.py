import numpy as np
import json
from prompts import initial_query, LLMScoring
from LLMs import generate_reformulated_queries
from Clustering import main as clustering
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
scoring_llm = ChatOllama(model="mistral-7b:latest")  # LLM 설정

def json_to_text_without_braces(json_data):
    """
    JSON 데이터에서 중괄호 {}를 없애고 텍스트로 변환하는 함수.
    :param json_data: JSON 형식의 데이터 (딕셔너리)
    :return: 텍스트 형식의 문자열
    """
    # JSON 데이터를 문자열로 변환
    json_str = json.dumps(json_data, indent=2)

    # 중괄호를 제거
    no_braces_str = json_str.replace("{", "").replace("}", "")

    return no_braces_str

def get_llm_score(q_init, cluster_queries):
    """
    LLM을 사용하여 재구성된 쿼리들의 품질을 평가하는 함수.
    
    :param query: 초기 쿼리 (문자열)
    :param cluster_queries: 클러스터링된 쿼리 집합 (리스트)
    :return: LLM으로 평가된 점수 (리스트)
    """
    cluster_queries = json_to_text_without_braces(cluster_queries)
    scoring_prompt = ChatPromptTemplate.from_template(LLMScoring(q_init, cluster_queries))
    chain = scoring_prompt | scoring_llm | StrOutputParser()
    score = chain.invoke({"prompts": scoring_prompt})
    return score

def score_dw(q_init, q_f, q_f_text, w0=0.7, score_theta=60):
    """
    ScoreDW 함수: 초기 쿼리와 비교하여 특정 재구성된 쿼리의 퀄리티를 LLM을 통해 계산하여 가중치를 계산.
    
    :param q_init: 초기 쿼리 벡터 (numpy array)
    :param q_f: 재구성된 특정 쿼리 벡터 (numpy array)
    :param q_f_text: 재구성된 쿼리 텍스트 (문자열)
    :param w0: 초기 쿼리에 대한 고정 가중치 (float, 기본값 0.7)
    :param score_theta: LLM으로 평가한 스코어 필터링을 위한 임계값 (float, 기본값 60)
    :return: 가중치가 적용된 집계 쿼리 (numpy array)
    """
    
    # 초기 쿼리 q_init에 고정 가중치 w0를 적용
    q_agg = w0 * q_init
    
    # LLM을 사용해 재구성된 쿼리의 점수 계산 (다차원 평가)
    score = float(get_llm_score(q_init, q_f_text))
    
    # 점수가 임계값 theta 이상인 경우에만 가중치 적용
    if score >= score_theta:
        q_agg += score * q_f
    
    return q_agg, score

def main(Q_final, q_init):

    Q_final = json_to_text_without_braces(Q_final)
    q_init = q_init

    score = get_llm_score(q_init = q_init, cluster_queries = Q_final)
    # print("LLM Score :", score)
    return score

# if __name__ == "__main__":
#     main()
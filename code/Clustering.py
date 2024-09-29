import json
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from LLMs import generate_reformulated_queries
from prompts import ClusteringRefinement, initial_query

# Ollama 모델 설정
clustering_llm = ChatOllama(model="mistral-7b:latest")

def format_queries(reform_queries):
    """
    생성된 reformulated queries의 딕셔너리 형태를 텍스트 형식으로 변환하는 함수
    :param query_dict: q_gen과 같은 딕셔너리
    :return: reformulated queries를 변환한 텍스트 -> clsutering 가능하도록
    """
    formatted_text = []
    
    for prompt_type, queries in reform_queries.items():
        # 프롬프트 타입에 따른 이름 설정
        if prompt_type == 'CE_prompt':
            prompt_name = "ContextualExpansion"
        elif prompt_type == 'DS_prompt':
            prompt_name = "Detail Specific"
        elif prompt_type == 'AS_prompt':
            prompt_name = "Aspect Specific"
        
        # 각 쿼리에 번호를 붙여서 추가
        for i, query in enumerate(queries):
            query_number = "First" if i == 0 else "Second"
            formatted_text.append(f"{query_number} {prompt_name} query: {query}")
    
    return "\n".join(formatted_text)


def cluster_generation(query):
    """
    LLM을 사용하여 주어진 쿼리와 재구성된 쿼리들을 클러스터링하여 대표 쿼리를 생성하는 함수.

    :param query: 초기 쿼리 (문자열)
    :param reformulated_queries: 재구성된 쿼리들 (리스트)
    :return: 클러스터링된 대표 쿼리들 (JSON 형식의 문자열)
    """
    # 프롬프트 생성 (초기 쿼리와 재구성된 쿼리들)
    clustering_prompt = ClusteringRefinement(query = query)
    
    # Scoring LLM 호출
    prompt = ChatPromptTemplate.from_template(clustering_prompt)
    chain = prompt | clustering_llm | StrOutputParser()
    
    # 클러스터링 결과 얻기
    result = chain.invoke({"prompts": prompt})

    return result


def main(q_gen):
    user_query = initial_query()
    # reform queries 정의 및 cluster qeuries 생성
    q_gen = q_gen
    # reform queries들을 모두 텍스트 형태로 변환
    q_gen_text = format_queries(q_gen)
    # print("formatted queries :", q_gen_text)
    # 변환된 쿼리들을 clustering prompt에 입력
    clustered_queries = cluster_generation(q_gen_text)

    return clustered_queries

    # 결과 출력
if __name__ == "__main__":
    clustered_queries = main(generate_reformulated_queries(initial_query()))
    print(clustered_queries)
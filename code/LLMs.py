import json
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompts import ContextualExpansion, DetailSpecific, AspectSpecific, initial_query

# Ollama 모델 설정
# top_p = 0.95 설정, temperature = 0.8은 ollama modelfile에서 수정함
llm = ChatOllama(model="llama-3.1-8B-instrcut:latest", top_p=0.95, num_predict=100)

# 함수 정의
def generate_reformulated_queries(user_query):
    """
    사용자 쿼리를 기반으로 각 프롬프트에서 2개의 쿼리를 생성하는 함수

    :param user_query: 사용자가 입력한 초기 쿼리
    :return: 각 프롬프트에서 생성된 쿼리들 (딕셔너리 형태)
    """
    # 각각의 프롬프트를 ChatPromptTemplate으로 생성
    CE_prompt = ChatPromptTemplate.from_template(ContextualExpansion(user_query))
    DS_prompt = ChatPromptTemplate.from_template(DetailSpecific(user_query))
    AS_prompt = ChatPromptTemplate.from_template(AspectSpecific(user_query))

    # 재구성된 쿼리를 저장할 딕셔너리
    reform_queries = {}

    # 프롬프트와 그에 맞는 딕셔너리 키를 설정
    prompts = [("CE_prompt", CE_prompt), ("DS_prompt", DS_prompt), ("AS_prompt", AS_prompt)]

    # 각 프롬프트를 처리하여 2개의 쿼리를 생성 및 결과 저장
    for name, prompt in prompts:
        reform_queries[name] = []
        for i in range(2):  # 각 프롬프트당 2개의 쿼리 생성
            chain = prompt | llm | StrOutputParser()
            reform_queries[name].append(chain.invoke({"prompts": prompt}))

    return reform_queries

# reform_queries['CE_prompt'][0] -> CE_prompt reformulation의 첫 번째 쿼리

# # 함수 사용 예시
# if __name__ == "__main__":
#     user_query = initial_query()  # 초기 쿼리 설정
#     queries = generate_reformulated_queries(user_query)  # 함수 호출
#     print(queries)
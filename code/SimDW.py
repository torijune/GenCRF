import numpy as np
from prompts import initial_query
from LLMs import generate_reformulated_queries
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_embedding(query):
    """
    입력된 쿼리의 임베딩을 생성하는 함수.
    
    :param query: 텍스트로 된 쿼리 (문자열)
    :return: 임베딩 벡터 (numpy array)
    """
    # 쿼리의 임베딩 생성
    embedding = model.encode(query)
    return embedding

def sim_dw(q_init, q_f, w0=0.7, sim_theta=0.2):
    """
    SimDW 함수: 초기 쿼리와 특정 재구성된 쿼리 간의 유사도를 기반으로 가중치를 적용하여 최종 쿼리를 계산.
    
    :param q_init: 초기 쿼리 벡터 (numpy array)
    :param q_f: 재구성된 특정 쿼리 벡터 (numpy array)
    :param w0: 초기 쿼리에 대한 고정 가중치 (float, 기본값 0.7)
    :param sim_theta: 유사도 필터링을 위한 임계값 (float, 기본값 0.2)
    :return: 가중치가 적용된 집계 쿼리 (dictionary)
    """
    
    # 초기 쿼리 q_init에 고정 가중치 w0를 적용
    q_agg = w0 * q_init
    
    # q_init과 q_f 간의 코사인 유사도 계산
    similarity = cosine_similarity([q_init], [q_f])[0][0]
    
    # 유사도가 임계값 theta 이상인 경우에만 가중치 적용
    if similarity >= sim_theta:
        q_agg += similarity * q_f
    
    return q_agg, similarity

def main(q_init, q_gen):
    '''
    재구성된 쿼리들의 SimDW를 실제로 계산
    sim_agg_dw['AS_prompt'][0]: AS_prompt로 재구성한 첫 번째 쿼리의 SimDW

    :param q_init: 초기 텍스트 쿼리 -> 임베딩 필요
    :param q_gen: reformulated queries를 입력 -> 임베딩 필요(분리 후)
    :return: reformulation queries에 대한 SimDW (dictionary)
    '''
    q_init = get_embedding(q_init)
    reform_queries = q_gen

    # 각 재구성된 쿼리들에 대해 SimDW를 적용하여 결과를 저장할 딕셔너리
    sim_agg_dw = {
        'CE_prompt': [],
        'DS_prompt': [],
        'AS_prompt': []
    }

    weights = {
        'CE_prompt': [],
        'DS_prompt': [],
        'AS_prompt': []
    }

    # CE_prompt에 대해 각각 SimDW와 임베딩 벡터(쿼리) 계산
    for i in range(2):
        sim_agg_dw['CE_prompt'].append(sim_dw(q_init, get_embedding(reform_queries['CE_prompt'][i]))[0])
        weights['CE_prompt'].append(sim_dw(q_init, get_embedding(reform_queries['CE_prompt'][i]))[1])

    # DS_prompt에 대해 각각 SimDW 계산
    for i in range(2):
        sim_agg_dw['DS_prompt'].append(sim_dw(q_init, get_embedding(reform_queries['DS_prompt'][i]))[0])
        weights['DS_prompt'].append(sim_dw(q_init, get_embedding(reform_queries['DS_prompt'][i]))[1])

    # AS_prompt에 대해 각각 SimDW 계산
    for i in range(2):
        sim_agg_dw['AS_prompt'].append(sim_dw(q_init, get_embedding(reform_queries['AS_prompt'][i]))[0])
        weights['AS_prompt'].append(sim_dw(q_init, get_embedding(reform_queries['AS_prompt'][i]))[1])

    return sim_agg_dw, weights

# if __name__ == "__main__":
#     # 사용자 쿼리와 재구성된 쿼리들
#     queries = generate_reformulated_queries(initial_query())

#     # SimDW 계산
#     sim_agg_dw_result, weights_result = main(queries['CE_prompt'][0], queries['CE_prompt'][1], 
#                                              queries['DS_prompt'][0], queries['DS_prompt'][1], 
#                                              queries['AS_prompt'][0], queries['AS_prompt'][1])

#     # 결과 출력
#     print("SimDW Results (Embedding):")
#     print(f"CE_prompt[0]: {sim_agg_dw_result['CE_prompt'][0]}")
#     print(f"CE_prompt[1]: {sim_agg_dw_result['CE_prompt'][1]}")
#     print(f"DS_prompt[0]: {sim_agg_dw_result['DS_prompt'][0]}")
#     print(f"DS_prompt[1]: {sim_agg_dw_result['DS_prompt'][1]}")
#     print(f"AS_prompt[0]: {sim_agg_dw_result['AS_prompt'][0]}")
#     print(f"AS_prompt[1]: {sim_agg_dw_result['AS_prompt'][1]}")
    
#     print("\nSimDW Weights (Cosine Similarity):")
#     print(f"CE_prompt[0]: {weights_result['CE_prompt'][0]}")
#     print(f"CE_prompt[1]: {weights_result['CE_prompt'][1]}")
#     print(f"DS_prompt[0]: {weights_result['DS_prompt'][0]}")
#     print(f"DS_prompt[1]: {weights_result['DS_prompt'][1]}")
#     print(f"AS_prompt[0]: {weights_result['AS_prompt'][0]}")
#     print(f"AS_prompt[1]: {weights_result['AS_prompt'][1]}")
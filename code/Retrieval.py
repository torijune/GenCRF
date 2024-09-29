import numpy as np

'''
일단 q_agg_simDW와 q_agg_scoreDW가 있어야함
Q_retri = q_init + q_agg_simDW + q_agg_scoreDW : retrieval을 진행할 input query(q_init(raw), q_gen(weighted), Q_final(Weight)를 모두 가지고 있음)
Ranking(Cosine_Similiarity(Q_retri, Contents embedding vectors)) : 모든 embedding 과정에서는 동일한 model을 사용해야함
'''

def new_retrieval_query(q_agg_simDW, q_agg_scoreDW, q_init):
    """
    Q_retri를 생성하는 함수.
    q_init, q_agg_simDW, q_agg_scoreDW 모두 같은 임베딩 차원을 가지며, 이를 합산하여 최종 검색 쿼리를 생성.
    
    :param q_agg_simDW: SimDW로부터 얻은 임베딩 벡터 (numpy array)
    :param q_agg_scoreDW: ScoreDW로부터 얻은 임베딩 벡터 (numpy array)
    :param q_init: 초기 쿼리 임베딩 벡터 (numpy array)
    :return: 최종 검색 쿼리 Q_retri (numpy array)
    """
    # 벡터 합산을 통해 최종 검색 쿼리 생성
    Q_retri = q_init + q_agg_simDW + q_agg_scoreDW
    return Q_retri

# 예시
q_init = np.array([0.5, 0.6, 0.7])  # 초기 쿼리 임베딩 벡터
q_agg_simDW = np.array([0.4, 0.5, 0.6])  # SimDW로부터 얻은 임베딩 벡터
q_agg_scoreDW = np.array([0.3, 0.4, 0.5])  # ScoreDW로부터 얻은 임베딩 벡터

# 최종 검색 쿼리 생성
Q_retri = new_retrieval_query(q_agg_simDW, q_agg_scoreDW, q_init)
print("최종 검색 쿼리 (Q_retri):", Q_retri)
from prompts import initial_query
# 모든 과정에서 reformulation query를 통일하기 위해 처음부터 재구성 쿼리를 생성 후 진행 
from LLMs import generate_reformulated_queries
# q_gen을 input으로 받아 cluster 생성
# Q_final = clustering(~)으로 클러스터링 쿼리 정의
from Clustering import main as clustering
# q_init과 q_gen을 input으로 받아 코사인 유사도 측정 및 q_agg_simDW 생성
from SimDW import main as simdw
# q_init과 Q_final를 input으로 받아 LLMSocring 진행 및 q_agg_scoreDW 생성
# 이때, Q_final은 클러스터링 쿼리
from ScoreDW import main as scoredw
# 나중에 fusion 및 retrieval 코드 필요

"""
전체 코드 흐름 :
LLMs를 통해 q_init에서 q_gen 생성 (LLMs.py)
-> q_gen과 q_init의 SimDW 및 q_agg_simDW 계산 (SimDW.py)
-> q_gen을 클러스터링하여 Q_final 생성 (cluster.py)
-> Q_final과 q_init의 score를 측정하여 ScoreDW 계산 (ScoreDW.py)
-> ScoreDW와 q_init의 조합으로 q_agg_scoreDW 계산 (추후 ScoreDW.py에 추가)
-> 이렇게 얻은 W, Q_final과 q_init의 조합으로 Retireval 진행 (추후 Retrieval.py 작성)
-> Retrieval 결과를 토대로 QERM으로 clsutering & reformulation 평가 및 재진행
여전히 고민되는 벡터 차원 문제.....나중에 형한테 질문해보자 확실히 fusion 쪽이 많이 어렵네
"""

# 나중에는 initial_query를 prompts.py에서 불러오는게 아니라 user가 직접 입력하도록 수정
def main():
    q_init = initial_query()
    q_gen = generate_reformulated_queries(q_init)
    # q_gen 생성 이상 무
    Q_final = clustering(q_gen)
    # Q_final 생성 이상 무, 1~3개로 동적으로 클러스터링 되는거 확인 함
    ScoreDW = scoredw(Q_final = Q_final, q_init = q_init)
    # ScoreDW 생성 이상 후 -> score가 2개만 나왔는데 아마도 클러스터링이 2개 그룹으로 되서 그런 듯?
    q_agg_simDW, SimDW = simdw(q_init = q_init, q_gen = q_gen)
    # q_agg_simDW 및 SimDW 모두 잘 생성 됨

if __name__ == "__main__":
    main()
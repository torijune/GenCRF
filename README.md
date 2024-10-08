**This repositorry is not official code, but code implemented by individuals as an exercise.**   
   
**This repository contains the implementation of a retrieval query optimization system that uses **SimDW** and **ScoreDW** to generate and rank queries based on user input.**   

# GenCRF  
The system integrates **Llama-3.1-8B** for query reformulation and **Mistral-7B** for clustering, with evaluation done using **LLM(Mistral-7B)** scoring techniques with **innovative weighted aggregation strategies(SimDW & ScoreDW)** to optimize retrieval performance and crucially integrates a novel **Query Evaluation Rewarding Model (QERM)** to refine the process through feedback loops.

## Overview
In this project, we aim to optimize the retrieval query by using **customized prompts**, then **clusters** them into groups to distinctly represent diverse intents:
- **q_init**: The original user query.
- **q_gen**: Reformulated queries generated using **Llama-3.1-8B**.
- **Q_final**: Clustering results generated by **Mistral-7B**.

The system calculates **SimDW (Similarity Dynamic Weights)** and **ScoreDW (Score Dynamic Weights)** for these queries and uses these values to form an optimized query (`Q_retri`) for retrieval.

### Components:
1. **Customized prompts**: 
2. **SimDW**: Uses cosine similarity to assign dynamic weights to reformulated queries based on their similarity to `q_init`.
3. **ScoreDW**: Uses LLM scoring to evaluate the quality of reformulated queries based on five key metrics: Relevance, Specificity, Clarity, Comprehensiveness, and Usefulness.
4. **Q_retri**: The final retrieval query, combining `q_init`, `q_agg_simDW`, and `q_agg_scoreDW`.

## Features

- **Query Reformulation**: Generate reformulated queries based on the original query using **Llama-3.1-8B**.
- **Query Clustering**: Cluster reformulated queries using **Mistral-7B**.
- **SimDW Calculation**: Calculate similarity-based dynamic weights using cosine similarity.
- **ScoreDW Calculation**: Use LLMs to score and calculate dynamic weights for reformulated queries.
- **Final Query Generation**: Combine `q_init`, `q_agg_simDW`, and `q_agg_scoreDW` to form the final retrieval query (`Q_retri`).

## Installation

To run this project, you need to install the necessary Python packages and ensure the required models are available.

### 1. Clone the repository:

```bash
git clone https://github.com/torijune/GenCRF.git
cd GenCRF

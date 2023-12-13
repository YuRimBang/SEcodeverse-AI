import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def user_recommend(level, try_count, score, memory, user_solved_question_pk):
  weight_level = 4
  weight_try_count = 3
  weight_score = 2
  weight_memory = 1

  user_features = np.array([level, try_count, score, memory])
  user_features_weighted = user_features * np.array([weight_level, weight_try_count, weight_score, weight_memory])

  anonymous_data = pd.read_csv('SECodeVerse_AI\output.csv')

  #user_solved_question_pk = input("이미 푼 문제의 question_pk를 입력하세요 (쉼표로 구분): ").split(',')
  user_solved_question_pk = list(map(int, user_solved_question_pk.split(',')))

  similarity_scores = cosine_similarity([user_features_weighted], anonymous_data[['level', 'tryCount', 'score', 'memory']])
  most_similar_problem_index = np.argmax(similarity_scores)

  recommended_problems = []
  for i in range(5):
      while (
          anonymous_data.iloc[most_similar_problem_index]['question_pk'] in recommended_problems
          or anonymous_data.iloc[most_similar_problem_index]['question_pk'] in user_solved_question_pk
      ):
          similarity_scores[0, most_similar_problem_index] = -1  # 이미 추천된 문제나 사용자가 푼 문제는 제외
          most_similar_problem_index = np.argmax(similarity_scores)
      recommended_problems.append(anonymous_data.iloc[most_similar_problem_index]['question_pk'])
  return recommended_problems

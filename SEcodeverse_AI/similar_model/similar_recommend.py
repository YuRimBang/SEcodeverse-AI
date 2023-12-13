import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def similar_recommend(level, category):

  weight_level = 2
  weight_category = 5

  user_features = np.array([level, category])
  user_features_weighted = user_features * np.array([weight_level, weight_category])

  anonymous_data = pd.read_csv('C:\project\project\SECodeVerse_API\output.csv')  # 여기에 익명의 사용자들이 푼 문제 데이터가 들어있는 파일 경로를 넣어주세요

  similarity_scores = cosine_similarity([user_features_weighted], anonymous_data[['level', 'inserted_pk']])
  most_similar_problem_index = np.argmax(similarity_scores)

  recommended_problems = []
  for i in range(5):
      while (
          anonymous_data.iloc[most_similar_problem_index]['question_pk'] in recommended_problems
      ):
          similarity_scores[0, most_similar_problem_index] = -1  # 이미 추천된 문제나 사용자가 푼 문제는 제외
          most_similar_problem_index = np.argmax(similarity_scores)
      recommended_problems.append(anonymous_data.iloc[most_similar_problem_index]['question_pk'])

  return recommended_problems

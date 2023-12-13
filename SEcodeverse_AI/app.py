import json
import tensorflow as tf
import openai
from flask import Flask, request, jsonify, Response

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

openai.api_key = '' #조심

anonymous_data = pd.read_csv('SECodeVerse_AI\output.csv')

def user_recommend(level, try_count, score, memory, user_solved_question_pk):
  weight_level = 4
  weight_try_count = 3
  weight_score = 2
  weight_memory = 1

  user_features = np.array([level, try_count, score, memory])
  user_features_weighted = user_features * np.array([weight_level, weight_try_count, weight_score, weight_memory])


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

def similar_recommend(level, category):

  weight_level = 2
  weight_category = 5

  user_features = np.array([level, category])
  user_features_weighted = user_features * np.array([weight_level, weight_category])

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

def fqa_res(prompt):
  response = sentence_generation(prompt)

  return response

#추론 모드
def decoder_inference(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output_sequence = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

    return tf.squeeze(output_sequence, axis=0)

def sentence_generation(sentence):
    prediction = decoder_inference(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    #print('입력 : {}'.format(sentence))
    #print('출력 : {}'.format(predicted_sentence))

    return predicted_sentence

def coco_code_view(code):

    model = "gpt-3.5-turbo"

    query = code+"에 대해 리뷰하고 개선할 점 알려줘."

    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": query
    }]

    response = openai.ChatCompletion.create(model=model, messages=messages)
    answer = response['choices'][0]['message']['content']

    return answer

@app.route('/', methods=['GET', 'POST'])
def api_sentence_generation():
    if request.method == 'GET':
        data = {"response": "안녕."}
        return jsonify(data)

    elif request.method == 'POST':
        try:
            data = request.get_json()
            sentence = data.get('sentence')
            code = data.get('code')

            if sentence == "내가 푼 문제는 어디서 확인해?":
                response = "마이페이지에 들어가면 문제 확인 탭을 누르면 틀린문제  만든 문제  풀었던 문제 이렇게 확인할 수 있습니다."
            elif sentence == "문제 등록은 어디서 해?":
                response = "문제 목록 페이지 검색창 옆에 있습니다."
            elif sentence == "티어에 대해 설명해줘":
                response = "저희 SeCodeVerse의 티어는 알->아기까마귀->초딩까마귀->사춘기까마귀->대딩까마귀->석박사까마귀로 이루어져있습니다."
            elif sentence == "CTF가 뭐야?":
                response = "CTF는 Capture The Flag의 약자로 본진에 침투해 깃발을 탈취하여 가져오면 점수를 얻는 게임 방식입니다. 우리 SeCodeVerse에서는 CTF리그를 열어 기간안에 컴퓨터지식 관련된 주관식  객관식 문제를 테스트하는 리그를 진행합니다. 문제를 풀어 점수를 획득하여 많은 점수를 모은 팀에게는 뱃찌를 수여합니다!"
            elif sentence == "게시글은 어디서 봐?":
                response = "커뮤니티로 가면 다양한 카테고리 별로 게시글을 확인 할 수 있습니다."
            elif sentence == "리뷰":
                response = coco_code_view(code)
            else:
                response = fqa_res(sentence)
            return Response(response=json.dumps({'response': response}, ensure_ascii=False).encode('utf-8'), content_type='application/json; charset=utf-8')
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/userRecommend', methods=['GET', 'POST'])
def api_sentence_generation():
    if request.method == 'GET':
        data = {"response": "사용자 수준 문제 추천."}
        return jsonify(data)

    elif request.method == 'POST':
        try:
            data = request.get_json()
            level = data.get('level')
            try_count = data.get('tryCount')
            score = data.get('score')
            memory = data.get('memory')
            user_solved_question_pk = data.get('userSolvedQuestionPK')#쉼표로 구분된 pk들

            response_array = user_recommend(level, try_count, score, memory, user_solved_question_pk)
            response_dict = {'response': response_array}
            return Response(response=json.dumps(response_dict, ensure_ascii=False).encode('utf-8'), content_type='application/json; charset=utf-8')
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/similarRecommend', methods=['GET', 'POST'])
def api_sentence_generation():
    if request.method == 'GET':
        data = {"response": "사용자 수준 문제 추천."}
        return jsonify(data)

    elif request.method == 'POST':
        try:
            data = request.get_json()
            level = data.get('level')
            category = data.get('category')

            response_array = user_recommend(level, category)
            response_dict = {'response': response_array}
            return Response(response=json.dumps(response_dict, ensure_ascii=False).encode('utf-8'), content_type='application/json; charset=utf-8')
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
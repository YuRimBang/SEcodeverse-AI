import tensorflow as tf 

import tensorflow_datasets as tfds 

import re 

import pandas as pd 

  

  

'''전처리''' 

data = pd.read_csv('SEcodeverse_AI/trans_model/coco_data_final.csv', encoding='cp949') 

data 

   

MAX_SAMPLES = 203 

  

# 전처리 함수 

def preprocess_sentence(sentence): 

    sentence = sentence.lower().strip() #토큰수 줄이기, 일관성, 일반화 

  

    # 단어와 구두점(punctuation) 사이의 거리를 만듭니다. 

    # 구두점 주변에 공백을 추가하면 모델이 구두점을 더 잘 이해하고 구분할 수 있음. 단일 공백으로 대체 -> 노이즈 줄이기 

    sentence = re.sub(r"([?.!,])", r" \1 ", sentence) 

    sentence = re.sub(r'[" "]+', " ", sentence) 

  

    # (가-힣 ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체합니다. 

    sentence = re.sub(r"[^가-힣?.!,]+", " ", sentence) #한글 전처리 

    sentence = sentence.strip() 

    return sentence 

  

# 질문과 답변의 쌍인 데이터셋을 구성하기 위한 데이터 로드 함수 

def load_conversations(): 

    inputs, outputs = [], [] 

    for i in range(len(data)): 

        # 전처리 함수를 질문에 해당되는 inputs와 답변에 해당되는 outputs에 적용. 

        inputs.append(preprocess_sentence(data['Q'][i]))  # questions 

        outputs.append(preprocess_sentence(data['A'][i]))  # answers 

  

        if len(inputs) >= MAX_SAMPLES: 

            return inputs, outputs 

  

    return inputs, outputs 

  

#로드한 데이터의 샘플 수를 확인 

# 데이터를 로드하고 전처리하여 질문을 questions, 답변을 answers에 저장합니다. 

questions, answers = load_conversations() 

print('전체 샘플 수 :', len(questions)) 

print('전체 샘플 수 :', len(answers)) 

  

# 질문과 답변 데이터셋에 대해서 Vocabulary 생성. 

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13) 

# 시작 토큰과 종료 토큰에 고유한 정수를 부여합니다. 

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1] 

  

#시작 토큰과 종료 토큰에 부여된 정수를 출력 

print('START_TOKEN의 번호 :' ,[tokenizer.vocab_size]) 

print('END_TOKEN의 번호 :' ,[tokenizer.vocab_size + 1]) 

  

# 시작 토큰과 종료 토큰을 고려하여 +2를 하여 단어장의 크기를 산정합니다. 

VOCAB_SIZE = tokenizer.vocab_size + 2 

print(VOCAB_SIZE) 

  

# 샘플의 최대 허용 길이 또는 패딩 후의 최종 길이 

MAX_LENGTH = 150 

print(MAX_LENGTH) 

  

  

# 정수 인코딩, 최대 길이를 초과하는 샘플 제거, 패딩 

def tokenize_and_filter(inputs, outputs): 

    tokenized_inputs, tokenized_outputs = [], [] 

  

    for (sentence1, sentence2) in zip(inputs, outputs): 

        # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가 

        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN 

        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN 

  

        # 최대 길이 이하인 경우에만 데이터셋으로 허용 

        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH: 

            tokenized_inputs.append(sentence1) 

            tokenized_outputs.append(sentence2) 

  

    # 최대 길이로 모든 데이터셋을 패딩 

    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences( 

        tokenized_inputs, maxlen=MAX_LENGTH, padding='post') 

    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences( 

        tokenized_outputs, maxlen=MAX_LENGTH, padding='post') 

  

    return tokenized_inputs, tokenized_outputs 

  

questions, answers = tokenize_and_filter(questions, answers) 

print('단어장의 크기 :',(VOCAB_SIZE)) 

print('필터링 후의 질문 샘플 개수: {}'.format(len(questions))) 

print('필터링 후의 답변 샘플 개수: {}'.format(len(answers))) 

  

#교사 강요 

BATCH_SIZE = 64 

BUFFER_SIZE = 10000 

  

# 디코더는 이전의 target을 다음의 input으로 사용 

# 이에 따라 outputs에서는 START_TOKEN을 제거 

dataset = tf.data.Dataset.from_tensor_slices(( 

    { 

        'inputs': questions, 

        'dec_inputs': answers[:, :-1] 

    }, 

    { 

        'outputs': answers[:, 1:] 

    }, 

)) 

  

dataset = dataset.cache() 

dataset = dataset.shuffle(BUFFER_SIZE) 

dataset = dataset.batch(BATCH_SIZE) 

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) 
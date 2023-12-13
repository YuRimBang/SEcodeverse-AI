import openai

openai.api_key = '' #조심

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
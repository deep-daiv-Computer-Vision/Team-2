from openai import OpenAI
import os
from datetime import datetime
import json

# OpenAI API Key 로드 (gitignore 에 포함되어 있음)
with open('../key.json', 'r', encoding='utf8') as f:
    data = json.load(f)
    print(data)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=data['open-ai-key'])

# 강의 주제 리스트 (영어)
lecture_topics = [
    "Introduction to Artificial Intelligence and Its History",
    "Core Concepts of Machine Learning",
    "Understanding Deep Learning and Neural Networks",
    "Fundamentals of Natural Language Processing",
    "Computer Vision and Image Processing"
]

def generate_lecture_script(topic):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI education expert. Create a lecture script of approximately 1000 tokens. The script should be in English and suitable for an academic audience. Do not use any markdown formatting, topic sentences, or concluding remarks. Start directly with the content and end with the last point. Write in a continuous, flowing narrative style without any section breaks or numbering. Do not use any markdown formatting in your response."
                },
                {
                    "role": "user",
                    "content": f"Create a direct lecture script about {topic}. Skip any introductory or concluding sentences. Write as a continuous text without any section breaks. Do not use any markdown formatting."
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {str(e)}"

def save_to_file(topic, content):
    
    # 파일명에서 사용할 수 없는 문자 처리
    safe_topic = "".join(x for x in topic if x.isalnum() or x in [' ', '-', '_']).strip()
    filename = f"lecture_{safe_topic}.txt"
    
    # 강의 대본 저장
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filename

def main():
    print("AI Lecture Script Generation Started...")
    
    for topic in lecture_topics:
        print(f"\nGenerating script for {topic}...")
        content = generate_lecture_script(topic)
        filename = save_to_file(topic, content)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main() 
from openai import OpenAI

client = OpenAI(api_key='your_api_key_here')

def generate_report(prompt, model="gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a medical AI specialized in generating radiology reports."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response

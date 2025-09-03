import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import google.generativeai as genai
import requests

from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

class QueryRequest(BaseModel):
    advisor_role: str
    my_role: str
    question: str
    password: str

@app.post("/consult")
async def get_consultation(req: QueryRequest):
    # 密碼驗證
    if req.password != "888":
        raise HTTPException(status_code=403, detail="密碼錯誤，拒絕存取。")
    
    try:
        prompt_template = (
            f"以下是我的身份和問題：\n"
            f"我的身份：{req.my_role}\n"
            f"我的問題：{req.question}\n\n"
            f"請以一位「{req.advisor_role}」的身份，根據上述資訊給我具體的建議。請務必提供詳細、有洞察力的見解。"
        )

        task_gpt = asyncio.create_task(call_gpt(req.advisor_role, prompt_template))
        task_gemini = asyncio.create_task(call_gemini_for_advice(req.advisor_role, prompt_template))
        task_deepseek = asyncio.create_task(call_deepseek(req.advisor_role, prompt_template))

        gpt_response, gemini_advice, deepseek_response = await asyncio.gather(task_gpt, task_gemini, task_deepseek)

        summary_prompt = (
            f"以下是三位不同 AI 顧問針對同一個問題提供的建議。你的任務是擔任總顧問，閱讀並分析這三份建議，然後提供一份統整性的、更全面、更深入的最終分析與結論。請指出它們的異同點、優點，並提出一個綜合性的最佳方案。\n\n"
            f"--- 顧問 1 (GPT) 的建議 ---\n{gpt_response}\n\n"
            f"--- 顧問 2 (Gemini) 的建議 ---\n{gemini_advice}\n\n"
            f"--- 顧問 3 (DeepSeek) 的建議 ---\n{deepseek_response}\n\n"
            f"請開始你的統整分析。"
        )
        
        final_summary = await call_gemini_for_summary(summary_prompt)

        return {
            "gpt_output": gpt_response,
            "gemini_output": gemini_advice,
            "deepseek_output": deepseek_response,
            "final_summary": final_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def call_gpt(role, prompt):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"你是一位「{role}」。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

async def call_gemini_for_advice(role, prompt):
    response = gemini_model.generate_content(
        f"你是一位「{role}」。{prompt}"
    )
    return response.text

async def call_gemini_for_summary(prompt):
    response = gemini_model.generate_content(prompt)
    return response.text

async def call_deepseek(role, prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": f"你是一位「{role}」。"},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise HTTPException(status_code=response.status_code, detail=f"DeepSeek API 錯誤: {response.text}")

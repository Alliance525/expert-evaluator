"""
专家能力评估平台 - 后端服务
用于模型训练数据标注平台，分析专家对项目的匹配度与能力评估
"""

import io
import json
import os
import re
from pathlib import Path
from typing import List

import httpx
import pypdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="专家能力评估平台")

# ── API 配置（支持本地代理模式 和 云端直连模式）──
# API_MODE=proxy  → 调本地 claude-code-proxy（默认）
# API_MODE=openai → 直接调 OpenAI-compatible API
API_MODE = os.environ.get("API_MODE", "proxy")

# 代理模式配置
PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:8082/v1/messages")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "xFDxJyazGB9Ag5t6jP778BMzoueew5GQ_GPT_AK")
PROXY_MODEL = os.environ.get("PROXY_MODEL", "claude-sonnet-4-5")

# 直连模式配置（云端部署时使用）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

DEFAULT_DIMENSIONS = [
    {
        "id": "model_understanding",
        "name": "模型理解",
        "weight": 1.0,
        "description": "对AI/ML模型原理、训练数据、RLHF等流程的理解深度",
    },
    {
        "id": "annotation_ability",
        "name": "标注能力",
        "weight": 1.5,
        "description": "文本/图像/多模态标注经验，标注规范理解与执行能力",
    },
    {
        "id": "skill_level",
        "name": "专业能力深度",
        "weight": 1.0,
        "description": "专业领域中难以速成的底层能力深度，如数学推理、行业知识、编程能力、母语语感等。这类能力无法通过短期培训获得，是评估中最关键的差异化因素",
    },
    {
        "id": "soft_quality",
        "name": "软素质",
        "weight": 0.8,
        "description": "沟通协作、细心严谨、自我管理与学习能力",
    },
    {
        "id": "willingness",
        "name": "意愿度",
        "weight": 1.2,
        "description": "对标注工作的主动性、稳定性及长期参与意愿",
    },
]

SYSTEM_PROMPT = """你是一位专业的数据标注项目专家招募顾问，擅长从候选人背景中提取与标注项目相关的能力信号。
你的分析必须：
1. 基于专家背景原文给出评分依据，尽量引用原文中的具体描述
2. 评分严格客观，信息不足时应给出偏低分数（60分以下），不得虚高
3. 风险点必须具体，不得泛泛而谈
4. 仅输出JSON，不加任何额外文字或markdown代码块"""

EVAL_PROMPT_TEMPLATE = """## 评估任务

**标注项目描述：**
{project_description}

**专家背景：**
{expert_background}

## 评估维度
{dimensions_spec}

## 额外维度：项目匹配度（综合评估专家与本项目的整体契合程度）

## 输出格式（严格JSON，不加任何额外文字）：
{{"dimension_scores":[{{"id":"<与输入id完全一致>","name":"<维度名称>","score":<0-100整数>,"weight":<与输入权重一致>,"evidence":"<原文依据，100字内>","risks":"<风险点，60字内>","suggestions":"<考察或改善建议，60字内>"}}],"project_match":{{"score":<0-100整数>,"evidence":"<匹配依据>","risks":"<不匹配风险>","suggestions":"<缓解建议>"}},"weighted_total":<加权总分，1位小数>,"recommendation":"<强推荐|推荐|待定|不推荐>","recommendation_reason":"<综合结论，150字内>","salary_estimate":{{"range_low":<最低时薪，整数，人民币>,"range_high":<最高时薪，整数，人民币>,"unit":"元/小时","rationale":"<定价理由，80字内，需结合项目难度和专家水平>","task_difficulty":"<低/中/高>"}},"acceptance_probability":{{"level":"<高/中/低>","score":<0-100整数>,"analysis":"<接受意愿分析，100字内，需结合专家当前水平与该薪资水位的匹配度>","concerns":"<主要顾虑点，60字内>"}},"training_cost_summary":{{"level":"<低/中/高>","estimated_days":<预计培训天数，整数>,"trainable_gaps":["<可通过培训弥补的能力缺口>"],"non_trainable_gaps":["<短期难以培训的底层能力缺口，如数学推理/行业知识/语言语感等>"],"onboarding_plan":"<建议的上手路径，80字内>"}}}}

评分标准（通用）：85-100有充分直接经验 | 70-84经验略有不足 | 55-69需培训 | 40-54风险较高 | 0-39几乎无相关背景

专业能力深度维度说明：重点考察难以通过培训速成的底层能力（数学推理、行业专业知识、母语语感、逻辑判断力等）。这类能力分低代表无法弥补的硬性缺口；分高代表专家具备深厚的、不可复制的专业基础。

培训成本派生说明：training_cost_summary 不是独立评分维度，而是综合所有维度缺口后推导出的分析。其中：
- trainable_gaps：标注流程、工具使用、规范理解等可通过 1-4 周培训弥补的缺口
- non_trainable_gaps：专业能力深度、特定行业知识、数学推理等短期无法速成的缺口（直接来源于各维度的低分项）
- estimated_days：根据 trainable_gaps 的数量和复杂度估算，non_trainable_gaps 不计入（视为不可解决的风险）

薪资预估说明：结合项目标注难度（需专业判断/逻辑推理/领域知识的任务定价更高）与专家稀缺性，给出合理时薪区间。参考：普通文本标注30-60元/h，需专业背景的标注80-150元/h，高难度专家标注150-300元/h。

接受意愿说明：结合专家当前职业水平推断其期望薪资，判断预估薪资水位对该专家的吸引力，以及标注工作性质（兼职/全职/短期项目）是否符合其职业阶段。"""


class DimensionDef(BaseModel):
    id: str
    name: str
    weight: float = Field(default=1.0, ge=0.1, le=10.0)
    description: str = ""


class EvaluateRequest(BaseModel):
    expert_background: str
    project_description: str
    dimensions: List[DimensionDef] = []


def build_dimensions_spec(dims: List[DimensionDef]) -> str:
    lines = []
    for i, d in enumerate(dims, 1):
        line = f"{i}. 【{d.name}】id={d.id}，权重={d.weight}"
        if d.description:
            line += f"\n   说明：{d.description}"
        lines.append(line)
    return "\n".join(lines)


@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")
    try:
        contents = await file.read()
        reader = pypdf.PdfReader(io.BytesIO(contents))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            raise HTTPException(status_code=422, detail="PDF 无法提取文字，可能是扫描件")
        return JSONResponse({"text": text})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {e}")


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/default-dimensions")
async def get_default_dimensions():
    return JSONResponse(DEFAULT_DIMENSIONS)


@app.post("/evaluate")
async def evaluate_expert(req: EvaluateRequest):
    if len(req.expert_background.strip()) < 30:
        raise HTTPException(status_code=400, detail="专家背景内容太少，请补充更多信息")
    if len(req.project_description.strip()) < 10:
        raise HTTPException(status_code=400, detail="请填写项目描述")

    dims = req.dimensions if req.dimensions else [DimensionDef(**d) for d in DEFAULT_DIMENSIONS]

    prompt = EVAL_PROMPT_TEMPLATE.format(
        project_description=req.project_description[:3000],
        expert_background=req.expert_background[:10000],
        dimensions_spec=build_dimensions_spec(dims),
    )

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            if API_MODE == "openai":
                # 直连 OpenAI-compatible API
                resp = await client.post(
                    f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                    json={
                        "model": OPENAI_MODEL,
                        "max_tokens": 3000,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                    },
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Api-Key": OPENAI_API_KEY,
                        "content-type": "application/json",
                    },
                )
            else:
                # 本地代理（Anthropic Messages 格式）
                resp = await client.post(
                    PROXY_URL,
                    json={
                        "model": PROXY_MODEL,
                        "max_tokens": 3000,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    headers={
                        "x-api-key": PROXY_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接 AI 服务，请检查网络或配置")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"AI 服务返回错误: {e.response.status_code}")

    try:
        if API_MODE == "openai":
            content = data["choices"][0]["message"]["content"]
        else:
            content = data["content"][0]["text"]
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            raise ValueError("未找到JSON内容")
        result = json.loads(json_match.group())
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"AI响应解析失败: {e}")

    # 后端重算加权总分，防止模型算错
    try:
        scores = result["dimension_scores"]
        # 截断越界分数
        for s in scores:
            s["score"] = max(0, min(100, int(s["score"])))
        total_w = sum(s["weight"] for s in scores)
        if total_w > 0:
            result["weighted_total"] = round(
                sum(s["score"] * s["weight"] for s in scores) / total_w, 1
            )
    except (KeyError, ZeroDivisionError, TypeError):
        pass

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8091))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)

# -*- coding: utf-8 -*-
"""
专家能力评估平台 - 后端服务
用于模型训练数据标注平台，分析专家对项目的匹配度与能力评估
"""

import io
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

import httpx
import pypdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="专家能力评估平台")

# ── SQLite 历史记录 ──
DB_PATH = Path(__file__).parent / "history.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            expert_name TEXT DEFAULT '',
            project_description TEXT NOT NULL,
            expert_background TEXT NOT NULL,
            recommendation TEXT,
            weighted_total REAL,
            project_match_score INTEGER,
            result_json TEXT NOT NULL
        )
    """)
    # 兼容旧表结构
    try:
        con.execute("ALTER TABLE evaluations ADD COLUMN expert_name TEXT DEFAULT ''")
    except Exception:
        pass
    con.commit()
    con.close()

def save_evaluation(project_desc: str, expert_bg: str, result: dict, expert_name: str = ""):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO evaluations (created_at, expert_name, project_description, expert_background, recommendation, weighted_total, project_match_score, result_json) VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.now().isoformat(timespec="seconds"),
            expert_name or result.get("expert_name", ""),
            project_desc[:500],
            expert_bg[:500],
            result.get("recommendation", ""),
            result.get("weighted_total"),
            result.get("project_match", {}).get("score"),
            json.dumps(result, ensure_ascii=False),
        ),
    )
    con.commit()
    con.close()

init_db()

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
        "description": "对项目工作的主动性、稳定性及长期参与意愿",
    },
    {
        "id": "adaptability",
        "name": "项目适配潜力",
        "weight": 0.6,
        "description": "对项目特定流程、工具和规范的理解与适应能力（如标注规范、访谈框架、质检流程等）。此类能力可通过1-4周培训获得，不作硬性门槛，重点评估可培训性与学习速度",
    },
]

SYSTEM_PROMPT = """你是一位专业的人才招募顾问，擅长从候选人背景中提取与项目相关的能力信号，包括直接经验和潜力迁移。
你的分析必须：
1. 优先引用原文中的具体描述作为评分依据
2. 当直接证据不足时，主动识别可迁移的潜力信号（如行业知识、相关技能、学习轨迹、跨界经验等），并明确标注为推测性评分（is_inferred=true）
3. 推测性评分上限为75分，须在evidence中说明"推测依据：xxx"
4. 评分严格客观，无任何间接信号时方可给低分（40分以下）
5. 风险点必须具体，不得泛泛而谈
6. 区分"硬性能力缺口"（无法速成）与"可培训缺口"（能通过培训弥补）
7. 仅输出JSON，不加任何额外文字或markdown代码块"""

EVAL_PROMPT_TEMPLATE = """## 评估任务

**标注项目描述：**
{project_description}

**专家背景：**
{expert_background}

## 评估维度
{dimensions_spec}

## 额外维度：项目匹配度（综合评估专家与本项目的整体契合程度）

## 输出格式（严格JSON，不加任何额外文字）：
{{"expert_name":"<从背景中识别的专家姓名，无法确认则返回空字符串>","potential_signals":[{{"signal":"<发现的可迁移潜力信号，如某种间接经验或跨界技能>","relevance":"<与本项目的关联性推断，40字内>"}}],"dimension_scores":[{{"id":"<与输入id完全一致>","name":"<维度名称>","score":<0-100整数>,"weight":<与输入权重一致>,"is_inferred":<true表示此分基于潜力推测，false表示有直接证据>,"evidence":"<若is_inferred=false则引用原文；若true则写'推测依据：xxx'，100字内>","risks":"<风险点，60字内>","suggestions":"<考察或改善建议，60字内>"}}],"project_match":{{"score":<0-100整数>,"evidence":"<匹配依据>","risks":"<不匹配风险>","suggestions":"<缓解建议>"}},"weighted_total":<加权总分，1位小数>,"recommendation":"<强推荐|推荐|待定|不推荐>","recommendation_reason":"<综合结论，150字内>","salary_estimate":{{"range_low":<最低时薪，整数，人民币>,"range_high":<最高时薪，整数，人民币>,"unit":"元/小时","rationale":"<定价理由，80字内，需结合项目难度和专家水平>","task_difficulty":"<低/中/高>"}},"acceptance_probability":{{"level":"<高/中/低>","score":<0-100整数>,"analysis":"<接受意愿分析，100字内，需结合专家当前水平与该薪资水位的匹配度>","concerns":"<主要顾虑点，60字内>"}},"training_cost_summary":{{"level":"<低/中/高>","estimated_days":<预计培训天数，整数>,"trainable_gaps":["<可通过培训弥补的能力缺口>"],"non_trainable_gaps":["<短期难以培训的底层能力缺口，如数学推理/行业知识/语言语感等>"],"onboarding_plan":"<建议的上手路径，80字内>"}}}}

评分标准（通用）：
- 85-100：有充分直接经验，直接可用
- 70-84：经验略有不足，但有明确迁移路径
- 55-75：无直接经验但有潜力信号（is_inferred=true，推测性评分上限75）
- 40-54：缺口较大，风险较高
- 0-39：几乎无相关背景或迁移可能

潜力推测规则：当专家无直接经验时，主动寻找以下信号：
① 领域相邻经验（如游戏设计→AI交互体验判断；心理学背景→用户访谈）
② 高强度学习记录（自学项目、跨界转型、快速晋升轨迹）
③ 底层能力迁移（逻辑训练→标注判断；写作能力→文本质量评估）
找到迁移信号则设is_inferred=true，在evidence中说明推测逻辑，得分不超过75。

专业能力深度维度说明：重点考察难以通过培训速成的底层能力（数学推理、行业专业知识、母语语感、逻辑判断力等）。此维度应积极寻找间接潜力信号。

项目适配潜力维度说明：评估专家对本项目特定流程/工具/规范的学习适配能力。此维度得分低不影响整体推荐（权重低），重点看是否有快速上手的条件。

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
    expert_name: str = ""


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


class WeightRecommendRequest(BaseModel):
    project_description: str
    dimensions: List[DimensionDef]


WEIGHT_PROMPT = """你是一位专业的人才招募顾问，需要根据项目描述为候选人评估维度分配合理的权重。

**项目描述：**
{project_description}

**当前评估维度：**
{dimensions_spec}

请分析该项目的核心诉求，为每个维度给出推荐权重（范围 0.5–3.0）。
权重越高代表该维度对项目越关键。所有维度权重之和建议在 4–8 之间。

请同时识别该项目的类型（如：数据标注、调研访谈、内容审核、翻译、专业咨询等）。

仅输出JSON，不加任何额外文字：
{{"project_type":"<项目类型>","rationale":"<整体分析，60字内>","weights":[{{"id":"<维度id>","weight":<推荐权重，1位小数>,"reason":"<推荐理由，30字内>"}}]}}"""


@app.post("/api/recommend-weights")
async def recommend_weights(req: WeightRecommendRequest):
    if len(req.project_description.strip()) < 10:
        raise HTTPException(status_code=400, detail="请先填写项目描述")

    dims_spec = "\n".join(
        f"- id={d.id}，名称={d.name}，当前权重={d.weight}，说明={d.description or '无'}"
        for d in req.dimensions
    )
    prompt = WEIGHT_PROMPT.format(
        project_description=req.project_description[:2000],
        dimensions_spec=dims_spec,
    )

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            if API_MODE == "openai":
                resp = await client.post(
                    f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                    json={"model": OPENAI_MODEL, "max_tokens": 800,
                          "messages": [{"role": "user", "content": prompt}]},
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                             "Api-Key": OPENAI_API_KEY, "content-type": "application/json"},
                )
            else:
                resp = await client.post(
                    PROXY_URL,
                    json={"model": PROXY_MODEL, "max_tokens": 800,
                          "messages": [{"role": "user", "content": prompt}]},
                    headers={"x-api-key": PROXY_API_KEY,
                             "anthropic-version": "2023-06-01", "content-type": "application/json"},
                )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接 AI 服务")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"AI 服务返回错误: {e.response.status_code}")

    try:
        content = data["choices"][0]["message"]["content"] if API_MODE == "openai" else data["content"][0]["text"]
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise ValueError("未找到JSON")
        return JSONResponse(json.loads(match.group()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {e}")


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

    save_evaluation(req.project_description, req.expert_background, result, req.expert_name)
    return JSONResponse(result)


@app.get("/api/history")
async def get_history():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, created_at, expert_name, project_description, expert_background, recommendation, weighted_total, project_match_score FROM evaluations ORDER BY id DESC LIMIT 100"
    ).fetchall()
    con.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/history/{eval_id}")
async def get_history_item(eval_id: int):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM evaluations WHERE id=?", (eval_id,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="记录不存在")
    d = dict(row)
    d["result"] = json.loads(d.pop("result_json"))
    return JSONResponse(d)


@app.delete("/api/history/{eval_id}")
async def delete_history_item(eval_id: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM evaluations WHERE id=?", (eval_id,))
    con.commit()
    con.close()
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8091))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)

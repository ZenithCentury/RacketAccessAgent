import os
import json
import logging
import ssl
import sys
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Union
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

_CURRENT_DIR = os.path.dirname(__file__)           #当前脚本目录
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir))    #当前脚本父目录即项目根目录
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)        #项目根目录导入到搜索路径

from base import get_agent_logger, truncate              #get_agent_logger()：标准化日志记录器。
                                                        #truncate 截断文本

from acps_aip.aip_base_model import TaskState
from acps_aip.aip_rpc_client import AipRpcClient
from acps_aip.mtls_config import load_mtls_config_from_json

"""环境与基础变量配置"""

load_dotenv()     #加载环境变量

#领导 Agent 标志
LEADER_ID = os.getenv("LEADER_AGENT_ID", "racket-leader")  #racket_leader
LOG_LEVEL = os.getenv("LEADER_LOG_LEVEL", "DEBUG").upper()      #日志级

#本目录
CURRENT_DIR = os.path.dirname(__file__)
#仓库根目录
ROOT_DIR = _PROJECT_ROOT

#OpenAI 大模型配置 兼容简单base_url 模式
openai.api_key = os.getenv("OPENAI_API_KEY")
_raw_base_url = os.getenv("OPENAI_BASE_URL")
if _raw_base_url and not _raw_base_url.endswith("/"):
    _raw_base_url += "/"
openai.base_url = _raw_base_url
LLM_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
DISCOVERY_BASE_URL = os.getenv("DISCOVERY_BASE_URL")          #发先服务地址
_DISCOVERY_TIMEOUT_SECONDS = os.getenv("DISCOVERY_TIMEOUT")

"""FastAPI应用"""
app = FastAPI(
    title='Racket Assistant (Leader)',
    description='首要面向用户：分析用户需求 ——》 调度 Partner ——》 整合结果',
)

#leader日志初始化创建
logger = get_agent_logger("agent.racket_leader", "LEADER_LOG_LEVEL", LOG_LEVEL)
logger.info(
    'envent=app_start leader_id=%s model=%s log_level=%s',
    LEADER_ID,
    LLM_MODEL,
    LOG_LEVEL
)  #这么写机器更好读？

""""mTLS 配置加载"""

_mtls_json_path = os.path.join(CURRENT_DIR, "racket_leader.json")
_mtls_config = load_mtls_config_from_json(_mtls_json_path)
_client_ssl_context = _mtls_config.create_client_ssl_context()         #创建ssl客户端上下文返回配置好的上下文对象
logger.info(
    'envent=mtls_config_loaded aic=%s cert_dir=%s',
    _mtls_config.aic,
    _mtls_config.cert_dir,
)

"""将会话与上下文存储"""

sessions: Dict[str, Dict[str, Any]] = {}    #以键 字典 型储存会话数据

class UserRequest(BaseModel):
    """用户请求数据模型：允许复用已有 session，或创建新的会话"""
    session_id: str
    query: str

#racket_assess_partner  user_assess_partner  racket_recommender_partner
#partner组成
_RACKET_DISCOVERY_QUERIES: Dict[str, str] = {
    "racket_assess": "球拍性能评价智能体",
    "user_assess": "适合用户评价智能体",
}

#启动智能体
def _load_static_agent(agent_type: str) -> Optional[dict]:
    mapping = {
        "racket_assess": ("racket_assess", "racket_assess_partner.json"),
        "user_assess": ("user_assess", "user_assess_partner.json"),
    }
    entry = mapping.get(agent_type)
    if not entry:
        return None
    file_path = os.path.join(ROOT_DIR, entry[0], entry[1])

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(
                "event=service_discovery_static_success agent_type=%s file=%s",
                agent_type,
                file_path,
            )
            return data
    except FileNotFoundError:
        logger.error(
            "event=service_discovery_static_missing agent_type=%s file=%s",
            agent_type,
            file_path,
        )
    except json.JSONDecodeError:
        logger.error(
            "event=service_discovery_static_invalid_json agent_type=%s file=%s",
            agent_type,
            file_path,
        )
    return None

def _discover_agent(agent_type: str) -> Optional[dict]:
    if not DISCOVERY_BASE_URL:
        logger.debug(
            "event=discovery_skipped reason=no_base_url agent_type=%s", agent_type
        )
        return None
    query = _RACKET_DISCOVERY_QUERIES.get(agent_type)
    if not query:
        return None
    url = DISCOVERY_BASE_URL.rstrip("/")               #http://bupt.ioa.pub:8005/api/discovery
    try:
        with httpx.Client(timeout=_DISCOVERY_TIMEOUT_SECONDS) as client:
            resp = client.post(url, json={"query": query, "limit": 1})
            resp.raise_for_status()        #检查状态码
            payload = resp.json()           #解析状态相应 将JSON响应体解析为Python字典
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.warning(
            "event=discovery_request_failed agent_type=%s error=%s",
            agent_type,
            exc,
        )
        return None

    agents = payload.get("agents") if isinstance(payload, dict) else None            #如果解析成功 返回agents 内容
    if not agents:
        logger.warning(
            "event=discovery_request_failed agent_type=%s payload_preview=%s",
            agent_type,
            truncate(str(payload), 120),              #通信不成功则返回有限的json格式resp预览排错
        )
        return None

    first = agents[0]        #待观察////////////////////
    candidate = None
    if isinstance(first, dict):
        if "acs" in first and isinstance(first["acs"], dict):      #格式正确
            candidate = first["acs"]
        elif "acs" in first and not isinstance(first["acs"], dict):        #格式错误
            try:
                candidate = json.loads(first["acs"])
            except json.JSONDecodeError as exc:
                logger.warning(
                    "event=discovery_request_failed agent_type=%s error=%s",
                    agent_type,
                    exc
                )

    if candidate:
        logger.info(
            "event=discovery_request_success agent_type=%s aic=%s",
            agent_type,
            candidate.get("aic"),
        )
        return candidate

    logger.warning(
        "event=discovery_request_unusable agent_type=%s first_preview=%s",
        agent_type,
        truncate(str(first), 120),
    )
    return None

"""优先通过 discovery-server 动态发现下级Partner Agent，失败时回退到本地静态配置。"""
def find_agent_service(agent_type: str) -> Optional[dict]:
    if agent_type in _RACKET_DISCOVERY_QUERIES:
        discovered = _discover_agent(agent_type)
        if discovered:
            return discovered
    return _load_static_agent(agent_type)

"""从能力描述中提取 JSONRPC 的 endpoint URL。
期望结构示例（可能的变体）:
    {
      "endPoints": [
         {"transport":"JSONRPC","url":"http://host:8011/acps-aip-v1/rpc"},
         {"transport":"SSE","url":"..."}
      ]
    }

    返回首个 transport == JSONRPC 的 url。选择接受由jsonrpc协议内容
"""
def extract_jsonrpc_endpoint(agent_info: dict) -> Optional[str]:           #获取对应jsonrpc协议发来自url
    if not agent_info:
        return None
    endpoint = (
        agent_info.get('endpoints')
        or agent_info.get('endPoints')
        or agent_info.get('endpoint')
        or agent_info.get('endPoint')
        or []
    )
    # 如果是字典格式（例如 {"rpc": {..}, "sse": {..}}） 则转为列表   仅保留值
    if isinstance(endpoint, dict):
        endpoint = list(endpoint.values())                  #此时会的到形如endpoint=[{"transport":"JSONRPC","url":"http://host:8011/acps-aip-v1/rpc"}]
    if not isinstance(endpoint, list):      #说明格式错了 也不是字典
        return None
    for ep in endpoint:
        if not isinstance(ep, dict):
            continue
        #此时ep 应为{"transport":"JSONRPC","url":"http://host:8011/acps-aip-v1/rpc"}型
        transport = str(ep.get("transport","")).upper()
        if transport == "JSONRPC":
            url = ep.get("url") or ep.get("URI") or ep.get("endpoint")
            if url:
                logger.debug("event=extract_jsonrpc_endpoint url=%s", url)
                return url
    return None

"""多阶段提示词模板"""
ANALYSIS_PROMPT_TEMPLATE = """
在这个针对球拍性能分析的工作中，你是核心的路由负责人，负责对用户输入进行三个维度拆解：
1) racket_assess 这个球拍本身是什么样的，核心分析 拍重weight:轻/重/适中；中杆toughness：软/硬/适中；最高磅数max_pounds:能拉高磅/不能拉高磅；
拍长length较长/较短；平衡点balance_point：高/低；评分score:相对高/相对低；厂家brand：是属于李宁，Yonex，Victor这三大厂/不属于三大厂；
价格price_range：高/中/低。
2) user_assess 这个球拍适合什么特点的球友，这与1中的racket_assess结果息息相关，价格高/低对应水平高/低；拍重轻/重对应速度型球友/力量型球友；
属于三大厂意味着适合关心售后保障的球友；中杆软/硬对应初阶型球友/进阶型球友；最高磅数高意味着更符合高水平球员的需要；平衡点高/低对应进攻型球友/防守型球友；
拍长较长/较短对应主单打球友/主双打球友。

当前已部署可用代理（agentId）
 - racket_assess_partner_001
 - user_assess_partner_001


【输出要求】
严格输出 JSON（无多余文本），结构如下（不要出现注释）：
{
  "target_racket": "...",
  "request_type": "new_request|modify_request|add_detail|question",
  "user_needs": {
     "user_characteristics": "...",
     "budget": "...",
     "preferences": ["..."],
     "special_requirements": "..."
  },
  "dimensions": {
     "racket_assess": {"needed": true,  "reason": "...", "sub_query": "..."},
     "user_assess":              {"needed": true,  "reason": "...", "sub_query": "..."},
  },
  "required_agents": ["racket_assess_partner_001", "user_assess_partner_001"],
  "unavailable_dimensions": [],
  "routing_reason": "总体路由理由",
  "notes": "补充说明，可为空"
}

规范：
1. 所有 needed=false 仍需给出 reason；若 needed=true 必须给出 sub_query（可精炼原始需求仅保留该维度相关要素）。
2. sub_query：避免包含其它维度的冗余内容，使用简洁指令式语句。
3. preferences 用字符串数组，不足则空数组。缺失字段给空字符串或空数组，不要 null。
4. 不允许添加未定义的顶层字段。

【历史上下文】
{context}

【当前请求】
{query}
"""

INTEGRATION_PROMPT_TEMPLATE = """你是关于本球拍信息、适用用户、面向用户推荐这三项结果的整合助手，请基于分析结果、分解子查询与多个智能体产出，生成用户可直接采用的最终方案。

【会话上下文】
{context}

【用户请求】
{query}

【需求分析 JSON】
{analysis_json}

【任务分解（decomposition）】
{decomposition_block}

【各智能体结果汇总】
{partner_results_block}

【整合要求】
1. 若 request_type = new_request：提供 球拍属性简览 / 适合用户说明 / 向提问者推荐 说明（若多代理）。
2. 若为 modify_request 或 add_detail：说明变化点，并给出更新后的完整方案。
3. 多代理时需合并去重。
4. 重点突出亮点与差异化体验，避免简单拼接；删除互相冲突或重复的说法。
5. 结构清晰，使用分级标题或条目。无信息的板块不输出标题。

直接输出最终方案文本（不要再输出 JSON）。"""


"""会话与辅助         从案例上抄的"""
# 简单占位符替换工具：仅替换已知 {name}，不会解析其它大括号，避免 JSON 中的花括号触发 format KeyError
def _fill_template(tpl: str, **kwargs) -> str:
    for k, v in kwargs.items():
        tpl = tpl.replace("{" + k + "}", v)
    return tpl

#获取或创建对话
def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],  # [{role, content, time}]
            "last_analysis": None,
            # 记录与各 Partner 的进行中任务与状态
            # partner_tasks[agent_id] = {"task_id": str, "state": TaskState, "last_product": str|None,
            #                            "awaiting_prompt": str|None, "updated_at": iso, "sub_query": str|None}
            "partner_tasks": {},
            # 当前等待用户补充的信息队列（用于前端提示）
            # [{"agent_id": str, "task_id": str, "question": str, "time": iso}]
            "pending_questions": [],
        }
    return sessions[session_id]

def append_message(session_id: str, role: str, content: str):
    sess = get_session(session_id)
    sess["messages"].append(
        {
            "role": role,
            "content": content,
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )


def _extract_text_from_data_items(items: Optional[List[Any]]) -> str:
    if not items:
        return ""
    texts: List[str] = []
    for di in items:
        try:
            # TextDataItem has attribute 'text'
            t = getattr(di, "text", None)
            if t:
                texts.append(str(t))
        except Exception:
            continue
    return "\n".join(texts).strip()


def _save_partner_task_state(
    session_id: str,
    agent_id: str,
    task_id: str,
    state: Any,
    sub_query: Optional[str] = None,
    last_product: Optional[str] = None,
    awaiting_prompt: Optional[str] = None,
):
    sess = get_session(session_id)
    sess["partner_tasks"][agent_id] = {
        "task_id": task_id,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_product": last_product,
        "awaiting_prompt": awaiting_prompt,
        "sub_query": sub_query
        or sess["partner_tasks"].get(agent_id, {}).get("sub_query"),
    }
    # 若处于 AwaitingInput，同时记录 pending question 供前端展示
    sess.setdefault("pending_questions", [])
    sess["pending_questions"] = [
        item for item in sess["pending_questions"] if item.get("agent_id") != agent_id
    ]
    if state == TaskState.AwaitingInput and awaiting_prompt:
        sess["pending_questions"].append(
            {
                "agent_id": agent_id,
                "task_id": task_id,
                "question": awaiting_prompt,
                "time": datetime.now(timezone.utc).isoformat(),
            }
        )

def _build_continue_payload(
    *,
    user_input: str,
    agent_id: Optional[str] = None,
    sub_query: Optional[str] = None,
    awaiting_prompt: Optional[str] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Package the Continue payload with helpful context for the partner agent."""
    # Keep the message human-readable so downstream LLM partners can interpret it easily.
    sections: list[str] = []
    if sub_query:
        sections.append(f"子任务指令：{sub_query.strip()}")
    if awaiting_prompt:
        sections.append(f"待补充问题：{awaiting_prompt.strip()}")
    sections.append(f"用户回复：{user_input.strip()}")
    return "\n\n".join(sections)

# 删除 LLM 字段抽取：Partner 现已支持在 Continue 时自行解析/合并补充

def build_context(session_id: str) -> str:
    sess = get_session(session_id)
    msgs = sess["messages"][-5:]  # 只取最近 5 条对话
    parts = ["=== 最近对话 ==="]
    for m in msgs:
        cn_role = "用户" if m["role"] == "user" else "助理"
        parts.append(f"{cn_role}: {m['content']}")
    return "\n".join(parts)

###################################################
"""LLM 调用阶段函数     将会得到一完整任务给大语言模型"""
###################################################

def llm_analysis(query: str, session_id: str) -> Dict[str, Any]:
    """阶段1  需求分析 """
    context = build_context(session_id)
    #内容替换需求提示内容
    prompt = _fill_template(ANALYSIS_PROMPT_TEMPLATE, context=context, query=query)         #这边不是很懂-------

    """# 如果text 中包含球拍相关的关键字
text1 = "这个球拍拍长多少啊，适合打单打吗"
dim1 = {}
dim1["racket_assess"] = any(k in text1 for k in RACKET_ATTR_KWS)  # True ("拍长"在text中)
dim1["user_assess"] = any(k in text1 for k in USER_CHAR_KWS)      # True ("用户"在text中)
print(dim1)  # 输出: {'racket_assess': True, 'user_assess': True}"""

    #启发式意图分析
    RACKET_ATTR_KWS = [
        "质量",
        "怎样",
        "属性",
        "特点",
        "拍长",
        "拍重",
        "价格",
        "厂家",
        "评分",
        "中杆",
        "磅数",
        "平衡点",
    ]

    USER_CHAR_KWS = [
        "球友",
        "球员",
        "适合",
        "技术",
        "力量",
        "速度",
        "单打",
        "双打",
        "售后",
        "进攻",
        "防守",
        "高手",
        "新手",
    ]

    #启发词检测
    def heuristic_detect(text: str) -> Dict[str, Any]:
        lower = text.lower() #大转小
        dim = {}
        dim["racket_assess"] = any(k in text for k in RACKET_ATTR_KWS)
        dim["user_assess"] = any(k in text for k in USER_CHAR_KWS)

        # 都不包含那就都说 纯夸
        if not any(dim.values()):
            dim["racket_assess"] = True
            dim["user_assess"] = True

        return {"dim_flags": dim}

    heur_basic = heuristic_detect(query)

    """对应格式：
    
    {
  "target_racket": "...",
  "request_type": "new_request|modify_request|add_detail|question",
  "user_needs": {
     "user_characteristics": "...",
     "budget": "...",
     "preferences": ["..."],
     "special_requirements": "..."
  },
  "dimensions": {
     "racket_assess": {"needed": true,  "reason": "...", "sub_query": "..."},
     "user_assess":              {"needed": true,  "reason": "...", "sub_query": "..."},

  },
  "required_agents": ["racket_assess_partner_001", "user_assess_partner_001", "racket_recommender_partner_001"],
  "unavailable_dimensions": [],
  "routing_reason": "总体路由理由",
  "notes": "补充说明，可为空"
}
    
    """
    #规划子任务描述
    def build_sub_query(demension: str, original: str) -> str:
        prefix_map = {
            "racket_assess": "请描述本球拍的各项数据，侧重于该数据是属于较高值，较低值，还是相对平均。",
            "user_assess": "请根据本球拍数据分析其适合的球友。",
        }
        return prefix_map.get(demension, "请针对该维度生成子任务:")  + " " + original



    llm_data: Optional[Dict[str, Any]] = None
    logger.info(
        "event=analysis_start session_id=%s query_chars=%d", session_id, len(query)
    )
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=1100,
        )
        content = resp.choices[0].message.content.strip()
        logger.info(
            "event=analysis_llm_end session_id=%s raw_chars=%d preview=%s",
            session_id,
            len(content),
            truncate(content.replace("\n", " "), 160),
        )
        try:
            llm_data = json.loads(content)
            logger.debug(
                "event=analysis_json_parse_success session_id=%s keys=%s",
                session_id,
                list(llm_data.keys()),
            )
        except json.JSONDecodeError:
            logger.error(
                "event=analysis_json_parse_error session_id=%s preview=%s",
                session_id,
                truncate(content, 120),
            )
            llm_data = None
    except Exception as e:
        logger.exception("event=analysis_llm_exception session_id=%s", session_id)
        llm_data = None

    if llm_data is None:
        # ---- Heuristic fallback structure ----
        logger.info(
            "event=analysis_fallback_heuristic session_id=%s dim_flags=%s",
            session_id,
            heur_basic["dim_flags"],
        )
        flags = heur_basic["dim_flags"]

        dimensions_obj = {}
        for d, enabled in flags.items():
            dimensions_obj[d] = {
                "needed": bool(enabled),
                "reason": (
                    "启发式判定包含相关关键词" if enabled else "未检测到相关关键词"
                ),
                **({"sub_query": build_sub_query(d, query)} if enabled else {}),
            }


        llm_data = {
            "target_racket": "this_racket",         ######################################接前端！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            "request_type": "new_request",
            "user_needs": {
                "user_characteristics": "",
                "budget": "",
                "preferences": [],
                "special_requirements": query,
            },
            "dimensions": dimensions_obj,
            "routing_reason": "LLM解析失败，使用启发式结果。",
            "notes": "fallback",
        }

# """预设partner：
# 当前已部署可用代理（agentId）
#  - racket_assess_partner_001
#  - user_assess_partner_001
#  - racket_recommender_partner_001
# # "dimensions": {
#      "racket_assess": {"needed": true,  "reason": "...", "sub_query": "..."},
#      "user_assess":              {"needed": true,  "reason": "...", "sub_query": "..."},
#      "racket_recommender":              {"needed": true, "reason": "...", "sub_query": "..."},

# """
#


    #确定参与partner
    dims = llm_data.get("dimensions") or {}
    available_agents: List[str] = []
    unavailable_dimensions: List[str] = []

    if dims["racket_assess"]["needed"]:
        available_agents.append("racket_assess_partner_001")
    else:
        unavailable_dimensions.append("racket_assess")
    if dims["user_assess"]["needed"]:
        available_agents.append("user_assess_partner_001")
    else:
        unavailable_dimensions.append("user_assess")

    llm_data["required_agents"] = available_agents
    llm_data["unavailable_dimensions"] = unavailable_dimensions
    # For backwards compatibility keep task_priority same order
    llm_data["task_priority"] = available_agents[:]
    logger.info(
        "event=analysis_resolved session_id=%s agents=%s unavailable=%s",
        session_id,
        ",".join(available_agents) or "-",
        ",".join(unavailable_dimensions) or "-",
    )

    get_session(session_id)["last_analysis"] = llm_data
    return llm_data

#异步分发任务
async def call_partner(
    agent_id: str, user_query: str, session_id: str, *, sub_query: Optional[str] = None
) -> Dict[str, Any]:
    """通用调用函数：根据 agent_id 查找 ACS → 提取 JSONRPC → 发起 AIP RPC 交互。

       返回结构:
         {
           "agent_id": agent_id,  对应llm_data 中available_agents

           "success": bool,
           "state": state_value,
           "product_text": str|None,
           "raw_task": task_json|None,
           "error": 错误信息（可选）
         }
       """

    #把内部id映射为查找文件所要的key
    look_up_key = None
    if agent_id == "racket_assess_partner_001":
        look_up_key = "racket_assess"
    elif agent_id == "user_assess_partner_001":
        look_up_key = "user_assess"

    if not look_up_key:
        return {"agent_id":agent_id,
                "success": False,
                "state": "unknown_agent",
                "product_text": None,
                "raw_task": None,
                "error": "unsupported_agent",
        }

    logger.info(
        "event=call_partner session_id=%s agent_id=%s look_up_key=%s",
            session_id,
            agent_id,
            look_up_key)

    agent_info = find_agent_service(look_up_key) #返回agent[0][asc]内容  来自于动态或静态

    partner_url = extract_jsonrpc_endpoint(agent_info) if agent_info else None
    if not partner_url:
        logger.error(
            "event=call_partner_endpoint_missing session_id=%s agent_id=%s",
            session_id,
            agent_id,
        )
        return{
            "agent_id": agent_id,
            "success": False,
            "state": "unavailable",
            "product_text": None,
            "raw_task": None,
            "error": f"{agent_id} JSONRPC endpoint not found",
        }

    client = AipRpcClient(
        partner_url=partner_url, leader_id=LEADER_ID, ssl_context=_client_ssl_context
    )
        # 新建任务
        # 程序会在这里暂停（但不会阻塞整个线程）
        #
        # 等待start_task()
        # 方法完成并返回结果
        #
        # 在此期间，事件循环可以处理其他任务
    try:
        # 新建任务：Start
        task = await client.start_task(session_id=session_id, user_input=user_query)
        logger.debug(
            "event=partner_task_started session_id=%s agent_id=%s task_id=%s state=%s",
            session_id,
            agent_id,
            task.id,
            task.status.state,
        )
        # 轮询与状态驱动
        max_loops = 60  # 最长 60s 轮询（每秒一次）
        loops = 0
        last_state = None
        did_auto_continue = False  # 避免在 AwaitingInput 时重复持续发送 Continue
        while True:
            state = task.status.state
            if state != last_state:
                logger.debug(
                    "event=partner_state_change session_id=%s agent_id=%s task_id=%s state=%s",
                    session_id,
                    agent_id,
                    task.id,
                    state,
                )
                last_state = state
            # 保存当前状态
            _save_partner_task_state(
                session_id,
                agent_id,
                task.id,
                state,
                sub_query=sub_query,
                last_product=(
                    _extract_text_from_data_items(task.products[0].dataItems)
                    if task.products and task.products[0].dataItems
                    else None
                ),
                awaiting_prompt=_extract_text_from_data_items(task.status.dataItems),
            )

            # 终态
            if state in (
                    TaskState.Completed,
                    TaskState.Canceled,
                    TaskState.Failed,
                    TaskState.Rejected,
            ):
                break

            # 需要用户/领导补充
            if state == TaskState.AwaitingInput:
                prompt_text = _extract_text_from_data_items(task.status.dataItems)
                logger.info(
                    "event=partner_awaiting_input session_id=%s agent_id=%s prompt=%s",
                    session_id,
                    agent_id,
                    truncate(prompt_text, 160),
                )
                # Leader 不再自动构造补充，直接返回由前端收集用户输入再继续
                break

            # 有产出，等待确认
            if state == TaskState.AwaitingCompletion:
                product_text = (
                    _extract_text_from_data_items(task.products[0].dataItems)
                    if task.products
                    else ""
                )
                satisfied, feedback = _evaluate_product_satisfaction(
                    sub_query or user_query, product_text
                )
                if satisfied:
                    task = await client.complete_task(
                        task_id=task.id, session_id=session_id
                    )
                    break
                else:
                    # Leader 不再改写反馈内容，直接使用用户输入进行下一次 Continue
                    task = await client.continue_task(
                        task_id=task.id,
                        session_id=session_id,
                        user_input=(feedback or "请继续完善上述方案。"),
                    )
                    loops += 1
                    if loops >= max_loops:
                        break
                    await asyncio.sleep(1)
                    task = await client.get_task(task.id, session_id)
                    continue

            # 仍在处理
            loops += 1
            if loops >= max_loops:
                break
            await asyncio.sleep(1)
            task = await client.get_task(task.id, session_id)

        # 汇总返回
        product_text = None
        if task.products and task.products[0].dataItems:
            product_text = _extract_text_from_data_items(task.products[0].dataItems)
        if (not product_text) and task.status.state == TaskState.AwaitingInput:
            product_text = _extract_text_from_data_items(task.status.dataItems)

        result = {
            "agent_id": agent_id,
            "success": task.status.state not in (TaskState.Failed, TaskState.Rejected),
            "state": task.status.state,
            "product_text": product_text,
            "raw_task": task.model_dump(),
            "needs_user_input": task.status.state == TaskState.AwaitingInput,
        }
        logger.info(
            "event=partner_call_end session_id=%s agent_id=%s state=%s success=%s product_chars=%s",
            session_id,
            agent_id,
            task.status.state,
            result["success"],
            len(product_text) if product_text else 0,
        )
        return result
    except Exception as e:
        logger.exception(
            "event=partner_call_exception session_id=%s agent_id=%s",
            session_id,
            agent_id,
        )
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "error",
            "product_text": None,
            "error": str(e),
        }
    finally:
        await client.close()


async def continue_partner(
        agent_id: str,
        task_id: str,
        user_input: str,
        session_id: str,
        *,
        sub_query: Optional[str] = None,
        awaiting_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """继续已有任务：发送 Continue → 轮询 → 根据 AwaitingCompletion 进行评估或继续。

    返回结构同 call_partner。
    """

    # 把内部id映射为查找文件所要的key
    look_up_key = None
    if agent_id == "racket_assess_partner_001":
        look_up_key = "racket_assess"
    elif agent_id == "user_assess_partner_001":
        look_up_key = "user_assess"

    if not look_up_key:
        return {"agent_id": agent_id,
                "success": False,
                "state": "unknown_agent",
                "product_text": None,
                "raw_task": None,
                "error": "unsupported_agent",
                }

    agent_info = find_agent_service(look_up_key)
    partner_url = extract_jsonrpc_endpoint(agent_info) if agent_info else None
    if not partner_url:
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unavailable",
            "product_text": None,
            "error": f"{agent_id} JSONRPC endpoint not available",
        }
    client = AipRpcClient(
        partner_url=partner_url, leader_id=LEADER_ID, ssl_context=_client_ssl_context
    )
    try:
        # Continue → 轮询 → 分支
        task = await client.continue_task(
            task_id=task_id,
            session_id=session_id,
            user_input=_build_continue_payload(
                user_input=user_input,
                sub_query=sub_query,
                awaiting_prompt=awaiting_prompt,
            ),
        )
        # 与 call_partner 相同的轮询/评估逻辑
        max_loops = 60
        loops = 0
        last_state = None
        while True:
            state = task.status.state
            if state != last_state:
                last_state = state
                logger.debug(
                    "event=partner_state_change session_id=%s agent_id=%s task_id=%s state=%s",
                    session_id,
                    agent_id,
                    task.id,
                    state,
                )
            _save_partner_task_state(
                session_id,
                agent_id,
                task.id,
                state,
                sub_query=sub_query,
                last_product=(
                    _extract_text_from_data_items(task.products[0].dataItems)
                    if task.products and task.products[0].dataItems
                    else None
                ),
                awaiting_prompt=_extract_text_from_data_items(task.status.dataItems),
            )
            if state in (
                    TaskState.Completed,
                    TaskState.Canceled,
                    TaskState.Failed,
                    TaskState.Rejected,
            ):
                break
            if state == TaskState.AwaitingInput:
                break  # 继续等待用户
            if state == TaskState.AwaitingCompletion:
                product_text = (
                    _extract_text_from_data_items(task.products[0].dataItems)
                    if task.products
                    else ""
                )
                satisfied, feedback = _evaluate_product_satisfaction(
                    sub_query or user_input, product_text
                )
                if satisfied:
                    task = await client.complete_task(
                        task_id=task.id, session_id=session_id
                    )
                    break
                else:
                    task = await client.continue_task(
                        task_id=task.id,
                        session_id=session_id,
                        user_input=_build_continue_payload(
                            user_input=(feedback or "请继续完善上述方案。"),
                            sub_query=sub_query,
                            awaiting_prompt=awaiting_prompt,
                        ),
                    )
                    loops += 1
                    if loops >= max_loops:
                        break
                    await asyncio.sleep(1)
                    task = await client.get_task(task.id, session_id)
                    continue
            loops += 1
            if loops >= max_loops:
                break
            await asyncio.sleep(1)
            task = await client.get_task(task.id, session_id)
        product_text = None
        if task.products and task.products[0].dataItems:
            product_text = _extract_text_from_data_items(task.products[0].dataItems)
        if (not product_text) and task.status.state == TaskState.AwaitingInput:
            product_text = _extract_text_from_data_items(task.status.dataItems)
        return {
            "agent_id": agent_id,
            "success": task.status.state not in (TaskState.Failed, TaskState.Rejected),
            "state": task.status.state,
            "product_text": product_text,
            "raw_task": task.model_dump(),
            "needs_user_input": task.status.state == TaskState.AwaitingInput,
        }
    finally:
        await client.close()


async def complete_partner(
        agent_id: str,
        task_id: str,
        session_id: str,
        *,
        sub_query: Optional[str] = None,
) -> Dict[str, Any]:
    """尝试直接完成处于 AwaitingCompletion 的任务（用户明确表示已满足/请完成）。"""

    # 把内部id映射为查找文件所要的key
    look_up_key = None
    if agent_id == "racket_assess_partner_001":
        look_up_key = "racket_assess"
    elif agent_id == "user_assess_partner_001":
        look_up_key = "user_assess"

    else:
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unknown-agent",
            "product_text": None,
            "error": "unsupported agent id",
        }
    agent_info = find_agent_service(look_up_key)
    partner_url = extract_jsonrpc_endpoint(agent_info) if agent_info else None
    if not partner_url:
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unavailable",
            "product_text": None,
            "error": f"{agent_id} JSONRPC endpoint not available",
        }
    client = AipRpcClient(
        partner_url=partner_url, leader_id=LEADER_ID, ssl_context=_client_ssl_context
    )
    try:
        task = await client.complete_task(task_id=task_id, session_id=session_id)
        # 简短轮询一次，获取最终产品文本
        await asyncio.sleep(0.5)
        task = await client.get_task(task.id, session_id)
        _save_partner_task_state(
            session_id,
            agent_id,
            task.id,
            task.status.state,
            sub_query=sub_query,
            last_product=(
                _extract_text_from_data_items(task.products[0].dataItems)
                if task.products and task.products[0].dataItems
                else None
            ),
            awaiting_prompt=_extract_text_from_data_items(task.status.dataItems),
        )
        product_text = None
        if task.products and task.products[0].dataItems:
            product_text = _extract_text_from_data_items(task.products[0].dataItems)
        return {
            "agent_id": agent_id,
            "success": task.status.state not in (TaskState.Failed, TaskState.Rejected),
            "state": task.status.state,
            "product_text": product_text,
            "raw_task": task.model_dump(),
            "needs_user_input": task.status.state == TaskState.AwaitingInput,
        }
    finally:
        await client.close()


def _evaluate_product_satisfaction(
        requirement_text: str, product_text: str
) -> Tuple[bool, Optional[str]]:
    """使用 LLM 对产出物进行轻量评估。返回 (是否满意, 如不满足给出反馈指令)。"""
    try:
        prompt = (
            "请判断给定的‘产出物’是否覆盖并满足‘需求’。只返回两行：\n"
            "第一行：YES 或 NO；\n"
            "第二行：若 NO，请用一句中文给出需要补充或修正的具体指令；若 YES 留空。\n\n"
            f"需求：\n{requirement_text}\n\n产出物：\n{product_text}"
        )
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        text = resp.choices[0].message.content.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        ok = lines and lines[0].upper().startswith("YES")
        feedback = None
        if not ok and len(lines) >= 2:
            feedback = lines[1]
        return ok, feedback
    except Exception:
        # 回退策略：无法判断则默认满意，避免阻塞
        return True, None


def llm_integrate(
        query: str,
        analysis: Dict[str, Any],
        partner_results: Dict[str, Dict[str, Any]],
        session_id: str,
) -> str:
    """阶段3：整合结果生成最终答复（支持多代理）。"""
    context = build_context(session_id)
    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)
    # 构造 partner results 文本块
    blocks = []

    # "racket_assess_partner": "球拍性能评价智能体",
    # "user_assess_partner": "适合用户评价智能体",
    # "racket_recommender_partner": "相似推荐智能体",


    for aid, res in partner_results.items():
        if aid == "racket_assess_partner_001":
            label = "球拍性能评价智能体"
        elif aid == "user_assess_partner_001":
            label = "适合用户评价智能体"

        else:
            label = aid
        text_part = res.get("product_text") or res.get("error") or "(无产出)"
        blocks.append(f"[{label} {aid}]\n{text_part}\n")
    partner_results_block = "\n".join(blocks) if blocks else "(无代理产出)"
    decomposition_block = json.dumps(
        analysis.get("decomposition", {}), ensure_ascii=False, indent=2
    )
    prompt = _fill_template(
        INTEGRATION_PROMPT_TEMPLATE,
        context=context,
        query=query,
        analysis_json=analysis_json,
        partner_results_block=partner_results_block,
        decomposition_block=decomposition_block,
    )
    logger.info(
        "event=integration_start session_id=%s query_chars=%d partner_count=%d",
        session_id,
        len(query),
        len(partner_results),
    )
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1600,
        )
        text = resp.choices[0].message.content.strip()
        logger.info(
            "event=integration_llm_end session_id=%s chars=%d preview=%s",
            session_id,
            len(text),
            truncate(text.replace("\n", " "), 160),
        )
        return text
    except Exception as e:
        logger.exception("event=integration_exception session_id=%s", session_id)
        return f"整合阶段发生错误: {str(e)}\n\n原始多代理结果:\n{partner_results_block}"


@app.post("/user_api")
async def handle_user_request(user_request: UserRequest):
    """统一入口：多阶段执行流程

    阶段流程：
    1) 需求分析（LLM）
    2) 调用 Partner（当前仅北京城区景点规划师）
    3) 结果整合（LLM）
    4) 返回最终方案
    """
    session_id = user_request.session_id or f"session-{uuid.uuid4()}"
    is_new_session = user_request.session_id is None
    user_query = user_request.query
    logger.info(
        "event=request_received session_id=%s new_session=%s query_chars=%d",
        session_id,
        is_new_session,
        len(user_query),
    )

    # 记录用户消息
    append_message(session_id, "user", user_query)

    sess = get_session(session_id)
    partner_results: Dict[str, Dict[str, Any]] = {}
    partner_subqueries: Dict[str, str] = {}

    t0 = datetime.now(timezone.utc)
    analysis = llm_analysis(user_query, session_id)
    t1 = datetime.now(timezone.utc)
    dimensions = analysis.get("dimensions") or {}
    required_agents = analysis.get("required_agents") or []
    if isinstance(required_agents, str):
        required_agents = [required_agents]

    # 判断：这次输入是否用于补充进行中的任务（等待用户/完成确认）
    pending = [
        (aid, info)
        for aid, info in sess.get("partner_tasks", {}).items()
        if info.get("state") in (TaskState.AwaitingInput, TaskState.AwaitingCompletion)
    ]
    is_supplement = len(pending) > 0 and not is_new_session

    # "racket_assess_partner": "球拍性能评价智能体",
    # "user_assess_partner": "适合用户评价智能体",
    # "racket_recommender_partner": "相似推荐智能体",

    agent_dim_map = {
        "racket_assess_partner_001": "racket_assess",
        "user_assess_partner_001": "user_assess",

    }

    def resolve_sub_query(agent_id: str) -> str:
        dim_key = agent_dim_map.get(agent_id)
        if dim_key and dim_key in dimensions:
            sub_obj = dimensions.get(dim_key) or {}
            candidate = sub_obj.get("sub_query")
            if candidate:
                return candidate
        return (
                sess.get("partner_tasks", {}).get(agent_id, {}).get("sub_query")
                or user_query
        )

    if is_supplement:
        logger.info(
            "event=input_classified supplement=true session_id=%s waiting_agents=%s",
            session_id,
            ",".join([aid for aid, _ in pending]),
        )
        sub_payloads = []
        for aid, info in pending:
            sub_q = resolve_sub_query(aid)
            partner_subqueries[aid] = sub_q
            if aid in sess.get("partner_tasks", {}):
                sess["partner_tasks"][aid]["sub_query"] = sub_q
            wants_complete = ("完成" in user_query) or ("整理结果" in user_query)
            if wants_complete and info.get("state") == TaskState.AwaitingCompletion:
                sub_payloads.append(
                    complete_partner(
                        aid,
                        info["task_id"],
                        session_id,
                        sub_query=sub_q,
                    )
                )
            else:
                sub_payloads.append(
                    continue_partner(
                        aid,
                        info["task_id"],
                        user_query,
                        session_id,
                        sub_query=sub_q,
                        awaiting_prompt=info.get("awaiting_prompt"),
                    )
                )
        results = await asyncio.gather(*sub_payloads)
        for r in results:
            partner_results[r["agent_id"]] = r
            logger.info(
                "event=supplement_continue_result session_id=%s agent_id=%s state=%s success=%s",
                session_id,
                r["agent_id"],
                r.get("state"),
                r.get("success"),
            )

    # 根据最新分析结果，决定是否需要启动新的 Partner 任务
    supported_ids = set(agent_dim_map.keys())
    active_states = {
        TaskState.Accepted,
        TaskState.Working,
        TaskState.AwaitingInput,
        TaskState.AwaitingCompletion,
    }
    call_list: list[str] = []
    for aid in required_agents:
        if aid not in supported_ids:
            continue
        current_info = sess.get("partner_tasks", {}).get(aid)
        if current_info and current_info.get("state") in active_states:
            # 仍在执行/等待中，沿用当前任务
            partner_subqueries.setdefault(aid, resolve_sub_query(aid))
            continue
        call_list.append(aid)
        partner_subqueries[aid] = resolve_sub_query(aid)

    if call_list:
        sub_payloads = [
            call_partner(
                aid,
                partner_subqueries[aid],
                session_id,
                sub_query=partner_subqueries[aid],
            )
            for aid in call_list
        ]
        logger.info(
            "event=partner_batch_start session_id=%s agents=%s",
            session_id,
            ",".join(call_list),
        )
        results = await asyncio.gather(*sub_payloads)
        logger.info(
            "event=partner_batch_end session_id=%s agents=%s",
            session_id,
            ",".join(call_list),
        )
        for r in results:
            partner_results[r["agent_id"]] = r

    # 阶段 3：结果整合
    t2 = datetime.now(timezone.utc)
    final_text = llm_integrate(user_query, analysis, partner_results, session_id)
    t3 = datetime.now(timezone.utc)


    logger.info(
        "event=request_complete session_id=%s analysis_ms=%d partner_ms=%d integration_ms=%d total_ms=%d",
        session_id,
        int((t1 - t0).total_seconds() * 1000),
        int((t2 - t1).total_seconds() * 1000),
        int((t3 - t2).total_seconds() * 1000),
        int((t3 - t0).total_seconds() * 1000),
    )

    # 记录助理输出
    append_message(session_id, "assistant", final_text)

    return {
        "session_id": session_id,
        "analysis": analysis,
        "partner_results": partner_results,
        "partner_subqueries": partner_subqueries,
        "final_response": final_text,
        # 额外返回：前端可据此展示和区分
        "pending_questions": get_session(session_id).get("pending_questions", []),
        "partner_tasks": get_session(session_id).get("partner_tasks", {}),
    }


@app.get("/")
def read_root():
    """健康检查 / 简易欢迎信息"""
    return {"message": f"欢迎使用旅游助理 {LEADER_ID}. 调用 /user_api 进行交互"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("racket_leader:app", host="0.0.0.0", port=8019, reload=True)



















import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL = os.getenv("MODEL_NAME")

# 模拟数据库：老人护理需求 & 护工信息
mock_elderly_needs = [
    {
        "id": "E001",
        "name": "张奶奶",
        "address": "成都市成华区XX小区1栋",
        "care_level": "中度护理",
        "needs": ["日常照护", "基础用药提醒", "血压测量", "心理陪伴"],
        "preferred_time": "上午9:00-11:00",
        "urgency": "常规"
    },
    {
        "id": "E002",
        "name": "李爷爷",
        "address": "成都市成华区XX小区3栋",
        "care_level": "重度护理",
        "needs": ["压疮护理", "导尿管护理", "康复辅助训练"],
        "preferred_time": "下午14:00-16:00",
        "urgency": "紧急"
    }
]

mock_caregivers = [
    {
        "id": "C001",
        "name": "王护工",
        "skills": ["日常照护", "基础用药提醒", "血压测量", "心理陪伴"],
        "location": "成都市成华区XX小区附近",
        "distance_km": 0.8,
        "available_time": ["上午9:00-12:00", "下午14:00-18:00"],
        "compliance_status": "合规（持证上岗、无不良记录）",
        "rating": 4.8
    },
    {
        "id": "C002",
        "name": "刘护工",
        "skills": ["压疮护理", "导尿管护理", "康复辅助训练", "日常照护"],
        "location": "成都市成华区XX小区附近",
        "distance_km": 1.2,
        "available_time": ["上午8:00-11:00", "下午13:00-17:00"],
        "compliance_status": "合规（高级护理资质、急救认证）",
        "rating": 4.9
    },
    {
        "id": "C003",
        "name": "张护工",
        "skills": ["日常照护", "基础用药提醒"],
        "location": "成都市成华区XX小区附近",
        "distance_km": 2.5,
        "available_time": ["上午9:00-12:00", "下午15:00-18:00"],
        "compliance_status": "合规",
        "rating": 4.5
    }
]

# ------------------------------
# Agent 1: 需求解析Agent
# ------------------------------
def demand_parse_agent(elderly_need):
    """解析老人护理需求，提取核心维度"""
    prompt = f"""
    你是专业的养老服务需求分析师，请解析以下老人护理需求，输出结构化JSON，包含：
    - core_needs: 核心护理需求列表（提炼关键项）
    - care_level: 护理等级（已给出，直接保留）
    - preferred_time: 服务时间窗口（已给出，直接保留）
    - urgency: 服务紧急程度（已给出，直接保留）
    - special_requirements: 潜在特殊要求（如需要持证护工、急救能力等）

    老人需求信息：
    {json.dumps(elderly_need, ensure_ascii=False, indent=2)}

    仅返回JSON，不要其他内容。
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

# ------------------------------
# Agent 2: 护工匹配Agent
# ------------------------------
def caregiver_match_agent(parsed_demand, caregivers):
    """根据解析后的需求，筛选符合条件的护工"""
    prompt = f"""
    你是护工匹配专家，请根据以下老人护理需求，从护工列表中筛选出符合条件的护工，输出JSON列表，包含：
    - caregiver_id: 护工ID
    - match_score: 匹配度评分（0-100）
    - match_reason: 匹配理由（技能匹配、时间匹配、距离优势等）
    - risk_warning: 潜在风险提示（如技能不满足、时间冲突等，无风险填"无"）

    老人需求解析结果：
    {json.dumps(parsed_demand, ensure_ascii=False, indent=2)}

    护工信息列表：
    {json.dumps(caregivers, ensure_ascii=False, indent=2)}

    仅返回JSON列表，不要其他内容。
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

# ------------------------------
# Agent 3: 方案优化Agent（长链推理）
# ------------------------------
def dispatch_optimize_agent(parsed_demand, matched_caregivers, elderly_info):
    """基于匹配结果，进行长链推理，生成最优派单方案"""
    prompt = f"""
    你是养老服务派单优化专家，请结合以下信息，生成最优派单方案，输出JSON，包含：
    - recommended_caregiver_id: 推荐护工ID
    - recommended_caregiver_name: 推荐护工姓名
    - service_schedule: 具体服务时间安排
    - service_plan_details: 详细服务流程（按时间拆解）
    - compliance_check: 合规性校验结果（如资质、技能是否符合护理等级要求）
    - fallback_plan: 备选方案（主选护工无法服务时的备选）

    要求：
    1. 优先满足紧急需求、护理等级匹配度
    2. 兼顾护工距离、排班、技能覆盖度
    3. 考虑服务连续性，避免护工时间冲突
    4. 确保所有方案符合养老护理行业合规要求

    老人信息：
    {json.dumps(elderly_info, ensure_ascii=False, indent=2)}

    解析后的护理需求：
    {json.dumps(parsed_demand, ensure_ascii=False, indent=2)}

    匹配的护工列表：
    {json.dumps(matched_caregivers, ensure_ascii=False, indent=2)}

    仅返回JSON，不要其他内容。
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

# ------------------------------
# Agent 4: 结果验证Agent
# ------------------------------
def dispatch_verify_agent(dispatch_plan, elderly_info, caregiver_info):
    """验证派单方案的合理性与合规性，输出验证报告"""
    prompt = f"""
    你是养老服务派单方案审核员，请对以下派单方案进行验证，输出JSON，包含：
    - plan_valid: 方案是否有效（true/false）
    - compliance_score: 合规性评分（0-100）
    - validation_details: 验证通过项列表
    - improvement_suggestions: 优化建议（无建议填"无"）
    - final_conclusion: 最终结论（如"方案合规，可执行"）

    老人信息：
    {json.dumps(elderly_info, ensure_ascii=False, indent=2)}

    护工信息：
    {json.dumps(caregiver_info, ensure_ascii=False, indent=2)}

    派单方案：
    {json.dumps(dispatch_plan, ensure_ascii=False, indent=2)}

    仅返回JSON，不要其他内容。
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

# ------------------------------
# 主流程：多Agent协作闭环
# ------------------------------
def run_dispatch_pipeline(elderly_id):
    """执行完整的派单Agent流程"""
    # 1. 获取老人信息
    elderly_info = next(e for e in mock_elderly_needs if e["id"] == elderly_id)
    print(f"=== 开始处理老人 {elderly_info['name']} 的派单请求 ===")

    # 2. Agent1：需求解析
    print("\n[1/4] 需求解析Agent运行中...")
    parsed_demand = demand_parse_agent(elderly_info)
    print("✅ 需求解析完成")

    # 3. Agent2：护工匹配
    print("\n[2/4] 护工匹配Agent运行中...")
    matched_caregivers = caregiver_match_agent(parsed_demand, mock_caregivers)
    print("✅ 护工匹配完成")

    # 4. Agent3：方案优化（长链推理）
    print("\n[3/4] 派单方案优化Agent运行中...")
    dispatch_plan = dispatch_optimize_agent(parsed_demand, matched_caregivers, elderly_info)
    print("✅ 派单方案生成完成")

    # 5. Agent4：结果验证
    print("\n[4/4] 派单方案验证Agent运行中...")
    # 获取推荐护工的完整信息
    recommended_caregiver = next(
        c for c in mock_caregivers if c["id"] == dispatch_plan["recommended_caregiver_id"]
    )
    verify_result = dispatch_verify_agent(dispatch_plan, elderly_info, recommended_caregiver)
    print("✅ 方案验证完成")

    # 输出最终结果
    print("\n================ 最终派单结果 ================")
    print(f"老人：{elderly_info['name']}")
    print(f"推荐护工：{dispatch_plan['recommended_caregiver_name']}（ID：{dispatch_plan['recommended_caregiver_id']}）")
    print(f"服务安排：{dispatch_plan['service_schedule']}")
    print(f"方案有效性：{'有效' if verify_result['plan_valid'] else '无效'}")
    print(f"合规性评分：{verify_result['compliance_score']}/100")
    print(f"最终结论：{verify_result['final_conclusion']}")

    return {
        "elderly_info": elderly_info,
        "parsed_demand": parsed_demand,
        "matched_caregivers": matched_caregivers,
        "dispatch_plan": dispatch_plan,
        "verify_result": verify_result
    }

# 运行Demo
if __name__ == "__main__":
    # 测试：处理张奶奶（E001）和李爷爷（E002）的派单请求
    run_dispatch_pipeline("E001")
    print("\n" + "="*60 + "\n")
    run_dispatch_pipeline("E002")

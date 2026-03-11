import json
import requests
import re

# ================= 配置区 =================
# Chatchat 原生知识库问答接口
RAG_API_URL = "http://127.0.0.1:7861/chat/knowledge_base_chat"

MODEL_NAME = "qwen1.5-7b-chat"  # 替换为你实际运行的模型名
KNOWLEDGE_BASE_NAME = "JAVA_GUIDE"    # 替换为你测试用的知识库名称
DATASET_PATH = "evaluation_dataset.jsonl"
# ==========================================

def query_rag_system(question):
    """向你的魔改版 RAG 系统提问，获取生成的答案"""
    payload = {
        "query": question,
        "knowledge_base_name": KNOWLEDGE_BASE_NAME,
        "top_k": 3,
        "score_threshold": 1.0,
        "model_name": MODEL_NAME,
        "temperature": 0.1
    }
    try:
        response = requests.post(RAG_API_URL, json=payload, stream=True)
        # Chatchat 的流式返回处理
        full_answer = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    try:
                        data_json = json.loads(data_str)
                        if "answer" in data_json:
                            full_answer += data_json["answer"]
                    except:
                        pass
        return full_answer.strip()
    except Exception as e:
        return f"RAG请求失败: {e}"

def llm_judge(question, ground_truth, rag_answer):
    """多维度 LLM-as-a-Judge 裁判 (基于 Chatchat 原生 /chat/chat 接口)"""
    prompt = f"""
    你是一个严苛的 AI 评估专家。请比较【RAG系统生成的答案】和【人类标准答案】。
    请从以下三个维度进行打分（每个维度 0-5 分，5分为完美）：
    
    1. Faithfulness (事实忠实度): RAG答案是否出现了标准答案中不存在的幻觉或错误？
    2. Answer Relevance (回答相关性): RAG答案是否直接回答了用户的问题，没有答非所问？
    3. Correctness (准确率): RAG答案的核心语义和结论是否与标准答案一致？

    【用户问题】: {question}
    【人类标准答案】: {ground_truth}
    【RAG系统生成的答案】: {rag_answer}

    请严格输出合法的 JSON 格式，包含三个维度的分数和一段简短的评价，不要包含其他字符：
    {{
        "faithfulness": 5,
        "relevance": 5,
        "correctness": 5,
        "reason": "评价理由"
    }}
    """
    
    # 使用与 RAG 相同的原生 payload 格式
    payload = {
        "query": prompt,
        "knowledge_base_name": "", 
        "model_name": MODEL_NAME,
        "temperature": 0.1,
        "stream": True
    }
    
    try:
        # 强制走 7861 原生对话接口
        response = requests.post("http://127.0.0.1:7861/chat/chat", json=payload, stream=True)
        result_text = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    if data_str.startswith('[DONE]'):
                        continue
                    try:
                        data_json = json.loads(data_str)
                        # 注意：原生 chat 接口返回的字段是 "text"
                        if "text" in data_json:
                            result_text += data_json["text"]
                    except:
                        pass
                        
        # 用正则表达式强行把 JSON 抠出来（防止大模型在 JSON 外面乱加废话）
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"faithfulness": 0, "relevance": 0, "correctness": 0, "reason": f"裁判未输出JSON: {result_text[:30]}..."}
    except Exception as e:
        return {"faithfulness": 0, "relevance": 0, "correctness": 0, "reason": f"裁判调用彻底失败: {e}"}

def run_evaluation():
    print("🚀 开始全自动 RAG 多维度评测...\n")
    results = []
    total_scores = {"faithfulness": 0, "relevance": 0, "correctness": 0}
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        data = json.loads(line)
        question = data['question']
        ground_truth = data['ground_truth']
        category = data['category']
        
        print(f"[{i+1}/{len(lines)}] 测试维度: {category} | 问题: {question[:20]}...")
        
        # 1. 向 RAG 获取答案
        rag_answer = query_rag_system(question)
        
        # 2. 裁判打分
        evaluation = llm_judge(question, ground_truth, rag_answer)
        
        # 3. 记录成绩
        total_scores["faithfulness"] += evaluation.get("faithfulness", 0)
        total_scores["relevance"] += evaluation.get("relevance", 0)
        total_scores["correctness"] += evaluation.get("correctness", 0)
        
        print(f"   => 得分: 忠实度 {evaluation.get('faithfulness')}/5 | 相关性 {evaluation.get('relevance')}/5 | 准确率 {evaluation.get('correctness')}/5")
        print(f"   => 裁判评语: {evaluation.get('reason')}\n")

    # 打印最终报告
    num_tests = len(lines)
    print("==================================================")
    print("📊 最终评测雷达图 (均分满分 5.0):")
    print(f"🔹 事实忠实度 (Faithfulness)  : {total_scores['faithfulness'] / num_tests:.2f}")
    print(f"🔹 回答相关性 (Relevance)     : {total_scores['relevance'] / num_tests:.2f}")
    print(f"🔹 核心准确率 (Correctness)   : {total_scores['correctness'] / num_tests:.2f}")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation()
# 股票分析数据流程详解 📊

## 1. 完整分析流程

### 流程概览:
```
用户提交表单 (股票代码、市场、分析师、研究深度、LLM提供商)
        ↓
    web/app.py → run_stock_analysis()
        ↓
    web/utils/analysis_runner.py (行号100)
    - 验证股票代码
    - 格式化代码 (A股/港股/美股)
    - 根据研究深度配置LLM模型
        ↓
    TradingAgentsGraph.__init__() (trading_graph.py:437)
    - 初始化LLM适配器
    - 选择 Gemini/DashScope/DeepSeek/OpenAI
    - 创建工具包和内存系统
        ↓
    graph.propagate() (行号446)
    - 创建初始state
    - 启动分析师并行执行
        ↓
    【并行阶段】分析师执行
    ├─ 市场分析师 (market_analyst.py:491) → llm.invoke()
    ├─ 基本面分析师 (fundamentals_analyst.py) → llm.invoke()
    ├─ 新闻分析师 (news_analyst.py:222,313) → llm.invoke()
    └─ 情绪分析师 (social_media_analyst.py) → llm.invoke()
        ↓
    【辩论阶段】
    ├─ 多头研究员 (bull_researcher.py) → llm.invoke()
    ├─ 空头研究员 (bear_researcher.py) → llm.invoke()
    ├─ 研究经理 (research_manager.py:69) → llm.invoke()
        ↓
    【交易决策】
    └─ 交易员 (trader.py:103) → llm.invoke()
        返回投资建议
        ↓
    【风险评估】
    ├─ 激进分析师 (aggressive_debator.py) → llm.invoke()
    ├─ 保守分析师 (conservative_debator.py) → llm.invoke()
    ├─ 中立分析师 (neutral_debator.py) → llm.invoke()
    └─ 风险经理 (risk_manager.py) → llm.invoke()
        ↓
    【信号处理】
    └─ SignalProcessor (signal_processing.py:98) → llm.invoke()
        提取JSON格式决策
        ↓
    返回最终结果到前端
```

---

## 2. 核心数据结构

### State (AgentState) 结构

初始化 (propagation.py:22-46):
```python
state = {
    "messages": [("human", "AAPL")],      # 消息历史（LLM交互记录）
    "company_of_interest": "AAPL",         # 分析的股票代码
    "trade_date": "2025-10-23",            # 分析日期
    
    # 各分析师的输出报告
    "market_report": "",                   # 市场/技术分析结果
    "fundamentals_report": "",             # 基本面分析结果
    "sentiment_report": "",                # 情绪分析结果
    "news_report": "",                     # 新闻分析结果
    
    # 投资辩论状态
    "investment_debate_state": {
        "history": "",                     # 多空辩论历史
        "current_response": "",            # 当前发言
        "count": 0                         # 轮次计数
    },
    
    # 风险评估状态
    "risk_debate_state": {
        "history": "",                     # 风险讨论历史
        "current_risky_response": "",      # 激进观点
        "current_safe_response": "",       # 保守观点
        "current_neutral_response": "",    # 中立观点
        "count": 0
    }
}
```

---

## 3. 分析师执行模式 (所有分析师遵循相同模式)

### 市场分析师 示例 (market_analyst.py)

```python
# 第一步: 构建系统提示词
system_message = f"""你是一位专业的股票技术分析师。
股票代码：{ticker}
公司名称：{company_name}
市场：{market_info['market_name']}
货币：{market_info['currency_name']}

你必须调用工具 get_stock_market_data_unified 获取数据
然后基于数据进行技术分析。"""

# 第二步: 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="messages")
])

# 第三步: 调用LLM (行号491)
result = llm.invoke(messages)

# 第四步: 检查工具调用
if hasattr(result, 'tool_calls') and result.tool_calls:
    # 执行工具获取真实数据
    tool_messages = execute_tools(result.tool_calls)
    
# 第五步: 基于工具结果再次调用LLM
messages = state["messages"] + [result] + tool_messages
final_result = llm.invoke(messages)

# 第六步: 更新状态
state["market_report"] = final_result.content
state["messages"].append(final_result)
return {
    "messages": [result] + tool_messages + [final_result],
    "market_report": final_result.content
}
```

### 所有分析师的调用位置:

| 分析师 | 文件 | 行号 | LLM调用行 |
|-------|------|------|---------|
| 市场分析师 | market_analyst.py | 346 | 491 |
| 基本面分析师 | fundamentals_analyst.py | 84 | 220+ |
| 新闻分析师 | news_analyst.py | 19 | 222, 313 |
| 情绪分析师 | social_media_analyst.py | 78 | 构建后调用 |
| 多头研究员 | bull_researcher.py | 15 | 60+ |
| 空头研究员 | bear_researcher.py | 15 | 构建后调用 |
| 研究经理 | research_manager.py | 9 | 69 |

---

## 4. LLM提供商与模型配置

### 配置来源: analysis_runner.py (行号224-344)

```python
# 根据LLM提供商设置
llm_provider = form_config.get('llm_provider', '阿里百炼默认')

if llm_provider == "dashscope":      # 阿里百炼
    quick_think_llm = "qwen-turbo"   # 最快模型
    deep_think_llm = "qwen-plus/qwen3-max"
    backend_url = "https://dashscope.aliyuncs.com/api/v1"

elif llm_provider == "deepseek":     # DeepSeek
    quick_think_llm = "deepseek-chat"
    deep_think_llm = "deepseek-chat"
    backend_url = "https://api.deepseek.com"

elif llm_provider == "openai":       # OpenAI
    quick_think_llm = "gpt-4o-mini"
    deep_think_llm = "gpt-4o"
    backend_url = "https://api.openai.com/v1"

elif llm_provider == "google":       # Google Gemini
    quick_think_llm = "gemini-2.5-flash"
    deep_think_llm = "gemini-2.5-pro"   # 最高质量
    backend_url = "https://generativelanguage.googleapis.com"
```

### 研究深度映射 (analysis_runner.py:228-293)

| 深度 | 辩论 | 风险 | 快速模型 | 深度模型 | 说明 |
|-----|------|------|--------|--------|------|
| 1级 | 1轮 | 1轮 | turbo | plus | 快速分析，用时1-2分钟 |
| 2级 | 1轮 | 1轮 | plus | plus | 基础分析，2-3分钟 |
| 3级 | 1轮 | 2轮 | plus | 3-max | 标准分析，3-5分钟 |
| 4级 | 2轮 | 2轮 | plus | 3-max | 深度分析，5-8分钟 |

---

## 5. LLM初始化 (trading_graph.py:68-143)

```python
provider = self.config["llm_provider"].lower()

if provider == "openai":
    self.deep_thinking_llm = ChatOpenAI(
        model=config["deep_think_llm"],
        base_url=config["backend_url"],
        api_key=os.getenv('OPENAI_API_KEY')
    )

elif provider == "google":
    self.deep_thinking_llm = ChatGoogleOpenAI(
        model=config["deep_think_llm"],
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.1
    )

elif provider == "dashscope":
    self.deep_thinking_llm = ChatDashScopeOpenAI(
        model=config["deep_think_llm"],
        temperature=0.1,
        max_tokens=2000
    )

elif provider == "deepseek":
    self.deep_thinking_llm = ChatDeepSeek(
        model=config["deep_think_llm"],
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL'),
        temperature=0.1
    )
```

---

## 6. 条件逻辑 (conditional_logic.py)

### 工具调用流程检查:

```python
def should_continue_market(state: AgentState):
    """检查是否需要执行市场数据工具"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果LLM最后一条消息包含工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools_market"  # 执行工具
    else:
        return "Msg Clear Market"  # 跳过工具，清除消息

# 同样的逻辑应用于:
# - should_continue_social() → "tools_social"
# - should_continue_news() → "tools_news"
# - should_continue_fundamentals() → "tools_fundamentals"
```

### 辩论继续条件:

```python
def should_continue_debate(state: AgentState) -> str:
    """检查是否继续多空辩论"""
    count = state["investment_debate_state"]["count"]
    max_rounds = config["max_debate_rounds"]
    
    # 达到最大轮次？
    if count >= 2 * max_rounds:  # 3 turns of back-and-forth
        return "Research Manager"  # 由研究经理总结
    
    # 根据当前发言者切换
    if state["investment_debate_state"]["current_response"].startswith("Bull"):
        return "Bear Researcher"  # 切换到空头
    else:
        return "Bull Researcher"  # 切换到多头
```

---

## 7. 最终决策生成

### 交易员 (trader.py:10-116)

```python
# 行号103: LLM调用
result = llm.invoke(messages)

# 消息包含:
messages = [
    {
        "role": "system",
        "content": """你是投资交易决策专家...
        
强制要求:
1. 目标价位: 必须给出具体数值 ($XXX 或 ¥XXX)
2. 置信度: 0-1之间的数值
3. 风险评分: 0-1之间的数值
4. 决策理由: 详细的投资逻辑

所有分析结果:
- 技术面: {market_report}
- 基本面: {fundamentals_report}
- 情绪面: {sentiment_report}
- 新闻面: {news_report}
- 辩论结论: {research_conclusion}
"""
    },
    {"role": "user", "content": "基于以上所有分析，给出最终投资建议"}
]

# LLM返回:
result = {
    "investment_decision": "买入/持有/卖出",
    "target_price": "$150.50",
    "confidence": 0.85,
    "reasoning": "详细的投资逻辑"
}
```

### 信号处理 (signal_processing.py:66-95)

```python
# 行号98: 让LLM提取JSON格式
messages = [
    ("system", f"""从投资报告中提取结构化JSON:
    
    {{
        "action": "买入/持有/卖出",
        "target_price": 数字(必须具体数值),
        "confidence": 0-1之间,
        "risk_score": 0-1之间,
        "reasoning": "摘要"
    }}
    """),
    ("human", full_signal)  # 交易员的完整输出
]

# LLM提取并返回JSON
```

---

## 8. 数据流示意

### 从用户输入到最终输出:

```
web/app.py (render_analysis_form)
    ↓ 获取表单数据
    {
        'stock_symbol': 'AAPL',
        'market_type': '美股',
        'research_depth': 3,
        'llm_provider': 'google',
        'analysts': ['market', 'fundamentals', 'news', 'social']
    }
    ↓
run_stock_analysis() (analysis_runner.py:100)
    ↓ 格式化和配置
    {
        'formatted_symbol': 'AAPL',
        'config': {
            'llm_provider': 'google',
            'quick_think_llm': 'gemini-2.5-flash',
            'deep_think_llm': 'gemini-2.5-flash',
            'max_debate_rounds': 1,
            'max_risk_discuss_rounds': 2
        }
    }
    ↓
TradingAgentsGraph.propagate() (行号446)
    ↓ 返回state和decision
    {
        'state': {
            'market_report': '技术面分析...',
            'fundamentals_report': '基本面分析...',
            'news_report': '新闻分析...',
            'sentiment_report': '情绪分析...',
            'messages': [...]
        },
        'decision': {
            'action': '买入',
            'target_price': 150.50,
            'confidence': 0.85,
            'risk_score': 0.3
        }
    }
    ↓
format_analysis_results() (analysis_runner.py:598)
    ↓ 格式化为前端友好格式
    {
        'success': True,
        'action': '买入',
        'target_price': 150.50,
        'confidence': 0.85,
        'risk_score': 0.3,
        'market_report': '...',
        'fundamentals_report': '...',
        'risk_assessment': '...'
    }
    ↓
render_results() (web/components/results_display.py)
    ↓ 前端显示
```

---

## 9. 关键文件位置速查

| 功能 | 文件 | 核心行号 |
|-----|------|--------|
| 用户表单 | web/components/analysis_form.py | 120-160 |
| 分析启动 | web/utils/analysis_runner.py | 100-120 |
| 图形初始化 | tradingagents/graph/trading_graph.py | 43-194 |
| 状态初始化 | tradingagents/graph/propagation.py | 22-46 |
| 条件逻辑 | tradingagents/graph/conditional_logic.py | 10-79 |
| 市场分析 | tradingagents/agents/analysts/market_analyst.py | 266-500 |
| 基本面分析 | tradingagents/agents/analysts/fundamentals_analyst.py | 84-240 |
| 新闻分析 | tradingagents/agents/analysts/news_analyst.py | 19-330 |
| 多头研究 | tradingagents/agents/researchers/bull_researcher.py | 15-99 |
| 空头研究 | tradingagents/agents/researchers/bear_researcher.py | 15-99 |
| 交易决策 | tradingagents/agents/trader/trader.py | 10-116 |
| 信号处理 | tradingagents/graph/signal_processing.py | 11-100 |

---

## 总结

### 💡 核心特点:

1. **并行执行**: 4个分析师同时运行，充分利用LLM并发能力
2. **工具驱动**: 所有数据来自真实API调用 (get_stock_market_data_unified等)
3. **多轮交互**: LLM多次调用
   - 第1次: 分析并选择工具
   - 第2次: 基于工具结果分析
4. **综合决策**: 交易员基于所有分析做最终决策
5. **灵活配置**: 支持4种LLM和4种研究深度

### 🎯 数据流向总结:

```
初始数据 → 股票代码格式化 → LLM配置 → 并行分析师
→ 工具调用获取数据 → LLM基于数据分析 → 多空辩论
→ 研究经理综合 → 交易员最终决策 → 风险评估
→ 信号处理(JSON提取) → 前端显示
```

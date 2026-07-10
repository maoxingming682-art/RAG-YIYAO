# 药业智能咨询助手 + AI库存预测

> 基于RAG+微调+合规风控的医药零售AI产品，包含两大模块：AI药学咨询助手 + AI库存预测看板

## 快速启动

```bash
# 双击启动
一键启动.bat

# 或手动启动
cd RAG-YIYAO
.\venv\Scripts\Activate.ps1
python run.py
```

复制 `.env.example` 为 `.env`，填入自己的 LLM API Key。`.env` 已加入 `.gitignore`，不要提交。

## 访问地址

| 页面 | 地址 | 功能 |
|------|------|------|
| 药学咨询助手 | http://localhost:5006 | AI用药咨询，多轮对话 |
| RAG测试管理 | http://localhost:5006/admin/rag-logs | 查看测试问题、筛选未回答、补知识库 |
| 库存预测看板 | http://localhost:5006/forecast | 5000/9000商品预测+补货建议 |

---

## 模块一：AI药学咨询助手

### 核心能力
- 药品用法用量/不良反应/禁忌查询
- 多轮对话追问（Query Rewrite指代消解）
- 症状分诊（急症/重症/中症/轻症）
- 超纲兜底（知识库没有的不编造）
- 4种回答风格切换（用法用量/不良反应/禁忌/一般）
- 流式输出（边生成边显示）
- 测试日志后台（筛选未回答/无效输入/需澄清/需复核）
- 知识库更新闭环（人工审核后入库，再重建向量库）

### 测试管理与知识库更新
- 入口：`http://localhost:5006/admin/rag-logs`
- 建议在 `.env` 设置 `ADMIN_TOKEN=`，公网分享/ngrok 测试时后台不会裸露。
- 用户测试问题会进入 `logs/audit_log.jsonl`，后台可按“待补知识库、需要澄清、无效输入、需复核”等状态筛选。
- 对无法回答的问题，可点击“AI生成草稿”自动填入标准问题、草稿答案和建议核验来源，再由人工审核修改后保存为待审核知识条目。
- 审核通过后点击“入库”，会追加到本地 `data/drug_knowledge.json`；再点击“重建向量库”，生成新的 `data/chunks.json` 和 `data/vectors.npy`。
- 医疗知识不要把用户问题自动写入知识库，必须人工确认来源和答案。

### 10层安全架构
```
用户提问 → 脱敏 → 分诊 → 无效输入拦截 → 闲聊拦截
→ Query Rewrite → 向量检索+关键词召回 → 超纲检测
→ 问题类型识别 → 大模型生成 → 禁用词过滤 → 免责声明 → 审计日志
```

### 技术栈
| 组件 | 技术 |
|------|------|
| 知识库 | 4万条药品知识，BAAI/bge-small-zh-v1.5向量检索 |
| 检索 | 混合检索（向量语义+关键词召回+融合加分） |
| 生成 | GLM-5.1/5.2/4.7-flash多API轮询，流式输出 |
| 微调 | QLoRA微调Qwen2.5-1.5B（离线兜底） |
| 安全层 | 18_safety_layer.py（分级免责+引导就医） |

---

## 模块二：AI库存预测看板

### 核心能力
- 5000/9000商品未来30天销量预测
- 5模型自动选优+Ensemble集成
- Transformer深度学习对比
- 数据漂移检测+一键重训
- AI生成补货操作清单
- 双数据集切换（1年/5000 vs 2年/9000）

### 5模型选优架构
| 模型 | 类型 | 擅长场景 | 胜出占比 |
|------|------|---------|---------|
| ExponentialSmoothing | 统计模型 | 趋势+季节性 | 39% |
| Ensemble（加权集成） | 集成方法 | 稳健兜底 | 32.3% |
| LinearRegression | 机器学习 | 缓慢趋势 | 13.2% |
| MovingAverage | 统计基线 | 平稳序列 | 12.7% |
| SeasonalNaive | 统计基线 | 强星期周期 | 2.7% |

### Transformer对比
| 场景 | 多模型选优 | Transformer | 胜者 |
|------|-----------|-------------|------|
| 1年/5000商品 | 14.2% | 17.4% | 多模型 |
| 2年/9000商品 | 9.8% | 14.4% | 多模型 |
| 140门店/500商品 | 9.8% | 11.3% | 多模型 |

> 结论：当前模拟数据下多模型选优更优；Transformer在真实数据+多变量融合后会反超

### 关键数字
- 知识库：4万条药品知识
- 预测商品：5000（1年）/ 9000（2年）
- 平均MAPE：15.0%（1年）→ 9.8%（2年）
- Ensemble最优：MAPE 12.3%
- 漂移商品：1394个（28%）

---

## 项目结构

```
RAG-YIYAO\
├── 一键启动.bat              ← 双击启动
├── run.py                    ← 启动入口
├── index.html                ← 药学咨询前端
├── admin_rag_logs.html       ← RAG测试管理后台
├── forecast.html             ← 库存预测看板前端
├── 使用说明.md               ← 详细使用文档
├── 面试话术.md               ← 面试讲解话术
├── 5000商品选型报告.md       ← 模型选型分析
├── .env.example              ← API配置模板（不含真实Key）
│
├── src/
│   ├── app.py                ← Flask主服务（API+路由）
│   ├── config.py             ← 配置
│   ├── llm_pool.py           ← 多API轮询+流式
│   ├── pipeline.py           ← 库存预测统一管线框架
│   ├── 17_triage.py          ← 症状分诊
│   ├── 18_safety_layer.py    ← 安全层
│   ├── 25_multi_model_forecast.py  ← 5模型选优
│   ├── 26_transformer_forecast.py  ← PyTorch Transformer
│   ├── 27_transformer_2year_9000.py ← 2年9000商品对比
│   ├── 28_multistore_transformer.py ← 140门店多门店对比
│   └── ...
│
├── data/
│   ├── README.md             ← 大数据文件说明
│   ├── drug_aliases.csv      ← 药品别名表
│   ├── rag_dialog_eval_cases.json ← 多轮评估用例
│   ├── chunks.json           ← 知识库文本（大文件，不提交）
│   ├── vectors.npy           ← 向量库（大文件，不提交）
│   ├── sales_data_5000.csv   ← 5000商品1年数据（大文件，不提交）
│   ├── sales_data_9000_2year.csv ← 9000商品2年数据（大文件，不提交）
│   ├── forecast_5000_result.json ← 5000商品预测结果
│   ├── forecast_9000_result.json ← 9000商品预测结果
│   └── transformer_result.json   ← Transformer预测结果
│
├── lora_output/              ← 微调模型权重
└── logs/                     ← 日志+预测结果
```

---

## 技术架构

### 药学咨询（RAG）
```
用户提问
→ 脱敏（正则替换隐私信息）
→ 分诊（规则引擎+LLM分诊，急症重症拦截）
→ Query Rewrite（大模型补全追问主语）
→ 混合检索（bge向量+关键词召回+融合加分）
→ 超纲检测（相似度<0.65拦截防编造）
→ 问题类型识别（4种风格切换）
→ 大模型生成（GLM API流式输出）
→ 禁用词过滤（广告法合规）
→ 免责声明（分级模板）
→ 审计日志
```

### 库存预测（ML管线）
```
ERP数据
→ 多表join构建特征矩阵
→ 5模型并行预测（ETS/SN/MA/LR/Ensemble）
→ 验证集PK选最优
→ 漂移检测（偏移>20%+统计显著）
→ LLM解读数字→补货操作清单
→ 看板展示
```

---

## 面试核心话术

### 30秒电梯演讲
"我做了医药零售AI产品，两个模块：药学咨询基于RAG+10层安全架构，4万条知识库，解决大模型幻觉和医疗合规；库存预测用5模型选优+Ensemble，覆盖9000商品，MAPE 9.8%，还有Transformer对比验证。"

### 技术亮点
1. **防幻觉**：超纲检测+混合检索+知识库约束
2. **多轮对话**：Query Rewrite解决指代丢失
3. **混合检索**：向量+关键词召回解决剂型不匹配
4. **合规风控**：10层安全架构，分诊拦截+禁用词+免责
5. **模型选优**：5模型竞赛+Ensemble集成，MAPE 12.3%
6. **Transformer对比**：3轮实测证明数据量决定模型选型
7. **流式输出**：SSE逐token推送
8. **QLoRA微调**：1.5B离线兜底

---

## 环境要求

- Python 3.11
- GPU: NVIDIA 4070 Ti SUPER（Transformer训练用）
- 依赖: flask, openai, torch, transformers, statsmodels, sklearn, prophet, pandas, numpy

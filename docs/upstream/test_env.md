**Please ensure your .env file is in the agentflow directory, and make sure the following API keys are configured:**
- `OPENAI_API_KEY` (used for RAG summary in tools)
- `GOOGLE_API_KEY` (for Google Search tool)
- `DASHSCOPE_API_KEY` (for calling Qwen-2.5-7B-Instruct - recommended for China/Singapore users)

**If you are not using these three default APIs, please adjust them in the corresponding locations.**

For example: adjust the model in [`tools/base_generator/tool.py`](../../agentflow/agentflow/tools/base_generator/tool.py) to `together-Qwen/Qwen2.5-7B-Instruct`

---

## Test your env before going on

Please ensure tools, engine and IP are properly configured before proceeding.

### Test tools
please run the following command to test all tools: 

```bash
cd agentflow/agentflow
bash ./tools/test_all_tools.sh
```

A `test.log` will be saved in each tool's file. 

Success example: 
```text
Testing all tools
Tools:
  - base_generator
  - google_search
  - python_coder
  - web_search
  - wikipedia_search

Running tests in parallel...
Testing base_generator...
✅ base_generator passed
Testing google_search...
✅ google_search passed
Testing python_coder...
✅ python_coder passed
Testing wikipedia_search...
✅ wikipedia_search passed
Testing web_search...
✅ web_search passed

✅ All tests passed
```

### LLM engine test
Please run the following command to test all LLM engines:

```bash
cd PROJECT_ROOT
python agentflow/scripts/test_llm_engine.py
```

Example output:
```text
🚀 Starting fault-tolerant test for 11 engines...
🧪 Testing: 'gpt-4o' | kwargs={}
✅ Success: Created ChatOpenAI
🧪 Testing: 'dashscope-qwen2.5-3b-instruct' | kwargs={}
✅ Success: Created ChatDashScope
🧪 Testing: 'gemini-1.5-pro' | kwargs={}
✅ Success: Created ChatGemini
============================================================
📋 TEST SUMMARY
============================================================
✅ Passed: 3
   • gpt-4o → ChatOpenAI
   • dashscope-qwen2.5-3b-instruct → ChatDashScope
   • gemini-1.5-pro → ChatGemini
❌ Failed: 8
   • azure-gpt-4 → 🚫 API key not found in environment
   • claude-3-5-sonnet → 🚫 API key not found in environment
   • deepseek-chat → 🚫 API key not found in environment
   • grok → 🚫 API key not found in environment
   • vllm-meta-llama/Llama-3-8b-instruct → 🚫 Connection failed
   • together-meta-llama/Llama-3-70b-chat-hf → 🚫 API key not found
   • ollama-llama3 → 🚫 Connection failed
   • unknown-model-123 → 💥 Unexpected error
============================================================
🎉 Testing complete. Script did NOT crash despite errors.
```

### IP test
test your public IP(just for saving the logs files)
```bash
cd PROJECT_ROOT
python util/get_pub_ip.py
```


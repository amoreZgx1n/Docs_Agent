#!/usr/bin/env python3
"""
Ollama Agent 流式工具调用最小demo
演示如何实现边生成边等待工具返回结果的机制
"""

import json
import re
import time
import requests
from typing import Dict, Any, List, Generator

class OllamaAgent:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen3:8b"  # 你可以改成你的模型名
        
    def get_tools_prompt(self) -> str:
        """定义工具的系统提示"""
        return """你是一个智能助手，可以调用以下工具：

可用工具：
1. search_web(query: str) - 搜索网络信息
2. calculate(expression: str) - 计算数学表达式
3. get_weather(city: str) - 获取天气信息

调用工具时，请使用以下格式：
<tool_call>
<tool_name>工具名称</tool_name>
<parameters>{"参数名": "参数值"}</parameters>
</tool_call>

工具调用完成后，你会收到结果，然后继续回答。"""

    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """模拟工具调用"""
        print(f"\n🔧 调用工具: {tool_name}")
        print(f"📝 参数: {parameters}")
        
        # 模拟工具执行时间
        time.sleep(1)
        
        # 模拟不同工具的返回结果
        if tool_name == "search_web":
            query = parameters.get("query", "")
            return f"搜索结果：关于'{query}'的最新信息显示..."
            
        elif tool_name == "calculate":
            expression = parameters.get("expression", "")
            try:
                result = eval(expression)  # 注意：实际项目中不要用eval
                return f"计算结果：{expression} = {result}"
            except:
                return f"计算错误：无法计算 {expression}"
                
        elif tool_name == "get_weather":
            city = parameters.get("city", "")
            return f"天气信息：{city}今天晴天，温度25°C"
            
        else:
            return f"未知工具：{tool_name}"

    def stream_generate(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """流式生成响应"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data and 'content' in data['message']:
                            yield data['message']['content']
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"生成错误: {str(e)}"

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """从生成的文本中提取工具调用"""
        tool_calls = []
        
        # 使用正则表达式匹配工具调用
        pattern = r'<tool_call>\s*<tool_name>(.*?)</tool_name>\s*<parameters>(.*?)</parameters>\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for tool_name, params_str in matches:
            tool_name = tool_name.strip()
            try:
                parameters = json.loads(params_str.strip())
                tool_calls.append({
                    "tool_name": tool_name,
                    "parameters": parameters
                })
            except json.JSONDecodeError:
                print(f"⚠️ 参数解析错误: {params_str}")
                
        return tool_calls

    def chat(self, user_input: str) -> str:
        """主对话函数 - 实现边生成边调用工具的逻辑"""
        messages = [
            {"role": "system", "content": self.get_tools_prompt()},
            {"role": "user", "content": user_input}
        ]
        
        print(f"👤 用户: {user_input}")
        print("🤖 助手: ", end="", flush=True)
        
        full_response = ""
        buffer = ""
        
        # 流式生成响应
        for chunk in self.stream_generate(messages):
            buffer += chunk
            full_response += chunk
            print(chunk, end="", flush=True)
            
            # 检查是否有完整的工具调用
            tool_calls = self.extract_tool_calls(buffer)
            
            if tool_calls:
                print("\n")  # 换行
                
                # 执行所有检测到的工具调用
                tool_results = []
                for tool_call in tool_calls:
                    result = self.call_tool(
                        tool_call["tool_name"], 
                        tool_call["parameters"]
                    )
                    tool_results.append(result)
                
                # 将工具结果添加到对话历史
                messages.append({"role": "assistant", "content": full_response})
                
                # 添加工具结果
                tool_results_text = "\n".join([f"工具结果: {result}" for result in tool_results])
                messages.append({"role": "user", "content": f"工具执行完成。{tool_results_text}\n请继续回答。"})
                
                print(f"\n🔄 继续生成...")
                print("🤖 助手: ", end="", flush=True)
                
                # 继续生成后续响应
                buffer = ""
                for chunk in self.stream_generate(messages):
                    buffer += chunk
                    full_response += chunk
                    print(chunk, end="", flush=True)
        
        print("\n" + "="*50)
        return full_response

def main():
    """演示函数"""
    agent = OllamaAgent()
    
    print("🚀 Ollama Agent 流式工具调用 Demo")
    print("="*50)
    
    # 测试用例
    test_cases = [
        "帮我搜索一下人工智能的最新发展",
        "计算 125 + 237 * 3 的结果",
        "查询北京的天气情况",
        "先搜索Python教程，然后计算 100 + 200，最后查询上海天气"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}:")
        try:
            agent.chat(test_case)
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        if i < len(test_cases):
            input("\n按回车键继续下一个测试...")

if __name__ == "__main__":
    # 检查依赖
    try:
        import requests
        print("✅ 依赖检查通过")
    except ImportError:
        print("❌ 请先安装 requests: pip install requests")
        exit(1)
    
    # 运行演示
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
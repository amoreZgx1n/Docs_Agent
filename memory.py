"""
Memory模块 - 智能记忆管理系统

功能：
1. 短期记忆：使用ChatMemoryBuffer保存最近的对话历史
2. 长期记忆：使用文件存储和LLM总结机制保存重要信息
3. 智能总结：定期总结短期记忆到长期记忆
4. 上下文检索：根据当前查询检索相关记忆
5. 持久化存储：支持记忆数据的保存和加载

设计特点：
- 模块化设计，方便外部调用
- 支持多轮对话记忆
- 自动总结机制
- 用户画像学习
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.zhipuai import ZhipuAI


os.environ['ZHIPUAI_API_KEY'] = 'bdefa5d91470437380967660cf6592f0.UN5HkGKi6hYu8kVB'



@dataclass
class MemoryEntry:
    """记忆条目 - 简化版"""
    timestamp: str
    summary: str                    # 对话总结
    theme: str = ""                 # 主题
    keywords: List[str] = None      # 关键词
    turn_count: int = 0             # 对话轮数
    importance_score: float = 0.0   # 重要性评分
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class MemoryManager:
    """
    智能记忆管理器
    
    核心功能：
    - 短期记忆：基于LlamaIndex ChatMemoryBuffer的滑动窗口
    - 长期记忆：基于文件存储和LLM总结的持久化记忆
    - 智能总结：定期将短期记忆总结到长期记忆
    - 用户建模：学习用户偏好和行为模式
    """
    
    def __init__(self, 
                 short_memory_size: int = 20,
                 long_memory_file: str = "long_term_memory.json",
                 summary_threshold: int = 5):
        """
        初始化记忆管理器
        
        Args:
            short_memory_size: 短期记忆的消息数量限制
            long_memory_file: 长期记忆文件路径
            summary_threshold: 触发总结的消息数量阈值
        """
        # 初始化大模型
        self.llm = ZhipuAI(model="glm-4-flash", api_key=os.environ['ZHIPUAI_API_KEY'])
        
        # 短期记忆：使用LlamaIndex的ChatMemoryBuffer
        self.short_term_memory = ChatMemoryBuffer.from_defaults(
            chat_store=SimpleChatStore(),
            chat_store_key="conversation",
            token_limit=4000  # 限制token数量
        )
        
        # 长期记忆文件路径
        self.long_memory_file = long_memory_file
        self.summary_threshold = summary_threshold
        
        # 加载长期记忆
        self.long_term_memory: List[MemoryEntry] = self._load_long_term_memory()
        
        # 统计信息
        self.stats = {
            "total_messages": 0,
            "summary_count": 0,
            "last_summary_time": "",
            "memory_efficiency": 0.0
        }
        
        print(f"记忆管理器初始化完成")
        print(f"- 短期记忆限制: {short_memory_size} 条消息")
        print(f"- 长期记忆文件: {long_memory_file}")
        print(f"- 已加载长期记忆: {len(self.long_term_memory)} 条")
    
    def add_message(self, role: str, content: str, importance: float = 1.0):
        """
        添加新消息到记忆中
        
        Args:
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            importance: 重要性评分 (0-1)
        """
        # 转换角色格式
        message_role = MessageRole.USER if role.lower() == 'user' else MessageRole.ASSISTANT
        
        # 创建消息并添加到短期记忆
        message = ChatMessage(role=message_role, content=content)
        self.short_term_memory.put(message)
        
        # 更新统计信息
        self.stats["total_messages"] += 1
        
        
        # 检查是否需要总结
        if self._should_summarize():
            self._summarize_to_long_term()
        
        print(f"已添加{role}消息到短期记忆")
    
    
    def _extract_keywords(self, text: str) -> List[str]:
        """简单的关键词提取"""
        # 这里使用简单的方法，实际应用中可以使用更复杂的NLP技术
        import re
        
        # 移除标点符号并分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤停用词（简化版）
        stop_words = {'我', '你', '他', '她', '它', '的', '了', '是', '在', '有', '和', '与', '或', '但', '如果', '那么', '这个', '那个'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords[:5]  # 返回前5个关键词
    
    def _should_summarize(self) -> bool:
        """判断是否应该进行总结"""
        current_messages = self.short_term_memory.get_all()
        return len(current_messages) >= self.summary_threshold * 2  # 每N轮对话总结一次
    
    def _summarize_to_long_term(self):
        """将短期记忆总结到长期记忆"""
        try:
            # 获取短期记忆中的所有消息
            messages = self.short_term_memory.get_all()
            if not messages:
                return
            
            # 构建对话历史文本
            conversation_text = "\n".join([
                f"{msg.role.value}: {msg.content}" for msg in messages
            ])
            
            # 构建简化总结提示
            summary_prompt = f"""
            请总结以下对话内容，用JSON格式返回：

            对话内容：
            {conversation_text}

            返回格式：
            {{
                "summary": "对话的核心内容总结（50-100字）",
                "theme": "主要话题或主题",
                "keywords": ["关键词1", "关键词2", "关键词3"],
                "turn_count": {len(messages)},
                "importance_score": 0.7
            }}

            要求：
            1. summary必须是具体的内容总结，不要写"对话总结(X条消息)"这种无意义内容
            2. theme要简洁明确
            3. keywords最多5个，要有实际意义
            4. 必须返回有效JSON格式
            """
            
            # 调用LLM进行总结
            response = self.llm.complete(summary_prompt)
            
            try:
                # 解析LLM返回的JSON
                summary_data = json.loads(response.text)
                
                # 创建记忆条目
                memory_entry = MemoryEntry(
                    timestamp=datetime.datetime.now().isoformat(),
                    summary=summary_data.get("summary", ""),
                    theme=summary_data.get("theme", ""),
                    keywords=summary_data.get("keywords", []),
                    turn_count=summary_data.get("turn_count", len(messages)),
                    importance_score=summary_data.get("importance_score", 0.5)
                )
                
                # 添加到长期记忆
                self.long_term_memory.append(memory_entry)
                
                # 保存到文件
                self._save_long_term_memory()
                
                # 更新统计信息
                self.stats["summary_count"] += 1
                self.stats["last_summary_time"] = datetime.datetime.now().isoformat()
                
                print(f"成功总结 {len(messages)} 条消息到长期记忆")
                
                # 清空部分短期记忆，保留最近的几条
                self._trim_short_memory()
                
            except json.JSONDecodeError:
                print("LLM返回的总结格式不正确，跳过本次总结")
                print(f"LLM原始回复: {response.text}")
                # 不保存无意义的总结，直接跳过
                
        except Exception as e:
            print(f"总结过程中出错: {e}")
    
    def _trim_short_memory(self):
        """修剪短期记忆，保留最近的消息"""
        messages = self.short_term_memory.get_all()
        if len(messages) > 6:  # 保留最近3轮对话
            # 重新初始化短期记忆
            self.short_term_memory.reset()
            # 添加最近的消息
            for msg in messages[-6:]:
                self.short_term_memory.put(msg)
    
    def get_relevant_context(self, query: str, max_items: int = 5) -> Dict[str, Any]:
        """
        根据查询获取相关的记忆上下文
        
        Args:
            query: 当前查询
            max_items: 最大返回项目数
            
        Returns:
            相关上下文字典
        """
        context = {
            "short_term": [],
            "long_term": [],
            "relevant_memories": []
        }
        
        # 获取短期记忆
        short_messages = self.short_term_memory.get_all()
        context["short_term"] = [
            {"role": msg.role.value, "content": msg.content} 
            for msg in short_messages[-max_items:]
        ]
        
        # 搜索相关的长期记忆
        relevant_memories = self._search_long_term_memory(query, max_items)
        context["long_term"] = [asdict(memory) for memory in relevant_memories]
        
        return context
    
    def _search_long_term_memory(self, query: str, max_items: int) -> List[MemoryEntry]:
        """搜索相关的长期记忆"""
        if not self.long_term_memory:
            return []
        
        # 简单的关键词匹配搜索（可以改进为语义搜索）
        query_keywords = self._extract_keywords(query)
        scored_memories = []
        
        for memory in self.long_term_memory:
            score = 0.0
            
            # 基于关键词匹配计算相关性分数
            for keyword in query_keywords:
                if keyword in memory.summary.lower():
                    score += 1.0
                if keyword in memory.theme.lower():
                    score += 1.5
                if keyword in memory.keywords:
                    score += 1.2
            
            # 考虑重要性分数
            score *= memory.importance_score
            
            if score > 0:
                scored_memories.append((score, memory))
        
        # 按分数排序并返回前N个
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:max_items]]
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统的整体摘要"""
        short_messages = self.short_term_memory.get_all()
        
        return {
            "stats": self.stats,
            "short_term_count": len(short_messages),
            "long_term_count": len(self.long_term_memory),
            "memory_efficiency": len(self.long_term_memory) / max(self.stats["total_messages"], 1)
        }
    
    def clear_memory(self, memory_type: str = "short"):
        """
        清空记忆
        
        Args:
            memory_type: "short", "long", "all"
        """
        if memory_type in ["short", "all"]:
            self.short_term_memory.reset()
            print("短期记忆已清空")
        
        if memory_type in ["long", "all"]:
            self.long_term_memory.clear()
            self._save_long_term_memory()
            print("长期记忆已清空")
        
        if memory_type == "all":
            self.stats = {
                "total_messages": 0,
                "summary_count": 0,
                "last_summary_time": "",
                "memory_efficiency": 0.0
            }
            print("统计信息已重置")
    
    def _load_long_term_memory(self) -> List[MemoryEntry]:
        """从文件加载长期记忆"""
        try:
            if os.path.exists(self.long_memory_file):
                with open(self.long_memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    memories = []
                    for item in data:
                        # 兼容旧格式转换为新格式
                        try:
                            # 新格式直接使用
                            if 'summary' in item and 'theme' in item:
                                memory_entry = MemoryEntry(
                                    timestamp=item.get('timestamp', ''),
                                    summary=item.get('summary', ''),
                                    theme=item.get('theme', ''),
                                    keywords=item.get('keywords', []),
                                    turn_count=item.get('turn_count', 0),
                                    importance_score=item.get('importance_score', 0.5)
                                )
                            else:
                                # 旧格式转换
                                summary = item.get('content', item.get('summary', ''))
                                # 跳过无意义的总结
                                if '对话总结' in summary and '条消息' in summary:
                                    continue
                                
                                memory_entry = MemoryEntry(
                                    timestamp=item.get('timestamp', ''),
                                    summary=summary,
                                    theme=item.get('conversation_theme', item.get('theme', '')),
                                    keywords=item.get('tags', item.get('keywords', [])),
                                    turn_count=0,
                                    importance_score=item.get('importance_score', 0.5)
                                )
                            
                            memories.append(memory_entry)
                        except Exception as e:
                            print(f"跳过无效记忆条目: {e}")
                            continue
                    
                    return memories
        except Exception as e:
            print(f"加载长期记忆失败: {e}")
        
        return []
    
    def _save_long_term_memory(self):
        """保存长期记忆到文件"""
        try:
            data = [asdict(memory) for memory in self.long_term_memory]
            with open(self.long_memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存长期记忆失败: {e}")
    


class ConversationTester:
    """对话测试器 - 用于终端测试记忆功能"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.conversation_count = 0
    
    def start_conversation(self):
        """开始对话测试"""
        print("\n" + "="*60)
        print("智能记忆系统 - 对话测试")
        print("="*60)
        print("说明:")
        print("- 输入 'quit' 或 'exit' 退出")
        print("- 输入 'memory' 查看记忆摘要") 
        print("- 输入 'clear' 清空短期记忆")
        print("- 输入 'save' 保存当前状态")
        print("- 开始对话，系统会记住你说的内容")
        print("="*60)
        
        while True:
            try:
                user_input = input(f"\n[用户 {self.conversation_count+1}] 你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit']:
                    print("再见！记忆已保存。")
                    break
                elif user_input.lower() == 'memory':
                    self._show_memory_summary()
                    continue
                elif user_input.lower() == 'clear':
                    self.memory_manager.clear_memory('short')
                    continue
                elif user_input.lower() == 'save':
                    self._save_state()
                    continue
                
                # 添加用户消息到记忆
                self.memory_manager.add_message('user', user_input)
                
                # 生成回复（这里使用简单的规则，实际可以集成更复杂的对话系统）
                assistant_reply = self._generate_reply(user_input)
                
                # 添加助手回复到记忆
                self.memory_manager.add_message('assistant', assistant_reply)
                
                print(f"[助手] {assistant_reply}")
                
                self.conversation_count += 1
                
                # 每5轮对话显示一次记忆状态
                if self.conversation_count % 5 == 0:
                    print(f"\n--- 已进行 {self.conversation_count} 轮对话 ---")
                    self._show_brief_memory()
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，记忆已保存。")
                break
            except Exception as e:
                print(f"出现错误: {e}")
    
    def _generate_reply(self, user_input: str) -> str:
        """生成助手回复（智能版本）"""
        # 获取相关上下文
        context = self.memory_manager.get_relevant_context(user_input)
        
        # 构建包含记忆的提示词
        prompt = self._build_conversation_prompt(user_input, context)
        
        try:
            # 使用LLM生成智能回复
            response = self.memory_manager.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"LLM调用失败: {e}")
            # 降级到简单回复
            return self._simple_reply(user_input, context)
    
    def _build_conversation_prompt(self, user_input: str, context: Dict) -> str:
        """构建包含记忆上下文的对话提示词"""
        
        # 构建对话历史
        conversation_history = ""
        if context["short_term"]:
            conversation_history = "最近对话历史：\n"
            for msg in context["short_term"][-4:]:  # 最近4条消息
                role = "用户" if msg["role"] == "user" else "助手"
                conversation_history += f"{role}: {msg['content']}\n"
        
        # 构建相关记忆
        relevant_memory = ""
        if context["long_term"]:
            relevant_memory = "相关记忆：\n"
            for memory in context["long_term"][:2]:  # 最相关的2条记忆
                relevant_memory += f"- 主题：{memory.get('theme', '未知')}\n"
                relevant_memory += f"  内容：{memory['summary']}\n"
        
        # 组合完整提示词
        prompt = f"""你是一个智能助手，具有记忆能力。请根据以下信息回答用户的问题：

{conversation_history}{relevant_memory}
当前用户问题：{user_input}

请注意：
1. 如果是数学问题，请详细计算并给出准确答案
2. 如果涉及用户个人信息，要体现出你记住了相关内容
3. 保持友好、有帮助的语调
4. 回复要简洁明了，不要过长

回复："""
        
        return prompt
    
    def _simple_reply(self, user_input: str, context: Dict) -> str:
        """简单回复逻辑（降级方案）"""
        # 简单的回复逻辑
        if "我叫" in user_input or "我是" in user_input:
            return "很高兴认识你！我会记住我们的对话。"
            
        elif "你还记得" in user_input or "你知道我" in user_input:
            if context.get("long_term"):
                return "我记得我们之前的一些对话内容。"
            else:
                return "抱歉，我还没有记住太多关于你的信息。"
        
        elif "谢谢" in user_input or "感谢" in user_input:
            return "不客气！有什么其他问题随时告诉我。"
        
        # 检查是否是数学问题
        elif any(op in user_input for op in ['+', '-', '×', '*', '÷', '/', '=', '计算', '等于']):
            return "抱歉，我的数学计算功能暂时不可用。请稍后再试或联系技术支持。"
        
        else:
            # 默认回复
            return f"我理解了。请继续告诉我更多，我会记住我们的对话。"
    
    def _show_memory_summary(self):
        """显示记忆摘要"""
        summary = self.memory_manager.get_memory_summary()
        print("\n" + "-"*40)
        print("记忆系统摘要")
        print("-"*40)
        print(f"总消息数: {summary['stats']['total_messages']}")
        print(f"短期记忆: {summary['short_term_count']} 条")
        print(f"长期记忆: {summary['long_term_count']} 条")
        print(f"总结次数: {summary['stats']['summary_count']}")
        print(f"记忆效率: {summary['memory_efficiency']:.2%}")
        print("-"*40)
    
    
    def _show_brief_memory(self):
        """显示简要记忆状态"""
        summary = self.memory_manager.get_memory_summary()
        print(f"短期记忆: {summary['short_term_count']} | 长期记忆: {summary['long_term_count']}")
    
    def _save_state(self):
        """保存当前状态"""
        self.memory_manager._save_long_term_memory()
        print("当前状态已保存！")


def main():
    """主函数 - 启动对话测试"""
    tester = ConversationTester()
    tester.start_conversation()


if __name__ == "__main__":
    main()

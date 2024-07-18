from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import(
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.messages import HumanMessage, AIMessage
from getpass import getpass
from chatbot.algorithm_code import AlgorithmCode
from utils import LOG

import re

class ChatBot:
    def __init__(self, model_name : str, api_key : str, base_url : str):
        try:
            chatmodel = ChatOpenAI(
                model = model_name,
                api_key = api_key,
                base_url = base_url
            )
            self.chatmodel = chatmodel

            algorithmcode = AlgorithmCode(model_name, api_key, base_url)
            self.algorithmcode = algorithmcode

            system_template1 = (
                """
                你是国际信息学奥林匹克竞赛冠军,并且也是国际大学生程序设计大赛ICPC-WF冠军。
                对如下代码，请您将代码重新排版同时给代码逐行注释，使得代码看起来条理清晰。
                如果你不知道答案，请您说不知道。
                """
            )
            system_prompt1 = SystemMessagePromptTemplate.from_template(system_template1)

            human_template1 = ("{input}")
            human_prompt1 = HumanMessagePromptTemplate.from_template(human_template1)
            self.human_prompt1 = human_prompt1

            prompt_analysis = ChatPromptTemplate.from_messages([system_prompt1, human_prompt1])
            self.prompt_analysis = prompt_analysis
            
            system_template2 = (
                """
                你是国际信息学奥林匹克竞赛冠军,并且也是国际大学生程序设计大赛ICPC-WF冠军。
                请展示您的算法储备知识，仔细严谨的和我交流。
                """
            )
            system_prompt2 = SystemMessagePromptTemplate.from_template(system_template2)

            self.system_prompt2 = system_prompt2

        except Exception as e:
            LOG.error(f"Initialization failed: {e}")
            raise

    def CodeAnalysis(self, input : str, score_threshold : int):
        try:
            chain = LLMChain(llm = self.chatmodel, prompt = self.prompt_analysis)
            code, statue, id = self.algorithmcode.GetAlgorithmCode(input, score_threshold)
            LOG.debug(f"chatbot:\n[code]{code}\n[statue]{statue}\n[id]{id}")
            result = code
            if statue == 1:#获取代码，通过大模型分析代码
                result = chain.invoke({'input' : code})['text']
                LOG.debug(f"[chatbot : Algorithm analysis results already exist]{result}")
            elif statue == 2:#数据库中已经存在分析代码，直接输出
                LOG.debug(f"[chatbot : The algorithm analysis result does not exist, analyze through code]{result}")

            return result, statue, id
        
        except Exception as e:
            LOG.error(f"chatbot : Retrieval failed: {e}")
            return "An error occurred during retrieval"

    def AlgorithmExchange(self, input : str, history : list):
        try:
            prompt_list = [self.system_prompt2]
            LOG.debug(f"[history len]{len(history)}")
            if len(history) > 0:
                LOG.debug(f"[history message]{history[0][0]}")
                human_template0 = (self.escape_braces(history[0][0]))
                LOG.debug(f"[history message]{ human_template0}")
                human_prompt0 = HumanMessagePromptTemplate.from_template(human_template0)
                prompt_list.append(human_prompt0)
                LOG.debug(f"[history_human]{human_prompt0}")

                LOG.debug(f"[history return]{history[0][1]}")
                ai_template0 = (self.escape_braces(history[0][1]))
                ai_prompt0 = AIMessagePromptTemplate.from_template(ai_template0)
                prompt_list.append(ai_prompt0)
                LOG.debug(f"[history_ai]{ai_prompt0}")
            
            prompt_list.append(self.human_prompt1)
            prompt_exchange = ChatPromptTemplate.from_messages(prompt_list)
            chain = LLMChain(llm = self.chatmodel, prompt = prompt_exchange)
            result = chain.invoke({'input' : input})['text']
            return result
        
        except Exception as e:
            LOG.error(f"chatbot : Retrieval failed: {e}")
            return "An error occurred during retrieval"
        
    def escape_braces(self, template : str):
        # 用正则表达式找到所有的花括号并进行转义
        escaped_template = re.sub(r'(\{)', r'{{', template)
        escaped_template = re.sub(r'(\})', r'}}', escaped_template)
        LOG.debug(f"[re to template]{escaped_template}")
        return escaped_template
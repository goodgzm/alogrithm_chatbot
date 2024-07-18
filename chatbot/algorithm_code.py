from langchain_openai import ChatOpenAI
from sql.mysql import SQLite
from chatbot.algorithm_describe import AlgorithmDescription
from langchain.chains import LLMChain
from langchain_core.prompts import(
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

from utils import LOG

class AlgorithmCode:
    def __init__(self, model_name : str, api_key : str, base_url : str):
        try:
            chatmodel = ChatOpenAI(
                model = model_name,
                api_key = api_key,
                base_url = base_url
            )
            self.chatmodel = chatmodel

            sqlite = SQLite()
            self.sqlite = sqlite

            algorithm__list = sqlite.Select_All_Data()

            name_list_prompt = """
你是国际信息学奥林匹克竞赛冠军,并且也是国际大学生程序设计大赛ICPC-WF冠军。
你将获得一段关于某个算法的大致描述。
请你在以下关键词中选取最相关的一个对应的序号整数作为输出:
"""
            for row in algorithm__list:
                name_list_prompt += row[1] + ":" + str(row[0]) + '\n'

            LOG.debug(f'[name_list_prompt]{name_list_prompt}')

            self.name_list_promt = name_list_prompt

            algorithm_description = AlgorithmDescription(model_name, api_key, base_url)
            self.algorithm_description = algorithm_description
        except Exception as e:
            LOG.error(f"Initialization failed: {e}")
            raise
    
    def GetAlgorithmCode(self, input : str, score_threshold : int):
        try:
            algorithm_chao_name, statue = self.algorithm_description.GetAlgorithmChaoName(input, score_threshold)
            LOG.debug(f"[algorithm_chao_name]{algorithm_chao_name}")
            if not statue:
                return algorithm_chao_name, statue
            
            system_template = (self.name_list_promt)
            system_prompt = SystemMessagePromptTemplate.from_template(system_template)

            human_template1 = ("请您仅输出序号整数!!!\n以下是我的输入:\n{input}")
            human_prompt1 = HumanMessagePromptTemplate.from_template(human_template1)

            prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt1])

            chain = LLMChain(llm = self.chatmodel, prompt = prompt)

            algorithm_id = chain.invoke({'input' : algorithm_chao_name})['text']
            LOG.debug(f"[algorithm_id]{algorithm_id}")
            
            result, flag = self.sqlite.Select_Data_By_Id(algorithm_id)

            if flag == 2:
                LOG.debug(f"[algorithm_code : Algorithm analysis results already exist]{result}")
            elif flag == 0:
                result = "知识库中无该算法模板"
                LOG.error(f'[algorithm_code :]{result}')
            else:
                LOG.debug(f"[algorithm_code : The algorithm analysis result does not exist, and the return code]{result}")

            return result, flag, algorithm_id
        
        except Exception as e:
            LOG.error(f"Retrieval failed: {e}")
            return "An error occurred during retrieval", 0


        
        
        

    
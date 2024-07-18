import os
import gradio as gr

from utils import LOG
from utils import ChatBotConfig
from utils import NoiseRemoval
from chatbot import ChatBot
from sql import DataTODB

def init(): 
    dir_path = os.path.dirname(os.path.abspath(__file__))
    config = ChatBotConfig()
    config.init(dir_path=dir_path)
    global chatbot, noiseremoval, db
    db = DataTODB()

    chatbot = ChatBot(config.model_name, config.api_key, config.base_url) 
    noiseremoval = NoiseRemoval()

    global slider_value_now, bot_model_now
    slider_value_now = 0.1
    bot_model_now = "算法模板模式"

def get_slider_value(slider_value):
    global slider_value_now
    slider_value_now = slider_value
    LOG.debug(f"[slider_value_now]{slider_value_now}")
    return slider_value

def get_bot_model(bot_model):
    global bot_model_now
    bot_model_now = bot_model
    LOG.debug(f"[bot_model_now]{bot_model_now}")
    return bot_model

def algorithm_chat(message, history):
    try:
        LOG.debug(f"[message]{message}")
        LOG.debug(f"[history]{history}")
        LOG.debug(f"[slider_value_now]{slider_value_now}")
        LOG.debug(f"[bot_model_now]{bot_model_now}")

        processed_message = noiseremoval.Do_NoiseRemoval(message)
        LOG.debug(f"[message after noise removal]{processed_message}")

        if bot_model_now == "算法模板模式":
            result, statue, id = chatbot.CodeAnalysis(processed_message, slider_value_now)

            if statue == 1:
                LOG.debug(f"[result]{result}\n[statue]{statue}\n[id]{id}")
                flag = db.Result_Online_ToDB_By_ID(id, result)
                if flag:
                    LOG.debug("Insert successful")
                else:
                    LOG.error("Insert failed")

            return result
    
        else:
            return chatbot.AlgorithmExchange(processed_message, history)
    
    except Exception as e:
        LOG.error(f"Error invoking retrieval chain: {e}")
        return "An error occurred. Please try again."

def launch_gradio():
    slider = gr.Slider(minimum=0.05, maximum=1, step=0.05, value=0.1, label="匹配阈值")
    bot_model = gr.Dropdown(choices=["算法模板模式", "算法交流模式"], label="机器人模式", value="算法模板模式")
    slider_output = gr.Label(label="当前匹配阈值", value=0.1)
    bot_model_output = gr.Label(label="当前机器人模式", value="算法模板模式")
    # 创建 ChatInterface 实例
    with gr.ChatInterface(
        fn=algorithm_chat,
        title="算法模板机器人(比赛ak机器人)",
        chatbot=gr.Chatbot(height=800),
        retry_btn=None,
        undo_btn=None,
        clear_btn=None,
        theme="soft",
        examples=["可以区间查询，区间修改的算法", "线段树区间查询", "可以快速判断字符串是否为子串"],
        css="""
        .gradio-container {
            border: 4px solid #A0C4FF;  /* 更浅的外边框 */
            border-radius: 15px;  /* 更大的圆角边框 */
            padding: 30px;  /* 内边距 */
        }
        """
    ) as chat:
            with gr.Row():
                with gr.Column():
                    txt1 = gr.Markdown(
                        """
                        <span style='color: #6A5ACD; font-size: 18px; background-color: #E0E7FF; padding: 10px; border-radius: 10px;'>【功能模块】</span>
                        """)
                    gr.Interface(fn=get_slider_value, inputs=slider, outputs=slider_output)
                    gr.Interface(fn=get_bot_model, inputs=bot_model, outputs=bot_model_output)
                with gr.Column():
                    txt1 = gr.Markdown(
                        """
                        <span style='color: #6A5ACD; font-size: 18px; background-color: #E0E7FF; padding: 10px; border-radius: 10px;'>【机器人使用说明】</span>
                        """)
                    with gr.Tab(label="匹配阈值"):
                        txt2 = gr.Markdown(
                            """
                            <span style='font-size: 16px;'>【请将阈值调高】当您需要让机器人专门匹配某个算法模板时/当机器人语无伦次时</span><br><br>
                            <span style='font-size: 16px;'>【请将阈值调低】当您不知道用算法的名称叫什么时/不知道用什么算法时/机器人匹配不到你需要的算法时</span>
                            """)
                    with gr.Tab(label="机器人模式"):
                        txt3 = gr.Markdown(
                            """
                            <span style='font-size: 16px;'>【算法模板模式】当您仅需要算法模板时</span><br><br>
                            <span style='font-size: 16px;'>【算法交流模式】当您需要和机器人对某个算法展开深入探讨时</span>
                            """)

# 启动聊天接口
    user_info = [
        ("admin", "123456"),
    ]

    chat.launch(
        share = True,
        server_port=9090,
        debug=False,
        auth=user_info,
        auth_message='欢迎登录大模型演示平台！'
    )

if __name__ == "__main__":
    init()
    launch_gradio()

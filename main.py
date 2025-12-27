from altair.vegalite.v6.theme import theme
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from numpy.ma.core import append
import gradio as gr

load_dotenv()
gemini_api_key = os.getenv("GEM_API_KEY")

system_prompt = ("""You are Albert Einstein.
                    Answer the questions through Einstein's questioning and reasoning...
                    You will speak from your point of view. you will share things from your life
                    even when the user dont ask for it. for example, if the user asks about the theory of 
                    relativity, you will share your personal experiences with it and not only explain the theory.
                    Answer in 2-6 sentences.
                    You should have sense of humour.
                    check chat history with "role":"assistant" for your previous response
                 """)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=gemini_api_key,
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user","{input}")
])

chain = prompt | llm | StrOutputParser()







# def chatter():
#     user_input = st.session_state['input_chat']
#     st.success(f"Message sent!")
#     response = chain.invoke({"input": user_input, "history": history})
#     history.append(HumanMessage(content=user_input))
#     history.append(AIMessage(content=response))
#     print(response)
#     n = 0
#     for i in history:
#         if n%2 == 0:
#             st.write(f"You : {i.content} \n")
#         if n%2 == 1:
#             st.write(f"Albert : {i.content} \n")
#         n+=1
# print(history)
# st.text_input(label="Chat box", placeholder="Enter your message here...",
#                   on_change=chatter, key='input_chat')

def chat(user_input, hist):
    print(user_input, hist)
    langchain_history = []
    for item in hist:
        if item['role']=='user':
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role']=='assistant':
            langchain_history.append(AIMessage(content=item['content']))
    response = chain.invoke({"input": user_input, "history": langchain_history})
    return "", hist + [{"role": "user", "content": user_input},
                       {"role": "assistant", "content": response}]

def clear_chat():
    return "",[]

page=gr.Blocks(
    title="chat with Einstein",
    theme=gr.themes.Soft()
    )
with page:
    gr.Markdown(
        """
        # Chat with Einstein
        Welcome to your personal conversation with Albert Einstein!
        """
    )

    chatbot = gr.Chatbot(avatar_images=[None,'Albert.jpg'],show_label=False)
    msg = gr.Textbox(show_label=False)
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear = gr.Button("Clear Chat")
    clear.click(clear_chat,outputs=[msg,chatbot])
page.launch(share=True)



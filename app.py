import openai
import logging
import nltkmodules
import gradio as gr

from langchain import OpenAI
from llama_index.readers import Document
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from urllib import request
from urllib.error import HTTPError
from bs4 import BeautifulSoup as bs
from nltk import word_tokenize


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

models_list = ['text-ada-001', 'text-curie-001', 'text-babbage-001']
global model, index, temperature


def get_url_data(url):
    global model
    global index
    global temperature
    try:
        page = request.urlopen(url).read()
        content = bs(page).get_text()
        content_tokenized = ' '.join(word_tokenize(content))
        logger.info("Page read..")
    except HTTPError as err:
        logger.error("Invalid URL")

    max_input_size = 4096
    num_outputs = 256
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_outputs,
                                 max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    logger.info(f"model found :{model}")
    logger.info(f"api key :{openai.api_key}")
    llm_predictor = LLMPredictor(llm=OpenAI(
        openai_api_key=openai.api_key, temperature=temperature, model_name=model, max_tokens=num_outputs))

    index = GPTSimpleVectorIndex([Document(
        content_tokenized)], llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    return f"index created for the article: \"{url}\""


def predict(history, query):
    global index
    history = history or []
    print("index found ", index)
    result = index.query(query, response_mode="compact")
    logger.info(result.response)
    history = history + [(query, result.response)]
    return history


def set_model(sel_model):
    global model
    model = sel_model
    return "Chosen model: " + model


def set_temperature(sel_temperature):
    global temperature
    temperature = sel_temperature
    return None


demo = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")
with demo:
    with gr.Tab(label='Chatbot'):
        with gr.Row():
            with gr.Column(scale=0.85):
                url = gr.Textbox(show_label=False, placeholder="Enter the URL here... ").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                send_url = gr.Button('Load')
        idx_display = gr.Textbox(show_label=False).style(
            container=False, border=True)
        url.submit(get_url_data, inputs=[url], outputs=[idx_display])
        send_url.click(get_url_data, inputs=[url], outputs=[idx_display])

        chatbot = gr.Chatbot(elem_id="chatbot").style(height=300)
        with gr.Row():
            with gr.Column(scale=0.85):
                msg = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                send_chat = gr.Button('Send')
        state = gr.State()
        msg.submit(predict, [chatbot, msg], chatbot)
        msg.submit(lambda: "", None, msg)
        send_chat.click(predict, [chatbot, msg], chatbot)

    with gr.Tab(label='Settings'):
        sel_model = gr.Dropdown(label='Select the Model',
                                choices=models_list, value=models_list[0])
        model_status = gr.Markdown("No Model selected")
        sel_temperature = gr.Slider(0, 1, value=0, step=0.1, label='Set the temperature')
        sel_model.change(set_model, sel_model, model_status)
        sel_temperature.change(set_temperature, sel_temperature, None)

demo.launch()

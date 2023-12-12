import logging

from dotenv import load_dotenv

from api import app
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from pprint import pprint

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("SECRET_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-16s %(levelname)-8s %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def main():
    STEP1_TEMPLATE_ABSTRACT = "datas/template_abstract.txt"
    STEP2_TEMPLATE_RESULT = "datas/template_result.txt"

    def document():
        f = open("datas/project_data_카카오싱크.txt", 'r', encoding='utf-8')
        doc = []
        while True:
            line = f.readline()
            if not line: break
            doc.append(line)
        f.close()
        return doc

    def read_prompt_template(file_path: str) -> str:
        with open(file_path, "r") as f:
            prompt_template = f.read()

        return prompt_template

    def create_chain(llm, template_path, output_key):
        return LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template(
                template=read_prompt_template(template_path),
            ),
            output_key=output_key,
            verbose=True,
        )

    def generate_response(query):
        chatbot_llm = ChatOpenAI(temperature=0.1, max_tokens=1024, model="gpt-3.5-turbo-16k")

        abstract_chain = create_chain(chatbot_llm, STEP1_TEMPLATE_ABSTRACT, "abstract")
        result_chain = create_chain(chatbot_llm, STEP2_TEMPLATE_RESULT, "result")

        preprocess_chain = SequentialChain(
            chains=[
                abstract_chain,
                result_chain
            ],
            input_variables=["guide", "query"],
            output_variables=["abstract", "result"],
            verbose=True,
        )

        context = dict(
            guide=document(),
            query=query
        )

        context = preprocess_chain(context)

        print(context["result"])
        return context

    generate_response("과정을 알려주라")


if __name__ == "__main__":
    main()

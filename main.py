import logging

from dotenv import load_dotenv

from api import app
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("SECRET_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-16s %(levelname)-8s %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def main():
    PROJECT_DATA_SYNC = "datas/project_data_카카오싱크.txt"
    PROJECT_DATA_CHANNEL = "datas/project_data_카카오톡채널.txt"
    PROJECT_DATA_SOCIAL = "datas/project_data_카카오소셜.txt"

    STEP1_TEMPLATE_ABSTRACT = "datas/template_abstract.txt"
    STEP2_TEMPLATE_RESULT = "datas/template_result.txt"

    INTENT_PROMPT_TEMPLATE = "datas/template_intent.txt"
    INTENT_LIST_TXT = "datas/intent_list.txt"

    CHROMA_PERSIST_DIR = os.path.join("upload/chroma-persist")
    CHROMA_COLLECTION_NAME = "dosu-bot"

    # _db = Chroma(
    #     persist_directory=CHROMA_PERSIST_DIR,
    #     embedding_function=OpenAIEmbeddings(),
    #     collection_name=CHROMA_COLLECTION_NAME,
    # )
    # _retriever = _db.as_retriever()
    #
    # def query_db(query: str, use_retriever: bool = False) -> list[str]:
    #     if use_retriever:
    #         docs = _retriever.get_relevant_documents(query)
    #     else:
    #         docs = _db.similarity_search(query)
    #
    #     str_docs = [doc.page_content for doc in docs]
    #     return str_docs

    def document(path):
        f = open(path, 'r', encoding='utf-8')
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
        chatbot_llm = ChatOpenAI(temperature=0.1, max_tokens=512, model="gpt-3.5-turbo-16k")

        abstract_chain = create_chain(chatbot_llm, STEP1_TEMPLATE_ABSTRACT, "abstract")
        result_chain = create_chain(chatbot_llm, STEP2_TEMPLATE_RESULT, "result")
        parse_intent_chain = create_chain(chatbot_llm, INTENT_PROMPT_TEMPLATE, "result")

        context = dict(query=query)
        context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)

        intent = parse_intent_chain(context)["result"]

        preprocess_chain = SequentialChain(
            chains=[
                abstract_chain,
                result_chain
            ],
            input_variables=["guide", "query"],
            output_variables=["abstract", "result"],
            verbose=True,
        )

        if intent == "카카오싱크":
            context["guide"] = document(PROJECT_DATA_SYNC)
        elif intent == "카카오톡채널":
            context["guide"] = document(PROJECT_DATA_CHANNEL)
        elif intent == "카카오소셜":
            context["guide"] = document(PROJECT_DATA_SOCIAL)
        # print(intent)
        answer = preprocess_chain(context)

        print(answer["result"])
        return answer["result"]

    generate_response("카카오톡 채널 기능 소개를 해달라")


if __name__ == "__main__":
    main()

import argparse
from typing import List, Sequence
from urllib.parse import urlparse

from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    ArxivLoader,
    AsyncChromiumLoader,
    YoutubeLoader,
    HNLoader,
)
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI

from loguru import logger


class AsyncChromiumHtmlLoader(AsyncChromiumLoader):
    def load(self) -> List[Document]:
        docs = super().load()
        docs = list(Html2TextTransformer().transform_documents(docs))
        return docs


def get_docs(url: str) -> Sequence[Document]:
    parsed_url = urlparse(url)
    match (parsed_url.netloc, parsed_url.path):
        case "", _:
            raise ValueError(f"Not a valid url: {url}")
        case "www.youtube.com" | "youtube.com" | "youtu.be", _:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        case "news.ycombinator.com", _:
            loader = HNLoader(url)
        case "arxiv.org", arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]
            loader = ArxivLoader(arxiv_id)
        case _, path if path.endswith(".pdf"):
            loader = PyMuPDFLoader(url)
        case _, _:
            loader = AsyncChromiumHtmlLoader([url])

    logger.debug(f"Using loader {loader.__class__.__name__} for URL {url}")
    docs = loader.load()
    logger.debug(f"Loaded {len(docs)} docs")
    return docs


def summarize(
    docs: Sequence[Document],
    verbose: bool = False,
    model: str = "llama3",
):
    match model:
        case "groq":
            llm = ChatGroq(model_name="llama3-8b-8192")
        case "gemini":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
        case _:
            llm = Ollama(
                model=model,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful summarization assistant. Please provide a markdown format {verbosity} combined summary of the following summaries without including any references or external links. The summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information. Remember, DO NOT include any external references or metadata!

    {content}
    """)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    verbosity = "detailed" if verbose else "concise"
    return chain.invoke({"content": docs, "verbosity": verbosity})


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="URL to summarize")
    parser.add_argument(
        "--model",
        "-m",
        default="llama3",
        help="Choose model to use: groq, gemini or any ollama model",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbosity",
        action="store_true",
        help="Output detailed summary",
    )
    parser.add_argument("--debug", dest="debug")
    args = parser.parse_args()
    docs = get_docs(url=args.url)

    summarize(docs=docs, verbose=args.verbosity, model=args.model)

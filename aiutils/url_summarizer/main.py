import sys
import argparse
from typing import List, Sequence
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    AsyncChromiumLoader,
    YoutubeLoader,
    HNLoader,
)
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI

from loguru import logger


MODELS = {
    "groq": "mixtral-8x7b-32768",
    "gemini": "gemini-1.5-flash-latest",
}


class AsyncChromiumHtmlLoader(AsyncChromiumLoader):
    def load(self) -> List[Document]:
        docs = super().load()
        docs = list(Html2TextTransformer().transform_documents(docs))
        return docs


class JinaReaderLoader(BaseLoader):
    def __init__(self, url: str):
        self.url = url

    def load(self):
        response = httpx.get(f"https://r.jina.ai/{self.url}")
        response.raise_for_status()
        return [
            Document(
                page_content=response.text,
                metadata={"status_code": response.status_code},
            )
        ]


class WebLoader(BaseLoader):
    def __init__(self, url: str) -> None:
        self.url = url

    def load(self) -> List[Document]:
        try:
            return JinaReaderLoader(self.url).load()
        except httpx.HTTPError as e:
            logger.error(
                f"Failed to fetch content from Jina for URL {self.url}: {e}. Falling back to AsyncChromiumHtmlLoader."
            )
            return AsyncChromiumHtmlLoader([self.url]).load()


def get_docs(url: str, is_pdf_url: bool = False) -> Sequence[Document]:
    parsed_url = urlparse(url)
    match (parsed_url.netloc, parsed_url.path):
        case "", _:
            raise ValueError(f"Not a valid url: {url}")
        case "www.youtube.com" | "youtube.com" | "youtu.be", _:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        case "news.ycombinator.com", _:
            loader = HNLoader(url)
        case _, path if path.endswith(".pdf") or is_pdf_url:
            loader = PyPDFLoader(url)
        case _, _:
            loader = WebLoader(url)

    logger.debug(f"Using loader {loader.__class__.__name__} for URL {url}")
    docs = loader.load()
    logger.debug(f"Loaded {len(docs)} docs")
    return docs


def summarize(
    docs: Sequence[Document],
    verbose: bool = False,
    model: str = "gemma2",
):
    match model:
        case "groq":
            llm = ChatGroq(model_name=MODELS["groq"])
        case "gemini":
            llm = ChatGoogleGenerativeAI(model=MODELS["gemini"])
        case _:
            llm = Ollama(model=model)
    logger.debug(f"Using model {llm.__class__.__name__} / {MODELS.get(model, model)}")

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful summarization assistant. Please provide a markdown format {verbosity} summary of the following content without including any references or external links. The summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information. Remember, DO NOT include any external references or metadata!

    {content}
    """)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    verbosity = "detailed" if verbose else "concise"
    for token in chain.stream({"content": docs, "verbosity": verbosity}):
        sys.stdout.write(token)
        sys.stdout.flush()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="URL to summarize")
    parser.add_argument("--pdf", action="store_true", help="Force use of PDF loader")
    parser.add_argument(
        "--model",
        "-m",
        default="gemma2",
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
    docs = get_docs(url=args.url, is_pdf_url=args.pdf)

    summarize(docs=docs, verbose=args.verbosity, model=args.model)

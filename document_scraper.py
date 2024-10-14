from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.vectorstores.base import VectorStoreRetriever


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import json
import pydantic
from typing import Literal, Union, Any, Optional, Type

from abc import ABC, abstractmethod

# NOTE IMPORTANT NOTE
# This approach will pass specific segments of html to an LLM to extract necessary data from it

# TODO
"""
> Meraki web pages are slightly different for get and post requests
    Write code to handle this difference as its needed for request body 

"""


"""

#######################################################
                        CONTENTS                      #
#######################################################
#                                                     #
# System Requirements                                 #
#                                                     #
#######################################################
"""

#######################################################
#                 System Requirements                 #
#######################################################
"""
System Requirements
    This module requires a chrome webdriver to be installed on the machine
    to be used by selenium.

    The local LLM models used in this module are 
    "llama3.1:latest" and "mistral-nemo:latest"
"""

class WebDocumentGenerator(ABC):

    EMBEDDINGS_LLM = "llama3.1:latest"
    DATA_EXTRACTION_LLM = "mistral-nemo:latest"
    CONVERSATION_LLM = "llama3.1:latest"

    def __init__(self):
        super().__init__()
        self._web_driver: webdriver.Chrome = None
        self._vectorstore: FAISS = None
    
        self._init_webdriver()

    #######################################################
    #                   Concrete Methods                  #
    #######################################################
    
    def _init_webdriver(self) -> None:
        """Initialises the web driver used for web scrapping"""

        # define chrome driver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")

        # define web driver
        self._web_driver = webdriver.Chrome(options=chrome_options)
    
    def _load_vectorstore(self, file_path: str) -> None:
        """Loads a vectorstore from a locally saved file"""
        self.vectorstore.save_local(file_path)

    def _save_vectorstore(self, file_path: str) -> None:
        """Exports the currently loaded vectorstore to a file."""
        embeddings = OllamaEmbeddings(WebDocumentGenerator.EMBEDDINGS_LLM)
        self._vectorstore = FAISS.load_local(file_path, embeddings)
    
    def _check_webdriver_init(self) -> None:
        if not self._web_driver.title:
            raise Exception("No web page has been loaded by the web driver.")
    
    def initialise_vectorstore(self, mode = Literal["web scrape", "file"], **kwargs) -> None:
        """Initialises the vectorstore."""
        self._vectorstore = None
        if mode == "web scrape":
            self._create_vectorstore(kwargs["urls"])
        elif mode == "file":
            self._load_vectorstore(kwargs["file_path"])
        else:
            raise ValueError(
                f"Error 'mode' can only be 'web scrape' or 'file'. The value {mode} was provided."
            )

    #######################################################
    #                   Abstract Methods                  #
    #######################################################
    
    @abstractmethod
    def _create_vectorstore(urls: Any) -> None:
        """Creates a vectorstore by creating documents via web scrapping."""
        pass

    #######################################################
    #                     Properties                      #
    #######################################################

    @property
    def vectorstore(self) -> FAISS:
        if self._vectorstore is None:
            raise TypeError("Error vector store has not been initialized")
        return self._vectorstore
    
    @property
    def retriever(self, k: int = 2) -> VectorStoreRetriever:
        return self.vectorstore.as_retriever(k=2)

class MerakiDocumentScrapper(WebDocumentGenerator):

    class API_DATA(pydantic.BaseModel):
        api_name: str
        description: str
        endpoint: str
        request_type: Literal["get", "post"]
        request_body_example: dict
        request_body_schema: dict
        response_example: dict
        response_schema: dict

    def __init__(self):
        super().__init__()
        self._URLS = [
            ("arp_dump_get", "https://developer.cisco.com/meraki/api-v1/get-device-live-tools-arp-table/"),
            ("arp_dump_create", "https://developer.cisco.com/meraki/api-v1/create-device-live-tools-arp-table/"),
            ("cable_test_get", "https://developer.cisco.com/meraki/api-v1/get-device-live-tools-cable-test/"),
            ("cable_test_create", "https://developer.cisco.com/meraki/api-v1/create-device-live-tools-cable-test/"),
            ("ping_test_get", "https://developer.cisco.com/meraki/api-v1/get-device-live-tools-ping-device/"),
            ("ping_test_create", "https://developer.cisco.com/meraki/api-v1/create-device-live-tools-ping-device/"),
            ("throughput_test_get", "https://developer.cisco.com/meraki/api-v1/create-device-live-tools-throughput-test/"),
            ("throughput_test_create", "https://developer.cisco.com/meraki/api-v1/get-device-live-tools-throughput-test/")
        ]
    
    def _create_vectorstore(self):
        self._get_api_info_from_webpage(self._URLS[1][1])

    def _get_api_info_from_webpage(self, api_url) -> API_DATA:
        """Extracts information about an api from a webpage."""

        self._web_driver.get(api_url)
        request_body_example = self._get_post_request_body_example()
        print(request_body_example)
        exit()

        api_name, description = self._get_api_name_and_description()
        endpoint_url, request_type = self._get_endpoint_and_request_type()
        request_body_schema = self._get_post_request_body_schema()

        print(api_name, description)
        print(endpoint_url, request_type)
        print(request_body_schema)
    
    def _get_api_name_and_description(self) -> tuple[str, str]:
        """Returns the operation ID and description of an api"""
        
        html_segment = self._extract_html_from_shadow_root(
            css_selector="div.swagger-operation-top",
            extract_type="outerHTML"
        )

        chat_prompt_template = ChatPromptTemplate([
            SystemMessage("You are a tool that is being used by another LLM. Your job is to extract the operation ID and description of the api from the html code."),
            SystemMessage("Return the operation ID and description exactly how it is written."),
        ])

        class DATA(pydantic.BaseModel):
            operation_id: str = pydantic.Field(description="A unique name of the api referred to as the operation id.")
            description: str = pydantic.Field(description="A description about the api.")
        
        extracted_info: DATA = self._use_llm_to_extract_info(
            chat_prompt_template=chat_prompt_template,
            html=html_segment,
            pydantic_output_model=DATA
        )

        return extracted_info.operation_id, extracted_info.description
    
    def _get_endpoint_and_request_type(self) -> tuple[str, str]:
        """Returns the endpoint url and request type of the api."""
        
        html_segment = self._extract_html_from_shadow_root(
            css_selector="div.swagger-operation-main div.path",
            extract_type="outerHTML"
        )

        chat_prompt_template = ChatPromptTemplate([
            SystemMessage("You are a tool that is being used by another LLM. Your job is to extract the endpoint url and request method of the api from the html code."),
            SystemMessage("Do not include html tags in your output.")
        ])

        class DATA(pydantic.BaseModel):
            endpoint_url: str = pydantic.Field(description="A url endpoint used to call the api.")
            request_method: Literal["get", "post"] = pydantic.Field(description="The request method needed to call the api.")
        
        extracted_info: DATA = self._use_llm_to_extract_info(
            chat_prompt_template=chat_prompt_template,
            html=html_segment,
            pydantic_output_model=DATA
        )

        return extracted_info.endpoint_url, extracted_info.request_method

    # TODO A check needs to be made to ensure that it is only used on post apis
    def _get_post_request_body_schema(self) -> dict:
        """Returns the schema of the api request for post request apis."""
        
        html_segment = self._extract_html_from_shadow_root(
            css_selector="div.swagger-operation-parameters div.tabs__content ul.schema-tree",
            extract_type="outerHTML"
        )

        chat_prompt_template = ChatPromptTemplate([
            SystemMessage("You are a tool that is being used by another LLM. Your job is to extract the schema of the api request."),
            SystemMessage("Your output should be json formatted where the key is the name of the parameter and the value is a nested dictionary holding the data type of the parameter and a description of it."),
            SystemMessage("For nested parameters the parameter name should be in the format outer_parameter.nested_parameter"),
            SystemMessage("Do not include html tags in your output.")
        ])

        class PARAMETER_DATA(pydantic.BaseModel):
            data_type: str
            description: str

        class DATA(pydantic.BaseModel):
            parameters: dict[str, PARAMETER_DATA] = pydantic.Field(description="A dictionary representation of the api's request schema.")
        
        extracted_info: DATA = self._use_llm_to_extract_info(
            chat_prompt_template=chat_prompt_template,
            html=html_segment,
            pydantic_output_model=DATA
        )
        
        return extracted_info.model_dump()["parameters"]

    # TODO Develop This
    # TODO A check needs to be made to ensure that it is only used on post apis
    def _get_post_request_body_example(self) -> dict:
        """Returns an example of the api request for post request apis."""
        
        html_segment = self._extract_html_from_shadow_root(
            css_selector="div.swagger-operation-parameters div.tabs__content pre.language-json.code-area",
            extract_type="outerHTML"
        )

        chat_prompt_template = ChatPromptTemplate([
            SystemMessage("You are a tool that is being used by another LLM. Your job is to extract the example of the api request body."),
            SystemMessage("Your output should be an exact copy of the example on the webpage"),
            SystemMessage("Do not include html tags in your output.")
        ])

        class DATA(pydantic.BaseModel):
            parameters: dict = pydantic.Field(description="A dictionary representation of the api's request body.")
        
        extracted_info: DATA = self._use_llm_to_extract_info(
            chat_prompt_template=chat_prompt_template,
            html=html_segment,
            pydantic_output_model=DATA
        )
        return extracted_info.model_dump()["parameters"]
        
    def _extract_html_from_shadow_root(self, css_selector: str, extract_type: Literal["innerHTML", "outerHTML"]) -> str:
        """Extracts a specific component from within a shadowRoot component"""
        self._check_webdriver_init()

        shadow_host = self._web_driver.find_element(By.CSS_SELECTOR, "dui-swagger-api-v3")
        shadow_root = self._web_driver.execute_script("return arguments[0].shadowRoot", shadow_host)
        element_inside_show_dom = shadow_root.find_element(By.CSS_SELECTOR, css_selector)
        extracted_html = self._web_driver.execute_script(f"return arguments[0].{extract_type}", element_inside_show_dom)

        return extracted_html

    def _use_llm_to_extract_info(
            self,
            chat_prompt_template: ChatPromptTemplate,
            html: str,
            pydantic_output_model: pydantic.BaseModel
        ) -> Type[pydantic.BaseModel]:
        """Uses a LLM to extract specific information specified in the chat prompt template.
        NOTE: The chat_prompt_template passed should only have system messages.
        """
        extractor_llm = OllamaLLM(model=WebDocumentGenerator.DATA_EXTRACTION_LLM)
        extractor_output_parser = PydanticOutputParser(pydantic_object=pydantic_output_model)
        
        chat_prompt_template.append(SystemMessage(extractor_output_parser.get_format_instructions()))
        chat_prompt_template.append(("human", "{html}"))

        
        extractor_chain = chat_prompt_template | extractor_llm | extractor_output_parser

        return extractor_chain.invoke({
            "html": html
        })
    
if __name__ == "__main__":
    m = MerakiDocumentScrapper()
    m._create_vectorstore()
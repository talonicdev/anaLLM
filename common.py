import sys
import json
import inspect
import re
import base64
import os
import ast
from pathlib import Path
import requests
import datetime
from pandas import DataFrame
from pandasai.llm import OpenAI
from pandasai.smart_datalake import SmartDatalake
from pandasai.smart_dataframe import SmartDataframe
from pandasai.helpers.openai_info import get_openai_callback, OpenAICallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, BasePromptTemplate
from typing import Union, Optional, List
from enum import Enum
from config import Config

tokens = {
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_cost': 0
}

class WriteType(Enum):
    # Exotic main types that are necessary for the operation and *always* expected by the script runner
    TOKENS = "tokens" # Token count used for a given call, { prompt_tokens, completion_tokens, total_cost }
    MILESTONE = "milestone" # Major status progress, used later to provide real-time progress updates to customers
    RESULT = "result" # The expected result of the script process (i.e. table, charts, text message, acknowledgement, ..)
    # Default log levels, output dependent on log level
    DEBUG = "debug" # Most commonly used to track general status and nominal events during development
    WARN = "warn" # Suggests a potential issue or fault, but doesn't require immediate attention
    ERROR = "error" # Indicates a major issue or fault and probable inability to process a request, needs attention soon
    FATAL = "fatal" # The script can't progress or fail gracefully and has to forcefully shutdown, investigate asap 
    TRACE = "trace" # Most verbose and only used to get extremely detailed information about every step taken
    REFERENCES = "references" # List of sheets (besides the calling sheet) that were used to process a request
    

class Common:
    def __init__(self, config: Optional[Config]=None):
        self.config = config if config else Config()
        self.tokens = tokens
        self.path = Path(__file__).parent.resolve()
        
    # Get the total amount of tokens/prices
    def get_tokens(self):
        return self.tokens
    
    def _get_type(self,data,r_type=None):
        t = type(data).__name__
        if r_type == 'plot':
            return 'image'
        if t == 'str':
            return 'string'
        if t == 'bool':
            return 'boolean'
        if t == 'int':
            return 'number'
        if t in ['DataFrame','SmartDataframe']:
            return 'sheet'
        if t == 'list':
            return 'array'
        if t == 'dict':
            return 'object'
        return t

    # Write any message or error to stdout
    def write(self,
                  messageType:WriteType,
                  message:Union[str,dict,list,int,bool,DataFrame,SmartDataframe],
                  context:Optional[Union[str,bool]] = None,
                  error:Optional[Union[str,Exception]] = None
                  ):
        
        data = None
        content = None
        result_type = None
        
        # Get UNIX timestamp in ms
        now = int(datetime.datetime.now().timestamp()) * 1000
        
        
        # Results are a dict containing the result content as well as its type (dataframe, string, number, plot)
        if messageType == WriteType.RESULT:
            if isinstance(message,dict):
                content = message['data']
                result_type = str(message['type'])
            else:
                content = message
            if result_type == "plot":
                # Result is a plot => read, validate, b64-encode and delete
                content = str(content)
                out_dir = str(self.path / 'output_plot')
                if content.startswith(out_dir):
                    try:
                        with open(content, 'rb') as image_file:
                            b64img = base64.b64encode(image_file.read()).decode('utf-8')
                            os.remove(content)
                            content = b64img
                    except Exception as e:
                        self.write(WriteType.ERROR,'Could not write Plot',True,e)
                        return
                else:
                    self.write(WriteType.ERROR,f'Invalid Plot File Path: {content}',True)
                    return
            else:
                # Result is string, number or (Smart)DataFrame
                if isinstance(content,str):
                    if str(self.path) in content:
                        # Data contains directory path -> omit result
                        self.write(WriteType.WARN,f"Message contains file path: '{content}'",True)
                        return
        else:
            content = message
                
        if isinstance(content,(str,dict,list,int,bool)):
            # message is a simple type, can stringify as is
            data = content
        elif isinstance(content,DataFrame):
            data = json.loads(content.to_json(path_or_buf=None))
        elif isinstance(content,(SmartDataframe)):
            # Message is a Pandas DataFrame or extended PandasAI SmartDataframe, 
            # invoke to_json without path to return JSON string, then parse it just to stringify it again later
            # data = json.loads(content.to_json(None))
            # NOTE: There's currently an incompatibility: Pandas' df expects a `path_or_buf` kwarg, but PandasAI passes it as `path`, leading to a runtime error
            data = json.loads(content.dataframe.to_json(path_or_buf=None))
        elif isinstance(content,set):
            data = list(content)
        else:
            if content:
                # Try to stringify it anyways
                try:
                    data = str(content)
                except:
                    data = ''
                    pass
            
        if isinstance(context,str):
            # Context is whatever the string says it is
            ctx = context
        else:
            # If context is `True`, the function calling `write` is just passing through, so we take the *second* entry in the stack
            index = 2 if context else 1
            # Context was not passed, so we deduce it from the stack
            frame = inspect.stack()[index].frame
            if 'self' in frame.f_locals:
                # stack item frame has a 'self', so we can assume it's a class
                # ctx is now `<ClassName>.<FunctionName>`
                ctx = f"{frame.f_locals['self'].__class__.__name__}.{frame.f_code.co_name}"
            else:
                # No 'self' = no class
                # ctx is `<FunctionName>`
                ctx = frame.f_code.co_name
        
        log_entry = {'type': messageType.value, 'context': ctx, 'datatype': self._get_type(content,result_type), 'timestamp': now }

        if data is not None:
            log_entry['data'] = data
        if error:
            log_entry['error'] = str(error)
            
        if result_type is not None:
            log_entry['resulttype'] = result_type

        try:
            sys.stdout.write(f"{json.dumps(log_entry)}\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Failed to log message: {e}\n")
            sys.stderr.flush()
     
    # Internal function to handle OpenAICallback, used for token counting
    def _handle_callback(self,
                        context:Optional[str],
                        cb:OpenAICallbackHandler):
        if cb and cb.prompt_tokens:
            self.tokens['prompt_tokens'] += cb.prompt_tokens
            self.tokens['completion_tokens'] += cb.completion_tokens
            self.tokens['total_cost'] += cb.total_cost
            self.write(WriteType.TOKENS,{'prompt':cb.prompt_tokens,'completion':cb.completion_tokens,'cost':cb.total_cost},context)
        else:
            #self.write(WriteType.ERROR,f'No openai_callback: {cb}',context)
            pass
            
    def chat_agent(self,
                   agent:SmartDatalake,
                   *args):
        with get_openai_callback() as cb:
            try:
                result = agent.chat(*args)
            except Exception as e:
                self.write(WriteType.ERROR,f"Agent Chat error : {agent.last_error}",True,e)
                
            self._handle_callback(True,cb)
            return result
        
    def invoke_chatOpenAI(self,
                   llm:ChatOpenAI,
                   *args):
        
        # No callback here :(
        with get_openai_callback() as cb:
            try:
                result = llm(*args)
                # TODO: Find out why callback doesn't work here
                self.write(WriteType.DEBUG,result,True)
                self._handle_callback(True,cb)
                return result
            except Exception as e:
                self.write(WriteType.ERROR,'LLM Invocation error',True,e)
                raise e
        
    def invoke_openAI(self,
                   llm:OpenAI,
                   *args):
        with get_openai_callback() as cb:
            try:
                result = llm(*args)
            except Exception as e:
                self.write(WriteType.ERROR,'LLM Invocation error',True,e)
                
            self._handle_callback(True,cb)
            return result
        
    def invoke_chain(self,
                     chain:Union[BasePromptTemplate,ChatPromptTemplate,FewShotChatMessagePromptTemplate,ChatOpenAI],
                     *args, **kwargs):
        with get_openai_callback() as cb:
            try:
                result = chain.invoke( *args, **kwargs)
            except Exception as e:
                self.write(WriteType.ERROR,'Chain Invocation error',True,e)
            self._handle_callback(True,cb)
            return result
        
    def get_openAI_llm(self,**kwargs):
        return OpenAI(
            api_token = self.config.OPENAI_API_KEY,
            model = self.config.MODEL_BIG,
            max_tokens = self.config.MAX_TOKENS,
            **kwargs
        )
        
    def get_chatOpenAI_llm(self,**kwargs):
        return ChatOpenAI(
            openai_api_key = self.config.OPENAI_API_KEY, 
            model_name = self.config.MODEL_BIG, 
            max_tokens = self.config.MAX_TOKENS, 
            request_timeout = self.config.REQUEST_TIMEOUT,
            **kwargs
        )
        
    def to_flat_list(self, input) -> List[str]:
        result = []
        if isinstance(input, str):
            try:
                # Attempt to evaluate the string as a Python literal
                evaluated_input = ast.literal_eval(input.strip())
                # If successful and a list, recursively process
                if isinstance(evaluated_input, list):
                    return self.to_flat_list(evaluated_input)
            except (ValueError, SyntaxError):
                # If not a valid Python literal or list, split by commas
                return [item.strip() for item in input.split(',')]
        
        elif isinstance(input, list):
            for item in input:
                # Recursively flatten if we encounter another list
                if isinstance(item, list):
                    result.extend(self.to_flat_list(item))
                else:
                    result.append(str(item).strip())
            return result
        else:
            raise ValueError("Input must be a string or a list")

        return result
    
        
    def pd_ai_is_expected_type(self, expected_type:str, response):
        match expected_type:
            case "dataframe":
                return isinstance(response,(DataFrame,SmartDataframe))
            case "string":
                return isinstance(response,str)
            case "plot":
                return isinstance(response,str) and bool(re.match(r"^(\/[\w.-]+)+(/[\w.-]+)*$|^[^\s/]+(/[\w.-]+)*$", response))
            case "number":
                return isinstance(response,(int,float))
            case _:
                return False
     
class Requests:
    def __init__(self, 
        config: Optional[Config]=None, 
        token: Optional[str]=None, 
        api_key: Optional[str]=None
    ):
        
        self.config = config if config else Config()
        self.api_key = api_key if api_key else self.config.SERVICE_API_KEY
        self.token = token if token else self.config.TOKEN
        
        self.session = requests.Session()  # Session for connection pooling
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}',
            'x-api-key': f'{self.api_key}'
            })

    def _make_request(self, method, endpoint, **kwargs):
        url = f'{self.config.SERVICE_API_URL}/{endpoint}'
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def get(self, endpoint, **kwargs):
        return self._make_request('get', endpoint, **kwargs)

    def post(self, endpoint, **kwargs):
        return self._make_request('post', endpoint, **kwargs)

    def put(self, endpoint, **kwargs):
        return self._make_request('put', endpoint, **kwargs)

    def patch(self, endpoint, **kwargs):
        return self._make_request('patch', endpoint, **kwargs)

    def delete(self, endpoint, **kwargs):
        return self._make_request('delete', endpoint, **kwargs)
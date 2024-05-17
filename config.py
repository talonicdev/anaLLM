import os
from typing import Optional
from decouple import config

def _parse_value(value, var_type):
    if var_type == bool:
        return value.lower() in ['true', '1', 't', 'y', 'yes']
    try:
        return var_type(value)
    except ValueError:
        return None

class __SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            cls._resolve_config(instance)  # Resolve config values upon creation
        return cls._instances[cls]

    def _resolve_config(cls, instance):
        for key, (default_value, var_type) in instance.DEFAULTS.items():
            # Only update the value if it's still the default (or None for optional values not in DEFAULTS)
            current_value = getattr(instance, key, default_value)
            if current_value == default_value or current_value is None:
                env_value = os.getenv(key, default_value)
                setattr(instance, key, _parse_value(env_value, var_type))

class __BaseConfig(metaclass=__SingletonMeta):
    DEFAULTS = {
        'TOKEN': (config('USER', default=''), str),
        'OPENAI_API_KEY': (config('KEY', default=''), str),
        'SERVICE_API_URL': (config('SERVICE_API_URL', default='https://backend.vhi.ai/service-api'), str),
        'SERVICE_API_KEY': (config('API_KEY', default=''), str),
        'MODEL_BIG': (config('MODEL_BIG', default='gpt-4-1106-preview'), str),
        'MODEL_SMALL': (config('MODEL_SMALL', default='gpt-3.5-turbo-instruct'), str),
        'COLLECTION_NAME': (config('COLLECTION_NAME', default='talonic_collection'), str),
        'MAX_TOKENS': (config('MAX_TOKENS', default=3000, cast=int), int),
        'REQUEST_TIMEOUT': (config('REQUEST_TIMEOUT', default=30000, cast=int), int),
        'MAX_RETRIES': (config('MAX_RETRIES', default=1, cast=int), int),
        'ERROR_CORRECTION_ATTEMPTS': (config('ERROR_CORRECTION_ATTEMPTS', default=1, cast=int), int),
        'LOG_LEVEL': (config('LOG_LEVEL', default='DEBUG'), str),
        'SHEET_ID': (config('SHEET_ID', default=''), str)
    }
    TOKEN:str = DEFAULTS['TOKEN'][0]
    OPENAI_API_KEY:str = DEFAULTS['OPENAI_API_KEY'][0]
    SERVICE_API_URL:str = DEFAULTS['SERVICE_API_URL'][0]
    SERVICE_API_KEY:str = DEFAULTS['SERVICE_API_KEY'][0]
    MODEL_BIG:str = DEFAULTS['MODEL_BIG'][0]
    MODEL_SMALL:str = DEFAULTS['MODEL_SMALL'][0]
    COLLECTION_NAME:str = DEFAULTS['COLLECTION_NAME'][0]
    MAX_TOKENS:int = DEFAULTS['MAX_TOKENS'][0]
    REQUEST_TIMEOUT:int = DEFAULTS['REQUEST_TIMEOUT'][0]
    MAX_RETRIES:int = DEFAULTS['MAX_RETRIES'][0]
    ERROR_CORRECTION_ATTEMPTS:int = DEFAULTS['ERROR_CORRECTION_ATTEMPTS'][0]
    LOG_LEVEL:str = DEFAULTS['LOG_LEVEL'][0]
    SHEET_ID:str = DEFAULTS['SHEET_ID'][0]
    
    def set_environment_variables(self):
        """Sets relevant environment variables based on config values."""
        os.environ['API_USER'] = self.TOKEN
        os.environ['OPENAI_API_KEY'] = self.OPENAI_API_KEY
        os.environ['API_KEY'] = self.SERVICE_API_KEY
    
    @classmethod
    def get(cls, key):
        if key in cls.DEFAULTS:
            default_value, var_type = cls.DEFAULTS[key]
            env_value = os.getenv(key, default_value)
            return _parse_value(env_value, var_type)
        else:
            raise KeyError(f"Config for '{key}' not found.")

class Config(__BaseConfig):
    def __init__(self, 
                 token: Optional[str]=None,
                 api_key: Optional[str]=None,
                 openai_api_key: Optional[str]=None,
                 sheet_id: Optional[str]=None,
                 **overrides):
        super().__init__()

        if token is not None:
            self.TOKEN = _parse_value(token,self.DEFAULTS['TOKEN'][1])
        if api_key is not None:
            self.SERVICE_API_KEY = _parse_value(api_key,self.DEFAULTS['SERVICE_API_KEY'][1])
        if openai_api_key is not None:
            self.OPENAI_API_KEY = _parse_value(openai_api_key,self.DEFAULTS['OPENAI_API_KEY'][1])
        if sheet_id is not None:
            self.SHEET_ID = _parse_value(sheet_id,self.DEFAULTS['SHEET_ID'][1])
            
        self.apply_overrides(overrides)

    def apply_overrides(self, overrides):
        for key, value in overrides.items():
            if key in self.DEFAULTS:  # Ensure only valid config keys are overridden
                setattr(self, key, _parse_value(value,self.DEFAULTS[key][1]))
        self.set_environment_variables()
import argparse

import pyperclip
import toml
import itertools
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings


class ModelManager:
    def __init__(self, config_path='model_providers.toml', load_test=False, libinfo=False):
        load_dotenv(override=True)
        self.load_test = load_test
        self.libinfo = libinfo
        self.config = toml.load(config_path)

        self.model_type = ''
        self.provider = ''
        self.model_name = ''
        self.provider_config = {}
        self.base_url = ''
        self.llm = None
        self.embedding = None
        self.dimensions = 0
        self._code = ''

    def _load_chat_model(self, model_name, temperature):
        if self.provider != 'azure':
            return ChatOpenAI(
                model=model_name,
                openai_api_base=self.base_url,
                openai_api_key=os.getenv(self.provider.upper() + '_API_KEY'),
                temperature=temperature
            )
        else:
            return AzureChatOpenAI(
                openai_api_key=os.getenv(self.provider.upper() + '_API_KEY'),
                azure_endpoint=self.base_url,
                azure_deployment=model_name,
                api_version=self.provider_config['api_version'],
                temperature=temperature
            )

    def _load_embedding_model(self, model_name):
        if self.provider == 'openai':
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_base=self.base_url,
                openai_api_key=os.getenv(self.provider.upper() + '_API_KEY')
            )
        elif self.provider == 'qwen':
            return DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=os.getenv(self.provider.upper() + '_API_KEY')
            )

    def unload(self):
        self.embedding = None
        self.llm = None
        self.model_type = ''
        self.provider = ''
        self.provider_config = {}
        self.model_name = ''
        self.base_url = ''

    def _load_model(self, model_name, temperature=0.1):
        if model_name in self.config['providers']['names']:
            model_name = self.config[model_name]['models'][0]
            return self._load_model(model_name, temperature)
        self.model_name = model_name
        for provider, details in itertools.islice(self.config.items(), 1, None):
            if model_name in details['models']:
                self.model_type = 'chat'
                self.provider = provider
                self.provider_config = self.config[provider]
                self.base_url = self.provider_config.get('base_url') or os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1')
                self.llm = self._load_chat_model(model_name, temperature)
                if self.provider != 'azure':
                    self.code = f"{self.provider}_llm=ChatOpenAI(model='{model_name}', openai_api_base='{self.base_url}', openai_api_key='{os.getenv(provider.upper() + '_API_KEY','api_key')}', temperature={temperature})"
                else:
                    self.code = f"{self.provider}_llm=AzureChatOpenAI(openai_api_key='{os.getenv(self.provider.upper() + '_API_KEY')}', azure_endpoint='{self.base_url}', azure_deployment='{model_name}', api_version='{self.provider_config['api_version']}', temperature={temperature})"
                return self.llm

            if model_name in details.get('embeddings', []):
                self.model_type = 'embedding'
                self.provider = provider
                self.provider_config = self.config[provider]
                self.base_url = self.provider_config.get('base_url') or os.getenv('OPENAI_BASE_URL',
                                                                              'https://api.openai.com/v1')
                self.embedding = self._load_embedding_model(model_name)
                if self.provider == 'openai':
                    self.code = f"{self.provider}_embedding=OpenAIEmbeddings(model='{model_name}', openai_api_base='{self.base_url}', openai_api_key='{os.getenv(provider.upper() + '_API_KEY')}')"
                elif self.provider == 'qwen':
                    self.code = f"{self.provider}_embedding=DashScopeEmbeddings(model='{model_name}', dashscope_api_key='{os.getenv(provider.upper() + '_API_KEY')}')"
                return self.embedding

        self.unload()

    def __str__(self):
        if any([self.llm, self.embedding]):
            info = f'{self.provider} | {self.model_type}: {self.model_name}'
            if self.dimensions:
                info += f' {self.dimensions} dimensions'
        else:
            info = 'Model not loaded'
        return info

    def __repr__(self):
        return self.__str__()

    def load_model(self, model_name, temperature=0.1):
        model = self._load_model(model_name, temperature)
        if model:
            if self.load_test:
                if self.model_type == 'chat':
                    self.llm.invoke('ping')
                elif self.model_type == 'embedding':
                    self.dimensions = len(self.embedding.embed_query('ping'))
            return model

    @property
    def code(self):
        base = f"#{self.__str__()}\n#{self.provider_config.get('doc_url')}\n#{self.provider_config['models']+self.provider_config.get('embeddings',[])}\n"
        libinfo = "from langchain_openai import ChatOpenAI, OpenAIEmbeddings"
        code = self._code.replace(',',',\n')
        if self.provider == 'qwen' and self.model_type == 'embedding':
            libinfo += '\nfrom langchain_community.embeddings import DashScopeEmbed'
        if self.provider == 'azure' and self.model_type == 'embedding':
            libinfo += '\nfrom langchain_openai import AzureChatOpenAI'
        if self.libinfo:
            return f"{base}\n{libinfo}\n\n{code}"
        return f'{base}\n{code}'

    @code.setter
    def code(self, value):
        self._code = value


def main():
    parser = argparse.ArgumentParser(description='Load model')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('--libinfo', type=bool, default=False, help='include import str')
    parser.add_argument('--load_test', type=bool, default=False, help='Test loading model')

    args = parser.parse_args()

    model_manager = ModelManager(libinfo=args.libinfo, load_test=args.load_test)
    model_manager.load_model(args.model_name)
    clip = model_manager.code
    print(clip)
    pyperclip.copy(clip)


if __name__ == '__main__':
    main()

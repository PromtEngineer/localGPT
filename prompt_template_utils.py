'''
This file implements prompt template for llama based models.
Modify the prompt template based on the model you select.
This seems to have significant impact on the output of the LLM.
'''

from langchain.prompts import PromptTemplate

# this is specific to Llama-2.

system_prompt = """You are a helpful DEX's features extractor, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on
the provided context, inform the user. Do not use any other information for answering user"""


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None):

    if promptTemplate_type=="llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        instruction = """
        Context: {context}
        User: {question}"""

        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    else:
        prompt_template = system_prompt + """

        Context: {context}
        User: {question}
        Value:"""
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    return prompt

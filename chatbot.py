
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from llms.gemini import GeminiLLm
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# reference = https://python.langchain.com/v0.1/docs/integrations/chat/google_generative_ai/

llm = GeminiLLm().chatBot

memory = ConversationBufferMemory(memory_key="messages",
                                   return_messages=True)

# connversation summary memory
# memory = ConversationSummaryMemory(memory_key="messages",
#                                    return_messages=True,
#                                    llm=llm)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

if __name__=="__main__":
    while True:
        content = input("Human: ")
        result = chain({
            'content': content
        })
        print(result['text'])

# messages can be saved as json as well
# using filechatmessagehistory from langchain.memory

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llms.gemini import GeminiLLm


llm = GeminiLLm().gemini_pro

tweet_prompt = PromptTemplate.from_template("{topic}.")

tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=False, output_key="funs")



if __name__=="__main__":
    # topic = "tODAY IN HISTORY"
    # resp = tweet_chain.run(topic=topic)
    result = tweet_chain({
        'topic': 'tODAY IN HISTORY'
    })
    print(result.keys())
    print(result['funs'])
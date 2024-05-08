from langchain.llms import openai #deprected
from langchain import prompts #prompt templates with multiple options
from langchain import chains #for sequential chains
from langchain_openai import OpenAI

API_KEY = "sk-proj-6CYVu9sfpT2svyUaBvNeT3BlbkFJVyGgLgRRdK5NEyOEg4pd" #copy OPENAI key generated

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=API_KEY) #text-ada-001 is now terminated

#Inital test woth single prompt. If you want mulruple, just keep on adding them differentiating with a ','.
#Try  using temperature parameters to adjust randomness of the output (0.7) by default
print(f"LLM ouput: {llm("Tell me a quick fact about human intelligence")}")

## Experimenting with a SIMPLE prompt template
prompt_template = "State {number} fun fact about {concept}"
prompt = prompts.PromptTemplate(
   input_variables=["number", "concept"],
   #input_types=[int, str],
   template=prompt_template
)
format_prompt = prompt.format(number=1, concept="Artificial Intelligence")
print(f"\nPrompt Template output: {llm(format_prompt)}")


##Explore multiple chain which is nothing but multiple commands
template = "What is the most popular dance in {country}? Just return the name of the dance form"
first_prompt = prompts.PromptTemplate(
   input_variables=["country"],
   template=template
)
chain_one = chains.LLMChain(llm = llm, prompt = first_prompt)
# second step in chain
second_prompt = prompts.PromptTemplate(
   input_variables=["dance"],
   template="What are the top three facts about {dance}. Just return the answer as bullet points."
)
chain_two = chains.LLMChain(llm=llm, prompt=second_prompt)
overall_chain = chains.SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
final_answer = overall_chain.invoke("India")

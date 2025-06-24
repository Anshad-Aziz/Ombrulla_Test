import os 
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import ChatOpenAI

load_dotenv()

def generate_description(prompt_text,detected_objects):
    formatted_objects = ", ".join([f"{label}({confidence:.2f})" for label, confidence in detected_objects])

    full_prompt=(f"{prompt_text}\n"
                 f"The image contains:{formatted_objects}.\n"
                 "Generate a detailed description of the image based on the detected objects.")
    
    llm=ChatOpenAI(temperature=0.7)

    api_key = os.getenv("GROQ_API_KEY")

    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/v1",
                         model="llama3-70b-instruct")
    
    prompt=PromptTemplate.from_template("{input}")

    chain=LLMChain(llm=llm, prompt=prompt)

    return chain.run(input=full_prompt)
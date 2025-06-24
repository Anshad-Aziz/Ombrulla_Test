# llm_processor.py
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

class LLMProcessor:
    def __init__(self, model_name="llama3-8b-8192"):
        """Initialize Groq LLM via LangChain."""
        try:
            self.llm = ChatGroq(
                model_name=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                # Base URL is handled internally by langchain-groq; no need to specify
            )
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {str(e)}")

    def generate_description(self, prompt_text, detected_objects):
        """Generate a detailed description combining the user prompt and detected objects."""
        try:
            # Format detected objects (list of dicts: {"label": str, "confidence": float})
            formatted_objects = ", ".join(
                [f"{obj['label']} ({obj['confidence']:.2f})" for obj in detected_objects]
            ) if detected_objects else "No objects detected."

            # Create the full prompt
            full_prompt = (
                f"{prompt_text}\n"
                f"The image contains: {formatted_objects}.\n"
                "Generate a detailed description of the image based on the detected objects."
            )

            # Define prompt template
            prompt = PromptTemplate.from_template("{input}")

            # Create LLM chain
            chain = LLMChain(llm=self.llm, prompt=prompt)

            # Run the chain
            return chain.run(input=full_prompt)
        except Exception as e:
            return f"Text generation failed: {str(e)}"

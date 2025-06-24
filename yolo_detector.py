from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

class LLMProcessor:
    def __init__(self, model_name="llama3-8b-8192"):
        """Initialize Groq LLM via LangChain."""
        try:
            self.llm = ChatGroq(
                model_name=model_name,
                api_key=os.getenv("GROQ_API_KEY")
            )
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {str(e)}")

    def generate_response(self, detections, user_prompt):
        """Generate a coherent response combining detections and user prompt."""
        try:
            # Format detections into a string
            if isinstance(detections, dict) and "error" in detections:
                detection_text = detections["error"]
            else:
                detection_text = ", ".join(
                    [f"{d['label']} (confidence: {d['confidence']:.2f})" for d in detections]
                ) if detections else "No objects detected."

            # Create prompt template
            prompt_template = PromptTemplate(
                template="Based on the following detected objects in an image: {detections}\n"
                         "And the user prompt: {user_prompt}\n"
                         "Generate a coherent and meaningful response.",
                input_variables=["detections", "user_prompt"]
            )

            # Generate response
            chain = prompt_template | self.llm
            response = chain.invoke({
                "detections": detection_text,
                "user_prompt": user_prompt
            })
            return response.content
        except Exception as e:
            return f"Text generation failed: {str(e)}"
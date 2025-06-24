# Image Analysis and Text Generation App
[![Watch the Demo]](https://drive.google.com/file/d/1rzgd7FZodsoyc4zEdRErhn09IC6f0JYP/view?usp=sharing)
A Python application that integrates YOLO for object detection, LangChain with Groq's LLaMA3 model for text generation, and Streamlit for the user interface. The app accepts an image and a text prompt, detects objects in the image, and generates a coherent text response combining the analysis with the prompt.

## Prerequisites
- Python 3.8 or higher
- Visual Studio Code (recommended)
- Groq API key (sign up at [groq.com](https://groq.com))

## 🔧 Technologies Used
- **YOLOv11n** (Ultralytics) for object detection
- **LangChain + Groq LLaMA3** for text generation
- **Streamlit** for web interface



## Setup Instructions
1. **Clone the Repository** (or create the project structure manually):
   ```bash
   git clone <repository_url>
   cd image_text_app
   pip install -r requirements.txt
   python streamlit app.py

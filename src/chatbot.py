import os
from PyPDF2 import PdfReader
from groq import Groq
from dotenv import load_dotenv
import json

class Chatbot:
    def __init__(self):
        # Load environment variables and configure Groq API
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        
        self.pdf_path = None
        self.pdf_reader = None
        self.total_pages = 0
        self.current_page_text = None

    def process_pdf(self, pdf_path):
        """Load PDF and get total page count"""
        try:
            self.pdf_path = pdf_path
            self.pdf_reader = PdfReader(pdf_path)
            self.total_pages = len(self.pdf_reader.pages)
            return True
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False

    def get_page_text(self, page_number):
        """Extract text from specific page"""
        try:
            if not self.pdf_reader or page_number < 1 or page_number > self.total_pages:
                return None
            
            page = self.pdf_reader.pages[page_number - 1]  # Convert to 0-based index
            return page.extract_text()
        except Exception as e:
            print(f"Error extracting page text: {str(e)}")
            return None

    def get_response(self, user_message, page_number=None):
        """Get response for user question about specific page"""
        try:
            # If no PDF is loaded
            if not self.pdf_reader:
                yield "Please upload a PDF document first."
                return

            # If no page number provided, ask for it
            if page_number is None:
                yield f"Please specify a page number between 1 and {self.total_pages}."
                return

            # Validate page number
            if not isinstance(page_number, int) or page_number < 1 or page_number > self.total_pages:
                yield f"Invalid page number. Please specify a page between 1 and {self.total_pages}."
                return

            # Get page text
            page_text = self.get_page_text(page_number)
            if not page_text:
                yield "Error: Could not extract text from the specified page."
                return

            # Create system prompt
            system_prompt = """You are a helpful assistant. Use the provided page content to answer the user's question.
            If the answer cannot be found in the provided content, say "I cannot find the answer in this page content."
            Always be clear and concise in your responses."""

            # Prepare messages for Groq
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the page content:\n\n{page_text}\n\nUser Question: {user_message}"}
            ]

            # Get streaming response from Groq
            completion = self.client.chat.completions.create(
                model="llama-3.2-90b-text-preview",  # or another Groq model of your choice
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None
            )

            # Stream the response
            for chunk in completion:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

        except Exception as e:
            yield f"Error generating response: {str(e)}"

    def get_total_pages(self):
        """Return total number of pages in the loaded PDF"""
        return self.total_pages
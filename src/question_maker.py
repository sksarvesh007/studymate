import json
import os
from groq import Groq

class QuestionMaker:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.system_prompt = """You are an expert question generator. Your task is to create diverse questions based on the provided text content.
        Generate a mix of questions including:
        - Factual recall questions
        - Conceptual understanding questions
        - Application-based questions
        - Analysis questions
        
        Format the output as a JSON array with each question object containing:
        - question_text: The actual question
        - question_type: The type of question (factual/conceptual/application/analysis)
        - correct_answer: The correct answer
        - difficulty: A difficulty rating (easy/medium/hard)
        
        Generate 5 questions for the given content."""

    def generate_questions(self, extracted_text):
        try:
            # Prepare the complete prompt
            prompt = f"""Context: {extracted_text}

            Based on the above context, generate questions following the format specified.
            Return only the JSON array without any additional text."""

            # Make the API call to Groq
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",  
                temperature=0.7,
                max_tokens=2000
            )

            # Extract the response
            response_text = completion.choices[0].message.content

            # Parse the JSON response
            questions = json.loads(response_text)

            # Save questions to a JSON file
            self._save_questions(questions)

            return questions

        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return None

    def _save_questions(self, questions):
        # Create questions directory if it doesn't exist
        os.makedirs('questions', exist_ok=True)
        
        # Save to JSON file
        output_file = 'questions/generated_questions.json'
        with open(output_file, 'w') as f:
            json.dump(questions, f, indent=4)


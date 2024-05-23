from transformers import pipeline
import gradio as gr


DESCRIPTION = '''
<div>
<h1 style="text-align: center;">SummarQA</h1>
<p>Quickly understand any text with clear, concise summaries.</p>
<p>Get answers to your questions based on the text, without needing to read everything.</p>
<p>It helps you find the information you need fast, boosting your productivity.</p>
</div>
'''

LICENSE = """
<p/>
---
Built with Falconsai/text_summarization and deepset/tinyroberta-squad2
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
  <h1>SummarQA</h1>
  <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ready to give it a try? </p>
</div>
"""


css = """
body {
  background: linear-gradient(to right, #E0EBF1, #F5F5F5); /* Light blue gradient background */
}
h1 {
  text-align: center;
  display: block;
  font-size: 28px; /* Increase heading size slightly */
  margin-bottom: 10px; /* Add some space after heading */
  color: #333; /* Darker text for better contrast */
}
"""

summarizer = pipeline("summarization", model="Falconsai/text_summarization")
qa = pipeline("question-answering", model="deepset/tinyroberta-squad2")  # or deepset/roberta-base-squad2


with gr.Blocks(fill_height=True, theme=gr.themes.Soft(),css=css) as interface:
    gr.Markdown(DESCRIPTION)
    chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')
    msg = gr.Textbox(label="Your Input", elem_id="gr-textbox")  # Add class to textbox
    clear = gr.Button("Clear", elem_id="gr-button")  # Add class to button
    gr.Markdown(LICENSE)


    def summarize(text):
        """Summarizes the provided text."""
        summary = summarizer(text, max_length=2000, min_length=50, do_sample=False)
        return summary[0]['summary_text']


    def answer(context, question):
        """Answers questions based on the provided context."""
        answer = qa(question=question, context=context)["answer"]
        return answer if answer else "No answer found for your question."


    def chat(text, history):
        """Handles user input, decides between summarization or Q&A, and updates history."""
        if "?" in text:  # If text contains a question mark, treat it as a question
            context, _ = history[-1]
            response = answer(context, text)
        else:
            response = summarize(text)
        history.append([text, response])
        return "", history  # Return both response and updated history


    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear.click(lambda: None, None, chatbot, queue=False)

interface.queue()
interface.launch(share=True)

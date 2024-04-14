# T5-qa-builder: QA Pair Generation

This is a fine-tuned version of the `google/flan-t5-base` model, aimed at generating question-answer pairs from input text. The model is trained on various question-answering datasets including SQuAD, QuAC, Natural Questions, and a custom Q/A dataset. It was initially used as additional embeddings on top of a SentenceTransformer to improve the retrieval of contextual data in knowledge base systems at runtime.

Additionally, the model is useful for generating synthetic data from several sources of textual data to train other conversational models.

The model is being distributed from HF on https://huggingface.co/sgarbi/t5-qa-builder.

Usage
The easiest way to use this model is through the custom QABuilderPipeline:


```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained('sgarbi/t5-qa-builder')
tokenizer = AutoTokenizer.from_pretrained('sgarbi/t5-qa-builder')

input = '''<qa_builder_context>A new breed of AI-powered coding tools have emerged—and they’re claiming to be more autonomous versions of earlier assistants like GitHub Copilot, Amazon CodeWhisperer, and Tabnine.

One such new entrant, Devin AI, has been dubbed an “AI software engineer” by its maker, applied AI lab Cognition. According to Cognition, Devin can perform all these tasks unassisted: build a website from scratch and deploy it, find and fix bugs in codebases, and even train and fine-tune its own large language model.

Following its launch, open-source alternatives to Devin have cropped up, including Devika and OpenDevin. Meanwhile makers of established assistants have not been standing still. Researchers at Microsoft, GitHub Copilot’s developer, recently uploaded a paper to the arXiv preprint server introducing AutoDev, which uses autonomous AI agents to generate code and test cases, run tests and check the results, and fix bugs within the test cases.''' # citation https://spectrum.ieee.org/ai-code-generator

output = model.generate(tokenizer(input, return_tensors='pt').input_ids, max_length=512)

decoded = tokenizer.decode(output[0], skip_special_tokens=False)
print(decoded)
```

#### QA Pairs (raw output):
```xml
<qa_builder_question>What are some of the new AI-powered coding tools emerging?<qa_builder_answer> Some of the new AI-powered coding tools are claiming to be more autonomous versions of earlier assistants like GitHub Copilot, Amazon CodeWhisperer, and Tabnine.<qa_builder_question> How has Devin AI been dubbed by Cognition?<qa_builder_answer> Devin AI has been dubbed an "AI software engineer" by Cognition.<qa_builder_question> What tasks can Devin AI perform unassisted?<qa_builder_answer> Devin AI can build a website from scratch, find and fix bugs in codebases, train and fine-tune its own large language model.<qa_builder_question> What open-source alternatives to Devin AI have cropped up after its launch?<qa_builder_answer> Open-source alternatives to Devin AI have cropped up, including Devika and OpenDevin.<qa_builder_question> What is the new AI-powered coding tool introduced by researchers at Microsoft?<qa_builder_answer> AutoDev, which uses autonomous AI agents to generate code and test cases, run tests, check results, and fix bugs within test cases, is introduced by researchers at Microsoft.

```

## Custom pipeline

The easiest way to generate text with more control and without length restrictions is by using a custom pipeline. The similarity_threshold is used to increase and decrease the similarity of questions returned. Additionally, the stride argument is used to overlap data in the context parts.
```shell
pip install git+https://github.com/ESgarbi/t5-qa-builder
```

Run the custom pipeline. Weights will be downloaded from the HF hub automatically. All special tokens will be handled internally, no need to supply the <qa_builder_context> token to initialise the task.

```python

from sgarbi import QABuilderPipeline

producer = QABuilderPipeline()

input = '''What is artificial intelligence?
If you hear the term artificial intelligence (AI), you might think of self-driving cars, robots, ChatGPT, other AI chatbots, and artificially created images. But it's also important to look behind the outputs of AI and understand how the technology works and its impacts on this and future generations.

AI is a concept that has been around formally since the 1950s when it was defined as a machine's ability to perform a task that would've previously required human intelligence. This is quite a broad definition that has been modified over decades of research and technological advancements.

When you consider assigning intelligence to a machine, such as a computer, it makes sense to start by defining the term 'intelligence' -- especially when you want to determine if an artificial system truly deserves it. 

Also: ChatGPT vs. Microsoft Copilot vs. Gemini: Which is the best AI chatbot?

Our level of intelligence sets us apart from other living beings and is essential to the human experience. Some experts define intelligence as the ability to adapt, solve problems, plan, improvise in new situations, and learn new things. 

With intelligence sometimes seen as the foundation for being human, it's perhaps no surprise that we'd try and recreate it artificially in scientific endeavors. 

Today's AI systems might demonstrate some traits of human intelligence, including learning, problem-solving, perception, and even a limited spectrum of creativity and social intelligence.

AI comes in different forms and has become widely available in everyday life. The smart speakers on your mantle with Alexa or Google voice assistant built-in are two great examples of AI. Other good examples include popular AI chatbots, such as ChatGPT, the new Bing Chat, and Google Bard. 

When you ask ChatGPT for the capital of a country, or you ask Alexa to give you an update on the weather, the responses come from machine-learning algorithms.

''' # Citation: https://www.zdnet.com/article/what-is-ai-heres-everything-you-need-to-know-about-artificial-intelligence

result = producer(context=input, silent_mode=False, json_output=True)
print(result)
```


#### Output:
```json
{
    "What is artificial intelligence? If you hear the term artificial intelligence (AI), you might think of self-driving cars, robots, ChatGPT, other AI chatbots, and artificially created images. But it's also important to look behind the outputs of AI and understand how the technology works and its impacts on this and future generations. AI is a concept that has been around formally since the 1950s when it was defined as a machine's ability to perform a task that would've previously required human intelligence. This is quite a broad definition that has been modified over decades of research and technological advancements. When you consider assigning intelligence to a machine, such as a computer, it makes sense to start by defining the term 'intelligence' -- especially when you want to determine if an artificial system truly deserves it. Also: ChatGPT vs. Microsoft Copilot vs. Gemini: Which is the best AI chatbot? Our level of intelligence sets us apart from other living beings and is essential to the human experience. Some experts define intelligence as the ability to adapt, solve problems, plan, improvise in new situations, and learn new things. With intelligence sometimes seen as the foundation for being human, it's perhaps no surprise that we'd try and recreate it artificially in scientific endeavors. Today's AI systems might demonstrate some traits of human intelligence, including learning, problem-solving, perception, and even a limited spectrum of creativity and social intelligence. AI comes in different forms and has become widely available in everyday life. The smart speakers on your mantle with Alexa or Google voice assistant built-in are two great examples of AI. Other good examples include popular AI chatbots, such as ChatGPT, the new Bing Chat, and Google Bard. When you ask ChatGPT for the capital of a country, or you ask Alexa to give you an update on the weather, the responses come from machine-learning algorithms. ": [
        {
            "question": "What is artificial intelligence (AI) and how has it been defined?",
            "answer": "Artificial intelligence (AI) is a concept that has been around formally since the 1950s when it was defined as a machine's ability to perform tasks that would've previously required human intelligence.",
            "score": 85.98722666501999
        },
        {
            "question": "What is the term 'intelligence' and how has it been modified?",
            "answer": "The term 'intelligence' has been modified over decades of research and technological advancements, and it has been defined as the ability to adapt, solve problems, plan, improvise in new situations, and learn new things.",
            "score": 85.98722666501999
        },
        {
            "question": "What are some traits of human intelligence mentioned in the text?",
            "answer": "Some traits of human intelligence mentioned in the text include learning, problem-solving, perception, and a limited spectrum of creativity and social intelligence.",
            "score": 85.98722666501999
        },
        {
            "question": "What are some examples of popular AI chatbots?",
            "answer": "Examples of popular AI chatbots include ChatGPT, the new Bing Chat, and Google Bard.",
            "score": 85.98722666501999
        }
    ]
}
```

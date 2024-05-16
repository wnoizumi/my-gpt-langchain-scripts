from gpt4all import GPT4All

MODEL_NAME = "mistral-7b-openorca.gguf2.Q4_0.gguf"
# Set with the path to the folder where you store your models
MODELS_FOLDER = "/"

SYSTEM_TEMPLATE = '''
<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI.
You strive to answer concisely.
Your goal is to perform sentiment analysis on the provided input.
Be analytical and critical in your analysis, and very importantly, don't repeat parts of your answer.
Give all the answers in Portuguese.
The answer should always be provided in two lines.
In the first line of the answer, provide a detailed analysis with the "Análise" label.
In the second line of the answer, provide a score 1 for positive, 0 for neutral, or -1 for negative with the "Escore" label.
<|im_end|>
'''

PROMPT_TEMPLATE = '''
<|im_start|>user
{0}<|im_end|>
<|im_start|>assistant
{1}<|im_end|>
'''

model = GPT4All(model_name=MODEL_NAME, model_path=MODELS_FOLDER, allow_download=False)

print("======== Análise de Sentimentos em Textos Curtos ========")
print("Escore: [1 - Positivo] / [0 - Neutro] / [-1 - Negativo]")
print("---------------------------------------------------------")

with model.chat_session(SYSTEM_TEMPLATE, PROMPT_TEMPLATE):
    while (True):
        input_text = input("Digite o texto a ser avaliado: ")
        analysis_result = model.generate(input_text)
        print(analysis_result)
        if input("\nDeseja avaliar mais textos? [s/n] ").lower() != 's':
            break
print("Fim do Programa!")
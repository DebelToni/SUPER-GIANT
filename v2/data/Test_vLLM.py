from vllm import LLM, SamplingParams
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("Teacher.yml")
model_path = config.model_path  # e.g., "/model" inside container or a HF model name
llm = LLM(model=model_path)  # Initialize vLLM engine with the teacher model:contentReference[oaicite:3]{index=3}

# Stage 1: Generate a batch of user questions in one go
prompt = config.question_generation_prompt  # the long prompt as described above
params = SamplingParams(temperature=config.temperature, top_p=config.top_p, 
                        max_tokens=config.max_tokens_question)
outputs = llm.generate([prompt], params)  # single prompt in a list
questions_text = outputs[0].outputs[0].text  # the generated text containing multiple questions
print("Raw questions output:\n", questions_text)


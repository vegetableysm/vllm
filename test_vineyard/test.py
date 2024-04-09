from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is haha. I",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

llm = LLM(model="/mnt/yuansm/model/llama-7b")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
# Fine-Tuned-Llama-2
![image](https://img.shields.io/badge/Llama2-0467DF.svg?style=for-the-badge&logo=Meta&logoColor=white)

Llama2 has undergone a fine-tuning process using a Persian dataset (https://huggingface.co/datasets/SajjadAyoubi/persian_qa), leveraging the Parameter Efficient Fine-Tuning approach, specifically the QLoRA approach. This novel methodology focuses on efficiently updating the model's parameters using a smaller amount of data, while still achieving significant improvements in performance. By employing QLoRA, Llama2 has been fine-tuned with Persian-specific data, enabling it to generate high-quality Persian text. This approach ensures that the model adapts and specializes in the nuances of the Persian language, resulting in enhanced fluency, coherence, and accuracy in generating Persian text. With its fine-tuned capabilities, Llama2 is poised to provide even more reliable and contextually appropriate responses, making it a valuable tool for Persian language tasks such as content generation, translation, and information retrieval. and I saved model in HuggingFace, you can find this model in the link below: [Model](https://huggingface.co/AminHagh78/llama-2-7b-Persian)



## Deployment

To deploy this project you need to first load this model from huggingface. For test run the code below :
```bash
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

HUGGING_FACE_USER_NAME = "AminHagh78"
model_name='llama-2-7b-Persian'

peft_model_id = f"{HUGGING_FACE_USER_NAME}/{model_name}"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
qa_model = PeftModel.from_pretrained(model, peft_model_id)
```
now the model is ready and you can use it
```bash
prompt = "علوم کامپیوتر"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```


## Limitation 
While Llama2 has been fine-tuned with a Persian dataset using the parameter efficient fine-tuning approach, specifically QLoRA, it is important to consider the limitations imposed by the training process. Due to resource constraints, the fine-tuning was conducted on a limited scale, with only one epoch and 700 steps, within the time constraints of approximately 5 hours on a Kaggle notebook. This abbreviated training duration might impact the model's performance and its ability to generate high-quality Persian text. Running the model for longer epochs and more steps could potentially yield improved results, as it would allow the model to further learn and refine its understanding of the Persian language. Therefore, it is crucial to acknowledge that the current performance of the fine-tuned Llama2 model might not be optimal, and further training with additional resources and longer training duration may be required to achieve better text generation outcomes.

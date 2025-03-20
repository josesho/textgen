# Prediction Envy

The code to accompany my submission to Michigan Quarterly Review's [Special Issue on Computer-Generated Text](https://mqr.submittable.com/submit/318008/a-special-issue-on-computer-generated-text/).

## Dependencies

- 	jupyterlab
- 	numpy
- 	pandas
- 	transformers


Start by installing with `pip`:



```shell
pip install jupyterlab numpy pandas
```

At the time of writing, to run the `gemma-3-4b-it` model used in `21-forking-chatbot.ipynb`, you will need to install the `transformers` directly from Github instead:


```shell
pip install git+https://github.com/huggingface/transformers.git
```

## Notes on running the models

Currently, the code used in the Jupyter notebooks here are meant to be used on an Apple Silicon system. So the models will be loaded and used as such:


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

gpt2_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large')
gpt2_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-large').to('mps')

prompt = "Hello, world!"
encoded = gpt2_tokenizer(prompt, return_tensors="pt").to("mps")
```

Where the model and the encoded prompt are moved to the MPS device.

If you want to run it on the CPU (which will take longer), use:


```python
gpt2_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-large').to('cpu')
encoded = gpt2_tokenizer(prompt, return_tensors="pt").to("cpu")
```

If you have a Nvidia GPU, you can use:


```python
gpt2_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-large').to('cuda:0')
encoded = gpt2_tokenizer(prompt, return_tensors="pt").to("cuda:0")
```

You can read more [here](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device).

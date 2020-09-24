# GPT2 NLG
# Overview
This repo provides the functionality for:
1. ```app.py```: A simple Flask web app that allows you to generate text with or without prompt based on a GPT2 model
2. ```nlg.ipynb```: Finetuning a 124M GPT2 model using Shakespeare's plays dataset. Code for CPU as well as GPU
 finetuning are provided.

# Setup
~~~bash
# Create environment
mkdir ~/envs
cd ~/envs
python3 -m venv ~/envs/nlg_env

# Activate environment
source ~/envs/project_env/bin/activate

# Upgrade pip setuptools wheel
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Deactivate
deactivate
~~~

~~~python
# Make the following modifications to files in the library so that GPU
# training works successfully

# ~/envs/nlg_env/lib/aitextgen/aitextgen.py
# add the line below relative to the 2 subsequent lines as
# a temporary fix regarding a "show_progress_bar" error
del train_params["show_progress_bar"]
trainer = pl.Trainer(**train_params)
trainer.fit(train_model)

# ~/envs/nlg_env/lib/transformers/tokenization_gpt2.py
def convert_tokens_to_string(self, tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    # Add the line below at the start of convert_tokens_to_string to fix
    # error where instances of None occur in tokens.
    tokens = list(filter(lambda a: a is not None, tokens))
    text = "".join(tokens)
    text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
    return text
~~~
~~~bash
# Run Flask from the root directory
flask run
~~~

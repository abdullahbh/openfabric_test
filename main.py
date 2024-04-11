import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass
# Load model directly
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

# Callback function called on update config
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass

# Callback function called on each execution pass
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        # Tokenize the input text
        inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        return_tensors="pt"
    )

        output_ids = model.generate(
        inputs["input_ids"],
        max_length=30,  # Reduce max_length for faster responses
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

        # Decode the output IDs to text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Add the output text to the output list
    output.append(output_text)

    # Create a SimpleText object and assign the output to the text attribute
    response = SimpleText()
    response.text = output
    return response


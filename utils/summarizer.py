from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .xai_gradient import *

"""
This file is for summarizer functions
Based on the model it could be different pipeline

*** parallel processing is recommended for summarization *** 

function signature:
    args: texts (list), any other arguments if needed
    returns: summary (str)

"""


def summarizer(texts: list, model="facebook/bart-large-cnn", max_length=1024, min_length=0, num_beams=4, cal_grad=False):
    """
    Summarizer based on language model

    Args:
    - texts: list of texts
    - model: model name
    - max_length: maximum length of the summary
    - min_length: minimum length of the summary
    - num_beams: number of beams for beam search
    - cal_grad: whether to calculate token importance using gradients

    Returns:
    - str: summary
    - dict (optional): token importance if cal_grad is True
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model_instance = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    inputs = tokenizer(texts, 
                       max_length=max_length,
                       truncation=True,
                       padding=False, 
                       return_tensors='pt'
    )

    # Prepare inputs for model
    inputs = {key: value.to(model_instance.device) for key, value in inputs.items()}

    # Enable gradient calculation if cal_grad is True
    if cal_grad:
        inputs_embeds = model_instance.get_input_embeddings()(inputs['input_ids'])
        inputs_embeds = inputs_embeds.clone().detach()
        inputs_embeds.requires_grad = True

        outputs = model_instance(inputs_embeds=inputs_embeds, decoder_input_ids=inputs['input_ids'])

        # Generate summaries
        summary_ids = model_instance.generate(inputs['input_ids'], num_beams=num_beams, min_length=min_length, max_length=max_length)

        # Calculate token importance
        output_logits = outputs.logits[:, 0, :]
        target_class = output_logits.max(dim=1)[1]
        target_score = output_logits[:, target_class]
        target_score.sum().backward()

        gradients = inputs_embeds.grad
        gradients_mean = gradients.abs().mean(dim=2)

    else:
        with torch.no_grad():
            summary_ids = model_instance.generate(inputs['input_ids'], num_beams=num_beams, min_length=min_length, max_length=max_length)
        gradients_mean = None

    # Decode summaries
    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    summary = " ".join(summaries)

    # Free unused memory
    del inputs, summary_ids, summaries
    torch.cuda.empty_cache()

    if cal_grad:
        return summary, gradients_mean
    return summary
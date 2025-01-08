from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

"""
This file is for summarizer functions
Based on the model it could be different pipeline

*** parallel processing is recommended for summarization *** 

function signature:
    args: texts (list), any other arguments if needed
    returns: summary (str)

"""


def summarizer(texts: list, model="facebook/bart-large-cnn", max_length=1024, min_length=0, num_beams=4)->str:
    """
    summarizer based on language model

    Args:
    - texts: list of texts
    - model: model name
    - max_length: maximum length of the summary
    - min_length: minimum length of the summary

    Returns:
    - str: summary
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    inputs = tokenizer(texts, 
                       max_length=max_length,
                       truncation=True,
                       padding='longest', 
                       return_tensors='pt'
    )
    with torch.no_grad():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        summary_ids = model.generate(inputs['input_ids'], num_beams=num_beams, min_length=min_length, max_length=max_length)

    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # concatenate summaries TODO: if needed add /n between summaries
    summary = " ".join(summaries)

    # free unusable memory
    del inputs, summary_ids, summaries

    return summary
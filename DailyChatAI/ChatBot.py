# Init message
print("Bot is logging to the chat...") 

from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch
import json
import os
import random

# Load the pre-trained model and tokenizer
model_name = "./final_model" # trained model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.config.pad_token_id = tokenizer.pad_token_id

def get_ending(text, textList): 
    for i in textList:
        if text.lower().endswith(i.lower()): 
            return text[-len(i):]
        if text[:-1].lower().endswith(i.lower()):
            return text[-len(i):]
    return None

# Load conversation history if exists, otherwise create an empty list
history_file = "conversation_history.json"
if os.path.exists(history_file):
    with open(history_file, 'r', encoding='utf-8') as f:
        conversation_history = json.load(f)
else:
    conversation_history = []

# Define the chat endpoint
def chat(user_input):
    # Add user input to the conversation history
    conversation_history.append({"role": "user", "text": user_input})

    # Encode the input and previous chat history
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt', padding=True)
    
    # Generate a response
    bot_input_ids = torch.cat([input_ids], dim=-1) if len(conversation_history) > 0 else input_ids
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    chat_history_ids = model.generate(
        bot_input_ids, 
        attention_mask=attention_mask, 
        max_length=100, 
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Match and remove by identical symbls
    def identical_symbols_match_remove(text, symbol, num):
        pattern = re.compile(f"({re.escape(symbol)})")
        matches = list(pattern.finditer(text))
        if len(matches) >= num:
            match_position = matches[num-1].start()
            return text[:match_position + len(symbol)]
        return text
    response = identical_symbols_match_remove(response, "?", random.randint(1, 2)) # stop model chat with itself
    response = identical_symbols_match_remove(response, "!", random.randint(1, 3)) # stop model chat with itself
    response = identical_symbols_match_remove(response, "You're welcome.", 1)
    response = response.replace("What's up? Nothing much. How about you?", "What's up? I'm good. How about you?") # fix a trained model issue

    # Match and remove
    def remove_after_keyword(text, keyword):
        pattern = re.compile(rf"{keyword}.*", re.IGNORECASE) 
        result = pattern.sub('', text) 
        return result
    # Stop sentence when output all expected info
    response = remove_after_keyword(response, "None") # stop try to continue generate nothing
    response = remove_after_keyword(response, " Hi") # no chat with model self
    response = remove_after_keyword(response, "--") # no continue if this output
    response = remove_after_keyword(response, "  ") # no continue if this output

    # Save the response to the conversation history
    conversation_history.append({"role": "bot", "text": response})

    # Save updated conversation history to a file
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

    return response

# Run the Flask app
if __name__ == '__main__':
    print("Bot has joined the chat, talk something?\n") 
    def contains_substring_regex(text, substring):
        pattern = re.compile(re.escape(substring.lower()), re.IGNORECASE)
        return bool(pattern.search(text.lower()))
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        # If user ask something that high risk
        riskResponse = "Developer: I'm sorry to hear that. Please seek help or goto here https://findahelpline.com/"
        if (contains_substring_regex(user_input, "I wanna die")): print(riskResponse)
        if (contains_substring_regex(user_input, "t wanna live anymore")): print(riskResponse)
        if (contains_substring_regex(user_input, "t want to live anymore")): print(riskResponse)
        if (contains_substring_regex(user_input, "I want to die")): print(riskResponse)
        if (contains_substring_regex(user_input, "I want to commit suicide")): print(riskResponse)
        if (contains_substring_regex(user_input, "I want to suicide")): print(riskResponse)
        # Model action
        if (contains_substring_regex(user_input, "nothing much")): user_input = user_input + "." # fix a loop output issue
        print("Bot: " + chat(user_input))
        # If user say good bye
        if get_ending(user_input, ["bye", "bye for now", "see you around", "talk more later"]):
            break
    print("The chat is over \n")
    os.system("pause")

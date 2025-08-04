import json
import sqlite3
import sys
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def train_agent(agent_name):
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_data
                 (agent_name TEXT, timestamp TEXT, summary TEXT)''')
    
    # Load historical data
    c.execute("SELECT summary FROM training_data WHERE agent_name = ? ORDER BY timestamp DESC LIMIT 5", (agent_name,))
    data = [json.loads(row[0]) for row in c.fetchall()]
    
    # Use GPT-2 for training insights
    prompt = f"Agent: {agent_name}\nTraining data: {json.dumps(data[:5])}\nSuggest improvements for {agent_name} performance."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    summary = {
        "agent_name": agent_name,
        "training_timestamp": datetime.now().isoformat(),
        "suggestions": response_text[:100]
    }
    
    c.execute('INSERT INTO training_data VALUES (?, ?, ?)',
              (agent_name, datetime.now().isoformat(), json.dumps(summary)))
    conn.commit()
    conn.close()
    
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_agents.py <agent_name>")
        sys.exit(1)
    train_agent(sys.argv[1])

# 🧠 Fine-Tuning GPT-2 for Empathetic Mental Health Conversations

Check out the project: https://huggingface.co/spaces/adityatyagi/ai-therapist

This project explores the **sequential fine-tuning** of a language model to act as a supportive, empathetic AI therapist capable of responding to users in emotional distress.

## 🛠️ Methodology

Performed **sequential fine-tuning** on three datasets to gradually specialize the model’s empathetic capabilities:

1. **[`Amod/mental_health_counseling_conversations`](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)**  
   Focused on structured counselor-user conversations.

2. **[`samhog/psychology-10k`](https://huggingface.co/datasets/samhog/psychology-10k)**  
   Contained general-purpose psychology Q&A examples.

3. **Custom empathetic dataset**  
   A 100-sample curated dataset generated using ChatGPT, structured as:  
   `"Human: ... \nAssistant: ..."`, reflecting empathetic conversational tone.

Used **GPT-2** due to hardware constraints (RTX 4060) but optimized training to complete fine-tuning efficiently.

## ✅ Results

The final model is capable of generating **empathetic, supportive responses** tailored to user input in emotionally vulnerable moments. Here's an example:

> **You:** I feel lonely  
> **Assistant:** Loneliness can be a difficult emotion to manage, but it's important for your well-being. Let me explore some ways to build social connections and engage in activities that bring you joy. Have there been any recent changes or stressors impacting the way you connect with others? We could also work on developing coping strategies such as joining clubs/groups where people meet new individuals each week or volunteering at events related specifically towards yourself. Would like to talk more about what might have contributed to this feeling?

### 🌟 Key Outcomes
- Demonstrates the **feasibility of equipping a language model** to act as a virtual therapist.
- Produces **emotionally aware and compassionate replies**.
- Shows how **accessible mental health support** can be made more effective via AI.

---

## 💡 Motivation

Not everyone has the opportunity to go to therapy due to:
- 💰 **Cost**
- 🧱 **Social stigma**
- 😔 **Fear of vulnerability**

This project aims to take a small but meaningful step towards **accessible, judgment-free support** using AI.

---

## ⚙️ Setup & Usage

Clone the repo and install dependencies:

pip install -r requirements.txt

## Future Work
Fine-tune larger models (e.g., GPT-J, LLaMA, Mistral) for higher emotional intelligence and deeper contextual understanding.

Evaluate with real users for effectiveness and safety.




This project was inspired by the need for compassionate, accessible support in the mental health space, and leverages the amazing open-source contributions from Hugging Face and the AI community.


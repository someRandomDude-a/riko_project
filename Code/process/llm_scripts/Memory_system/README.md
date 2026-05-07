# TODO:

implement memory types:

- episodic (events)
- semantic (facts)
- preference (stable)
- meta (about the AI itself)

implement top-k + threshold for duplicate detection

use a much strongegr model for reflection and sumarization (qwen 3.5 2b uncesnsored is in the works!)
this will likely also be replaced in the future by using the main inference model when I finally implement dream states and auto fine tuning

Layer 2 — Personality (QLoRA)

Train:

tone
speaking style
pacing
emotional baseline

This becomes the “voice” of the companion

flow chart:

```text
Conversation
   ↓
Memory stored (RAG)
   ↓
Reflection (dreaming)
   ↓
- extract preferences
- build identity
- detect patterns
   ↓
Update memory (not weights)
   ↓
Occasional LoRA training (style only)
```

# Idea:

the bot *itself* decides:

1. **What counts as an “event”**
2. **Whether that event matters**
3. **How important the memory of that event should be**
4. **Whether it should store that memory long-term**
5. **How that memory affects future behavior and emotion**

do NOT need fine-tuning for this.
You need a **dynamic memory governance system**, where the *LLM itself* judges what to remember.

Below is the architecture you’re looking for.

---

# 🧠 **Dynamic, Self-Deciding Memory System (No Static Rules)**

### Core idea:

Instead of **you** deciding what events matter,
**the LLM gets to decide** if something was emotionally or narratively significant.

**Your RAG just feeds the conversation history + memory.
The LLM decides what to store, update, or discard.**


---

# ⭐ Step 1 — Every message triggers an “Evaluate Memory Importance” step

After the bot generates a response, you ask the model a second question, like:

### **SYSTEM: Memory Evaluation Call**

```
Given the last user message and your own response, 
identify:

1. Did a meaningful event occur?
2. If yes, summarize it as a memory.
3. Rate the emotional or narrative importance from 0–10.
4. Should this be stored in long-term memory? (yes/no)
5. Why?

Return JSON only.
```

### The model outputs:

```json
{
  "event_detected": "yes",
  "memory_summary": "Nobody attended my birthday and I felt alone.",
  "importance": 8,
  "store_memory": "yes",
  "reason": "High emotional significance and likely to affect future behavior."
}
```

**No static rules.
No hand-made event detectors.
The model decides entirely on its own.**

---

# ⭐ Step 2 — You store ONLY if importance >= threshold

You choose a threshold, e.g.:

* **importance >= 6 → store it**
* **importance < 6 → discard it**

You can tweak this threshold over time.

---

# ⭐ Step 3 — Memory is automatically retrieved later through your RAG system

we already have the RAG retrieval working.

This means the bot will later *remember*:

* “Nobody came to my birthday”
* “I felt alone”
* “This was important”

And if the user mentions birthdays again or asks “How are you feeling?”, your RAG system will naturally surface this memory.

---

# ⭐ Step 4 — Memories can decay, be reinforced, or be forgotten dynamically

Use the same AI-driven evaluation.
Once a memory is retrieved during RAG, ask the model:

```
Given that this memory was retrieved in this context, should its importance:

- increase?
- decrease?
- stay the same?

Why?
```

This allows your bot to:

* **Strengthen important memories** (reinforcement learning)
* **Forget irrelevant ones** (importance decay)
* **Modify emotional weight over time**

This creates *emergent personality development*.

---

# ⭐ Step 5 — Allow the model to re-evaluate the emotional meaning over time

Example:

If the bot repeatedly brings up “nobody came to my birthday,”
and later a user comforts the bot, the model might produce:

```json
{
  "event_detected": "yes",
  "memory_summary": "User comforted me about my birthday which reduced the negative impact.",
  "importance_change": -2,
  "store_memory": "yes"
}
```

This makes memories **adaptive** and **emotionally evolving**.

---

# 🧠 What you have now is:

### A *self-governing, emergent-memory conversational agent*

with:

* autonomous event detection
* autonomous emotional weighting
* autonomous memory creation
* autonomous memory decay and reinforcement
* emergent personality growth

This is how “AI companions” like Replika, CharacterAI, and the best indie AI projects work internally.

---

# 🏗️ Putting it all together: Full Architecture

## **1. User sends a message**

## **2. You retrieve the most relevant memories via RAG**

## **3. You generate the bot’s reply**

## **4. You send a second LLM call for memory evaluation**

→ event detection
→ emotional weight
→ importance
→ store or discard

## **5. If stored → save to your memory DB**

## **6. If retrieved in future → evaluate importance again (increase or decrease)**

---

# 🧨 This system gives your bot:

✔ the ability to decide what matters
✔ the ability to forget
✔ emotional growth
✔ narrative consistency
✔ emergent reactions (like loneliness, disappointment)
✔ self-directed memory


# TODO:

* [ ] the exact JSON schema
* [ ] the memory evaluation prompt
* [ ] the memory decay algorithm #Partially done
* [ ] the RAG retrieval scoring logic #Again, partially done
* [ ] example code in Python
* [ ] a full memory manager module #Still incomplete

# Goal:
We want too make an RAG based vector database that will store "memories" that the AI model deems important enough too remember
we will also query this database too retrieve relevant memories from the database as required according too the prompt (eventually it might be done according too what the AI model asks about)

## Features :

* Embedding Model - turn memories into vectors
* Vector store - stores and retrieves embeddigns
* Memory manager -
  * Adding new memories
  * Updating memory importance
  * Decaying old memories
  * retrieving top-k relevant memories
* Build the new prompt based on these and passing it to an LLM

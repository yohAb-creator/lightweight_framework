# Evaluating LLM Agents in Android World: Report

**Author**: Yohannes Abateneh

**Date** : 2025-07-18

## Introduction

A report on the design details and performance of a lightweight framework built for Large Language Model (LLM) capability assessment as an agent for the `android_world` benchmark. 
Our focus was on designing a framework to test an LLM's capability for reasoning under visual perceptions and generating valid actions for achieving described goals under an Android simulation environment.
### Methodology

For evaluation, we employed the intrinsic `t3a` agent of the `android_world` package as our "expert" or "ground-truth" source.

Our approach was as follows:

1.  For each step of an assigned task, the present environment observation, the list of interactive user interface objects, was extracted from the `t3a` agent's state.

2.  This prompt, along with the purpose of the task, was transformed into a configurable prompt and presented to the custom-prompted LLM. For this project, we used Google's Gemini 1.5 Pro as our LLM.

3.  At the same time, the `t3a` agent also chose its future action given the same information.

4.  Our custom framework's output action had also been matched against the `t3a` agent's selected action. A step the LLM decided to take was considered correct if it exactly matched the step taken by the `t3a` agent.

Such a method facilitated the qualitative evaluation of many prompting techniques based on the performance of the pre-existing agent.

### Metrics

Performance was measured based on two metrics:

* **Episode Success Rate**: What is the percentage of tasks that root `t3a` agent finished? It is a measure of task completion since we would not have a complete source for the ground truth if the intrinsic agent could not achieve the goal.

* **Average Step Accuracy**: What percentage of steps in our LLM prompt had corresponding steps for the expert `t3a` agent steps? This is also our main measure of quality for our prompt and the LLM's reasoning capability.

## Prompting Methods

Two primary prompting strategies, **Zero-Shot** and **Few-Shot**, were applied and proven to estimate their impact upon agent performance. An alternative, **Self-Reflection**, is also being used. For ease of customization, the prompt baseline is initialized to load from a configurable file (`prompts/prompt_template.txt`).
* **Zero-Shot:** As is standard protocol with zero shot prompting, the LLM was presented with only the current observation and the objective, and the model had to generate the following action.

* **Few-Shot:** In this case, the prompt is preceded by hand-curated few-shot examples from `prompts/few_shot_examples.txt`, which provide enough reasoning for most of the frequent events.

* **Self-Reflection:** The LLM needed to learn to present first its reasoning and only subsequently its action, which elicited a more reflective mode of thought. This chain of thoughts approach yielded a surprising amount of increase in performance.

## 3. Performance Summary
-

Evaluations were conducted for **10 episodes** for each of the variant prompts (zero-shot, few-shot, self-reflection). See below for the compiled table of results.

| **Prompt Variant**  | **Episode Success Rate** | **Average Step Accuracy** |
|:--------------------|:-------------------------|:--------------------------|
| **Zero-Shot**       | *70.00*%                 | *32.17*%                  |
| **Few-Shot**        | *70.00*%                 | *33.33*%                  |
| **Self-Reflection** | *80.00*%                 | *33.59*%                  |





## Analysis of Episodes


We present two events, both a result of few-shot prompting, and analyze to determine agent/LLM reasoning strengths and weaknesses.

### Success Case: `ContactsAddContact` (Few-Shot Variant)

* **Goal**: Create a new contact for Santiago Lopez. Their number is +17725761375.

* **Result**: **Success**

* **Step Accuracy**: 18.18%

* **Analysis**: As can be seen from the success in achieving the goal, the agent correctly reasoned out the steps required for achieving the task. As usual, LLM steps for the few-shot prompt model coincided with expert agent selections at crucial points, such as correctly identifying and clicking on UI objects such as "Create contact," correctly filling out the appropriate boxes, and correctly saving the resulting output. It shows that the prompt performs well for typical, simple tasks. It is mainly due to the expert agent carrying out several `waits` and other necessary preparations due to the nature of the emulator, which the LLM correctly skipped, that the LLM's step accuracy is abysmal.
### 4.2. Failure Case: `ContactsAddContact` (Few-Shot)

* **Goal**: Create a new contact for Mohammad Liu. Their number is +13145361159.

* **Result**: **Failure**

* **Step Accuracy**: 46.67%

* **Analysis**: The failure is due to one of the typical LLM reasoning mistakes. After sending "Mohammad" correctly into the first box, step 7 of the LLM, `TYPE("Mohammad Liu")`, is the next step. It tried sending the first and last name boxes as one step. The 't3a' agent, however, correctly identified that they needed to take the "Last name" box next and created the step `TYPE("Liu")`. This mistake is an instance of weakness where the LLM could mistakenly understand the state and try an invalid compound step because of the absence of explicit memory of which box is active.
## 5. Key Findings

It is illustrated that, for a good prompt built by LLM, an LLM can attain good step accuracy for typical mobile UI tasks, where the **Few-Shot** model significantly excels the **Zero-Shot** baseline. The failure observed in some of these relatively high step accuracy events, seems to occur under limited memory and growing contextual comprehension.
Additionally, the device's limitations seemed to have limited the performance of the model too; the emulator would lag at times due to limitations in RAM causing some steps to be wasted.

## Possible Improvements
1.  **Append Memory**: A memory buffer to append a history of the last 1-2 steps and what has been seen. This would provide the LLM with enough context to break loops and resume multi-step sequences.
2. **Fallback**: A fallback system for when the emulator glitches.

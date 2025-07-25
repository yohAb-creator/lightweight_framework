# Report: Multi-Stage Prompt Optimization

**Author**: Yohannes Abateneh

**Date** : July 24, 2025

A report based on the advanced challenge version of the evaluation framework which involved designing and implementing a prompt optimization pipeline that produced different prompt policies to drive an agent of my design to complete tasks from the android_world benchmark.


## Methodology

The pipeline was constructed from three modules as specified.These modules are present in the same src directory that contains the previous challenge's modules. They are:  

 **Picture to Text(gemini_prompting.py)**
**Component:** A function that used `gemini-2.5-pro` to extract UI elements and generate a prompt.
 **Function:** Translated raw pixel data from UI screenshots into an actionable `P0` prompt. 

 **Prompt Optimization through text2grad(text2grad_opt.py)**
**Component:** A prompt mutation based function which simulates the Text2Grad process.
**Function:** Sequentially updated the baseline prompt (`P0`) against a metric for goal completion based on task completion, generating a structurally better prompt (`P*`) to warm-start the final optimization module.

**Prompt Optimization through Reinforcement Learning(apro_runner.py)**
**Component:** An Action-Aware Reinforcement Prompt Optimization (ARPO) loop.
**Function:** Used the improved prompt from the text2grad module(`P*`) as its starting policy and updated the prompt policy according to a composite reward signal made up of task completion and step accuracy, with the goal of settling at a policy of improved quality.

## Results
### ContactsAddContact
The final results, executed after upgrading the base model to `gemini-2.5-pro` after testing with `gemini-1.5-pro-latest` are presented below. The details are presented in the `arpo_full_pipeline_results.json` file.

| Policy (`P`) | Task Completion Rate | Step Accuracy (vs. Expert agent(t3a) |
| :--- | :--- |:-------------------------------------|
| **`P0` (Baseline)** | 100% | 31.4%                                |
| **`P*` (Text2Grad)** | 100% | **85.7%**                            |
| **`P**` (ARPO)** | 100% | 82.8%                                |
### Three Tasks

I wrote the code to run the benchmark on ten separate tasks. However, I came to the realization that this is extremely computationally intensive and would take a long time to finish. In order to test robustness, I used three episodes for each of the tasks making running
the bench mark on 10 tasks unrealistic given the timeframe. The details are presented in `arpo_full_pipeline_results_10tasks.json`


## Observations and Analysis




- The transformation of the relatively 'poor' baseline prompt (`P0`) into the mutated prompt (`P*`) generated through text2grad resulted in a 54.3% in step accuracy. This was a massive jump and we can hypothesize with a certain level of confidence that perhaps the most important factor in an otherwise competent agent's performance is the quality and coherence of its starting reasoning process. 



- The observation, which was surprising at initial glance, is that the ARPO module resulted in a prompt that did not improve step accuracy. In fact, the resulting prompt policy resulted in the agent's step accuracy decreasing by 2.9%. One possible explanation for this is the additive nature of the design. The results from `text2grad` paired with the `gemini-2.5-pro` model were already quite good, and the best APRO could do was add reactive negative constraints to the already highly effective prompt. This added noise did not contribute to advancing the reasoning trajectory of the model and, in fact, limited the model, leading to decreased step accuracy. This added noise might have been effective in increasing reasoning capabilities with a better-designed learning mechanism.

- The lower accuracy when running multiple tasks is due to only one tasks(adding contacts) task being used in the text2grad stage as well as the initial gemini_prompting stage. In order to
- alleviate overtraining on one task, the arpo stage randomly chooses one task at each step. This be leading to incompatible data being extracted from the tasks. Increasing the number of apro runs should combat this problem.
## Future Work Suggestions

For now, the refinement of the APRO script would be a misuse of the little remaining time. Future work should focus on updating the current APRO function with a synthesis module. 
The goal that APRO won't simply be introducing unnecessary restrictions. Instead, it will derive reasoning trajectories that lead to better results. The final goal is the module generalizing and improving upon failing reasoning pathways in the training log. The agent should then be able to learn not the just the constraints but actions to take from those constraints. 
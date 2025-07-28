import os
import re
import random
import time
import json
import sys
from typing import Any, List, Tuple, Dict
from absl import app, flags, logging
from android_world import registry
from android_world.env import env_launcher, json_action, interface
from android_world.agents import infer, base_agent, t3a

# --- Suppress verbose informational logs ---
logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# --- Flag Definitions ---
flags.DEFINE_integer('num_text2grad_mutations', 5, 'Number of prompt variations to test in the Text2Grad phase.')
flags.DEFINE_integer('num_training_iterations_per_task', 10,
                     'Number of RL iterations to run for each task in the training set.')
flags.DEFINE_integer('num_benchmark_episodes', 3, 'Number of episodes to run for each task in the final benchmark.')
flags.DEFINE_string('gcp_api_key', os.environ.get("GCP_API_KEY"), 'Your Google Cloud API Key.')
flags.DEFINE_string('adb_path', "C:\\Users\\yohan\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe",
                    'Path to ADB.')
flags.DEFINE_integer('console_port', 5554, 'Emulator console port.')
flags.DEFINE_integer('grpc_port', 8554, 'Emulator gRPC port.')
flags.DEFINE_integer('max_steps_per_episode', 25, 'Maximum steps for the agent in each episode.')


def get_llm_response(prompt: str) -> str:
    """Generic wrapper to call the LLM."""
    if not FLAGS.gcp_api_key:
        return "ERROR: API Key not set"
    llm = infer.GeminiGcpWrapper('gemini-2.5-pro')
    response, _, _ = llm.predict(prompt)
    return response


# --- Robust Parsing and Equivalence Functions ---

def parse_action_from_llm_response(response_text: str, ui_elements: list[Any]) -> dict[str, Any] | None:
    action_str = response_text
    if "Action:" in response_text:
        action_str = response_text.split("Action:")[-1].strip()

    click_match = re.search(r'CLICK\("(.+?)"\)', action_str, re.IGNORECASE)
    if click_match:
        target_text = click_match.group(1)
        for i, element in enumerate(ui_elements):
            label = (element.content_description or element.text or "").strip()
            if target_text.lower() == label.lower():
                return {"action_type": "click", "index": i}
        return {"action_type": "click", "target_text": target_text}

    type_match = re.search(r'TYPE\("(.+?)", "(.+?)"\)', action_str, re.IGNORECASE)
    if type_match:
        text_to_type = type_match.group(1)
        target_field_text = type_match.group(2)
        for i, element in enumerate(ui_elements):
            label = (element.content_description or element.text or "").strip()
            if target_field_text.lower() == label.lower() and element.is_editable:
                return {"action_type": "input_text", "text": text_to_type, "index": i}
        return {"action_type": "input_text", "text": text_to_type, "target_text": target_field_text}

    open_app_match = re.search(r'OPEN_APP\("(.+?)"\)', action_str, re.IGNORECASE)
    if open_app_match:
        app_name = open_app_match.group(1)
        return {"action_type": "open_app", "app_name": app_name}

    scroll_match = re.search(r'SCROLL\("(.+?)"\)', action_str, re.IGNORECASE)
    if scroll_match:
        direction = scroll_match.group(1).lower()
        if direction in ["up", "down", "left", "right"]:
            return {"action_type": "scroll", "direction": direction}

    if "navigate_back" in action_str.lower():
        return {"action_type": "navigate_back"}

    if "status_complete" in action_str.lower().replace("_", ""):
        return {"action_type": "status", "goal_status": "complete"}

    return None


def are_actions_equivalent(our_action_dict, expert_action_dict, ui_elements):
    if not our_action_dict or not expert_action_dict:
        return False
    our_action_type = our_action_dict.get("action_type")
    expert_action_type = expert_action_dict.get("action_type")
    if our_action_type != expert_action_type:
        return False
    if expert_action_type in ["status", "navigate_back", "navigate_home"]:
        return our_action_dict == expert_action_dict
    if expert_action_type == "click":
        our_idx, expert_idx = our_action_dict.get("index"), expert_action_dict.get("index")
        if our_idx is not None and expert_idx is not None:
            if our_idx == expert_idx: return True
            if our_idx < len(ui_elements) and expert_idx < len(ui_elements):
                our_text = (ui_elements[our_idx].content_description or ui_elements[our_idx].text or "").strip().lower()
                expert_text = (ui_elements[expert_idx].content_description or ui_elements[
                    expert_idx].text or "").strip().lower()
                if our_text and our_text == expert_text: return True
        if "target_text" in our_action_dict and expert_idx is not None:
            if expert_idx < len(ui_elements):
                expert_text = (ui_elements[expert_idx].content_description or ui_elements[
                    expert_idx].text or "").strip().lower()
                if our_action_dict["target_text"].lower() == expert_text: return True
    if expert_action_type == "input_text":
        return our_action_dict.get("text") == expert_action_dict.get("text")
    if expert_action_type == "scroll":
        return our_action_dict.get("direction") == expert_action_dict.get("direction")
    return False


# --- Agent and Environment Functions ---

class SimpleAgent(base_agent.EnvironmentInteractingAgent):
    def __init__(self, env: interface.AsyncEnv, prompt_policy: str):
        super().__init__(env)
        self.prompt_policy = prompt_policy
        self.history = []

    def reset(self, go_home: bool = False):
        super().reset(go_home)
        self.history = ["You have just started the task."]

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        state = self.get_post_transition_state()
        ui_elements = state.ui_elements
        screen_fingerprint = ".".join(sorted([str(e.text) for e in ui_elements if e.text]))
        clickable_elements = [f"- \"{(e.content_description or e.text or '').strip()}\"" for e in ui_elements if
                              e.is_clickable and (e.content_description or e.text)]
        observation_text = "\n".join(clickable_elements) if clickable_elements else "The screen is empty."
        history_text = "\n".join([f"Step {i + 1}: {h}" for i, h in enumerate(self.history)])
        current_prompt = self.prompt_policy.format(goal=goal, observation=observation_text, history=history_text)
        llm_response = get_llm_response(current_prompt)
        action_params = parse_action_from_llm_response(llm_response, ui_elements)
        action_summary = llm_response.split('Action:')[
            -1].strip() if 'Action:' in llm_response else llm_response.strip()
        is_done, action_succeeded = False, False
        if action_params:
            if action_params.get("action_type") == "status":
                is_done, action_succeeded = True, True
            else:
                try:
                    if "index" in action_params:
                        self.env.execute_action(json_action.JSONAction(**action_params))
                        self.history.append(action_summary)
                        action_succeeded = True
                    else:
                        self.history.append(f"Action failed: Element for '{action_summary}' not found.")
                except Exception as e:
                    self.history.append(f"Action failed with error: {e}")
        else:
            self.history.append(f"Could not parse LLM response: {llm_response}")
        step_data = {"screen_fingerprint": screen_fingerprint, "action_taken": action_summary,
                     "action_succeeded": action_succeeded}
        return base_agent.AgentInteractionResult(done=is_done, data=step_data)


def run_episode(prompt_policy: str, task_class, is_benchmark: bool = False) -> Dict:
    trajectory, env = [], None
    try:
        env = env_launcher.load_and_setup_env(adb_path=FLAGS.adb_path, console_port=FLAGS.console_port,
                                              grpc_port=FLAGS.grpc_port)
        task_instance = task_class(params=task_class.generate_random_params())
        goal = task_instance.goal
        print(f"Goal: {goal}")
        task_instance.initialize_task(env)

        our_agent = SimpleAgent(env, prompt_policy)
        our_agent.reset(go_home=True)
        time.sleep(2)

        expert_agent = t3a.T3A(env, infer.GeminiGcpWrapper('gemini-2.5-pro')) if is_benchmark else None

        for _ in range(FLAGS.max_steps_per_episode):
            if is_benchmark:
                state = expert_agent.get_post_transition_state()
                ui_elements = state.ui_elements
                clickable_elements = [f"- \"{(e.content_description or e.text or '').strip()}\"" for e in ui_elements if
                                      e.is_clickable and (e.content_description or e.text)]
                observation_text = "\n".join(clickable_elements) if clickable_elements else "The screen is empty."
                history_text = "\n".join(
                    [f"Step {i + 1}: {h.get('summary', 'N/A')}" for i, h in enumerate(expert_agent.history)])
                current_prompt = prompt_policy.format(goal=goal, observation=observation_text, history=history_text)
                llm_response = get_llm_response(current_prompt)
                our_action_dict = parse_action_from_llm_response(llm_response, ui_elements)

                expert_response = expert_agent.step(goal)
                raw_expert_output = expert_agent.history[-1].get('action_output', '')
                expert_action_dict = None
                if '\nAction: ' in raw_expert_output:
                    try:
                        expert_action_dict = json.loads(raw_expert_output.split('\nAction: ')[-1].strip())
                    except json.JSONDecodeError:
                        pass

                is_match = are_actions_equivalent(our_action_dict, expert_action_dict, ui_elements)
                trajectory.append(is_match)
                if expert_response.done: break
            else:
                result = our_agent.step(goal)
                trajectory.append(result.data)
                if result.done: break
            time.sleep(1)

        task_success = task_instance.is_successful(env) == 1.0
        if is_benchmark:
            accuracy = sum(1 for match in trajectory if match) / len(trajectory) if trajectory else 0.0
            print(f"  -> Accuracy this episode: {accuracy:.2%}, Task Success: {task_success}")
            return {"accuracy": accuracy, "success": task_success}
        else:
            num_failed = sum(1 for step in trajectory if not step['action_succeeded'])
            accuracy_heuristic = max(0, 1.0 - (num_failed / len(trajectory))) if trajectory else 0.0
            reward = 0.8 * task_success + 0.2 * accuracy_heuristic
            print(
                f"  -> Rollout Result: Task Success: {task_success}, Accuracy Heuristic: {accuracy_heuristic:.2f}, Composite Reward: {reward:.2f}")
            return {"reward": reward, "trajectory": trajectory}
    except Exception as e:
        print(f"An error occurred during episode: {e}")
        return {"reward": 0.0, "trajectory": [], "accuracy": 0.0, "success": False}
    finally:
        if env: env.close()


# --- Optimization and Synthesis Functions ---

def run_text2grad_optimization(training_tasks: List[Any]):
    print("\n" + "=" * 20 + " STAGE 1: Mixed-Task Text2Grad Optimization " + "=" * 20)
    base_template = """Your high-level goal is: {goal}
Here is a history of your previous actions:
{history}
Here are the available UI elements on the current screen:
{observation}
Based on your history and the current screen, what is your next action? If the task is complete, respond with STATUS_COMPLETE.
Respond ONLY in the format:
Reason: [Your reasoning here]
Action: [Your action here in the format CLICK("text"), OPEN_APP("App Name"), SCROLL("direction"), TYPE("text_to_type", "target_field_text"), or STATUS_COMPLETE]"""
    best_prompt, best_avg_reward = base_template, 0.0

    print("\nEvaluating baseline prompt P0 across all training tasks...")
    baseline_rewards = [run_episode(base_template, task)["reward"] for task in training_tasks]
    best_avg_reward = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0.0
    print(f"\nBaseline Average Reward for P0: {best_avg_reward:.2%}")

    for i in range(FLAGS.num_text2grad_mutations):
        print(f"\n--- Mutation {i + 1}/{FLAGS.num_text2grad_mutations} ---")
        mutated_prompt = random.choice(["Think step-by-step. ", "Be precise. "]) + best_prompt
        print(f"  -> Evaluating mutated prompt across {len(training_tasks)} tasks...")
        rewards = [run_episode(mutated_prompt, task)["reward"] for task in training_tasks]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"  -> Average Reward for Mutated Prompt: {avg_reward:.2%}")
        if avg_reward > best_avg_reward:
            best_avg_reward, best_prompt = avg_reward, mutated_prompt
            print("  -> Found a better prompt!")

    print(f"\nText2Grad Finished. Optimized Reward (P*): {best_avg_reward:.2%}")
    return best_prompt


def arpo_update_policy(base_policy: str, trajectory: List[Dict]) -> str:
    if len(trajectory) >= 2 and not trajectory[-1]['action_succeeded'] and not trajectory[-2]['action_succeeded'] and \
            trajectory[-1]['screen_fingerprint'] == trajectory[-2]['screen_fingerprint']:
        failed_action = trajectory[-1]['action_taken']
        correction = f"\n\nCRITICAL: Your previous attempts to perform actions like '{failed_action}' on this screen have failed. Re-examine the UI elements provided and choose a different, valid action."
        print("  -> ARPO Diagnosis: Invalid Action Repetition.")
        return base_policy + correction
    if len(trajectory) >= 4:
        s1, s2, s3, s4 = [t['screen_fingerprint'] for t in trajectory[-4:]]
        if s1 == s3 and s2 == s4 and s1 != s2:
            a1, a2 = trajectory[-2]['action_taken'], trajectory[-1]['action_taken']
            correction = f"\n\nCRITICAL: You are stuck in a loop between two screens with actions '{a1}' and '{a2}'. This strategy is failing. Re-read the goal and devise a new plan."
            print("  -> ARPO Diagnosis: State-Action Loop.")
            return base_policy + correction
    print("  -> ARPO Diagnosis: Strategic Stagnation.")
    return base_policy + "\n\nSTRATEGY FAILED. Your previous approach did not work. Re-read the goal carefully. Think step-by-step about a new strategy."


def synthesize_new_policy(base_policy: str, arpo_policy: str) -> str:
    print("\n" + "=" * 25 + " Stage 3: Synthesizing New Policy " + "=" * 25)
    constraints = arpo_policy.replace(base_policy, "").strip()
    if not constraints:
        print(" -> No constraints found. ARPO policy is already clean.")
        return arpo_policy

    few_shot_example = """**EXAMPLE 1:**

**BASE PROMPT:**
---
Your high-level goal is: {goal}
... (rest of prompt) ...
Respond ONLY in the format:
Reason: [Your reasoning here]
Action: [Your action here in the format ...]
---

**RAW ERROR CORRECTIONS TO ANALYZE:**
---
STRATEGY FAILED. Your previous approach did not work. Re-read the goal carefully. Think step-by-step about a new strategy.
CRITICAL: Your previous attempts to perform actions like 'CLICK("Non-existent Button")' on this screen have failed. Re-examine the UI elements provided and choose a different, valid action.
CRITICAL: You are stuck in a loop...
---

**NEW, SYNTHESIZED PROMPT:**
---
Here are three principles to follow:
1.  If your strategy is not working or you are stuck in a loop, stop and re-read the goal to devise a new plan.
2.  Only choose to interact with UI elements that are explicitly listed in the current observation.
3.  Before declaring the task complete, double-check that the goal has been fully met.

Your high-level goal is: {goal}
... (rest of prompt) ...
Respond ONLY in the format:
Reason: [Your reasoning here]
Action: [Your action here in the format ...]
---
"""
    meta_prompt = f"""You are an expert in AI agent behavior and prompt engineering.
Your task is to analyze an existing prompt and a list of error corrections, then synthesize them into a new, improved prompt.

**CRITICAL INSTRUCTIONS:**
1.  Follow the example provided to understand the task.
2.  Identify the underlying principles from the "RAW ERROR CORRECTIONS".
3.  Integrate these principles as 2-3 concise, general rules at the beginning of the "BASE PROMPT".
4.  The final output MUST be a complete, drop-in replacement for the base prompt.
5.  Crucially, the final output MUST strictly adhere to the original's structure, including the `Reason:` and `Action:` formatting and all placeholders.

{few_shot_example}

**TASK:**

**BASE PROMPT:**
---
{base_policy}
---

**RAW ERROR CORRECTIONS TO ANALYZE:**
---
{constraints}
---

**NEW, SYNTHESIZED PROMPT (must be a complete, valid prompt and nothing else):**
"""
    print(" -> Querying meta-LLM to synthesize principles...")
    synthesized_policy = get_llm_response(meta_prompt)
    synthesized_policy = re.sub(r'^---\s*$', '', synthesized_policy, flags=re.MULTILINE).strip()
    if "{goal}" not in synthesized_policy:  # Basic validity check
        print(" -> Synthesis failed to produce a valid prompt. Returning the un-synthesized ARPO policy.")
        return arpo_policy
    print(" -> Synthesis complete.")
    return synthesized_policy


# --- Main Execution ---

def main(argv):
    if FLAGS.gcp_api_key: os.environ["GCP_API_KEY"] = FLAGS.gcp_api_key
    task_registry, aw_registry = registry.TaskRegistry(), registry.TaskRegistry().get_registry(
        registry.TaskRegistry.ANDROID_WORLD_FAMILY)

    training_task_names = ['ContactsAddContact', 'SystemWifiTurnOff', 'SystemBrightnessMax']
    training_tasks = [aw_registry[name] for name in training_task_names if name in aw_registry]
    if not training_tasks:
        print("FATAL: No valid training tasks found. Exiting.")
        return

    gemini_baseline_policy = "Your goal is: {goal}\nHere are the UI elements on the screen:\n{observation}\nWhat is the next action? Respond ONLY in the format: Action: CLICK(\"element_text\")."
    text2grad_policy = run_text2grad_optimization(training_tasks)

    print("\n" + "=" * 25 + " Stage 2: ARPO Mixed-Task Training " + "=" * 25)
    current_policy, reward_history = text2grad_policy, []
    total_iterations = len(training_tasks) * FLAGS.num_training_iterations_per_task

    for i in range(total_iterations):
        training_task_class = training_tasks[i % len(training_tasks)]  # Cycle through tasks
        task_name = [k for k, v in aw_registry.items() if v == training_task_class][0]
        print(f"\n--- ARPO Iteration {i + 1}/{total_iterations} on '{task_name}' ---")
        result = run_episode(current_policy, training_task_class)
        reward_history.append(result["reward"])
        if result["reward"] < 1.0 and result["trajectory"]:
            print("  -> Policy is suboptimal. Updating with ARPO...")
            current_policy = arpo_update_policy(current_policy, result["trajectory"])
        elif result["reward"] == 1.0:
            print("  -> Policy is optimal. Retaining learned policy.")
    arpo_final_policy = current_policy

    synthesized_policy = synthesize_new_policy(text2grad_policy, arpo_final_policy)

    print("\n" + "=" * 25 + " FINAL MULTI-TASK BENCHMARK " + "=" * 25)
    benchmark_task_names = ['ContactsAddContact', 'SystemWifiTurnOff', 'SystemBrightnessMax']
    benchmark_results = {}
    policies_to_benchmark = {
        "Gemini_Baseline": gemini_baseline_policy,
        "Text2Grad_Optimized": text2grad_policy,
        "ARPO_Optimized": arpo_final_policy,
        "ARPO_Synthesized": synthesized_policy
    }

    for name, policy in policies_to_benchmark.items():
        all_task_successes, all_task_accuracies = [], []
        print(f"\n--- Benchmarking Policy: '{name}' ---")
        for task_name in benchmark_task_names:
            print(f"\n  -- Running Task: {task_name} --")
            task_class = aw_registry.get(task_name)
            if not task_class: continue
            task_successes, task_accuracies = 0, 0.0
            for i in range(FLAGS.num_benchmark_episodes):
                print(f"    - Episode {i + 1}/{FLAGS.num_benchmark_episodes}")
                result = run_episode(policy, task_class, is_benchmark=True)
                if result["success"]: task_successes += 1
                task_accuracies += result["accuracy"]
            all_task_successes.append(task_successes / FLAGS.num_benchmark_episodes)
            all_task_accuracies.append(task_accuracies / FLAGS.num_benchmark_episodes)
        if all_task_successes:
            benchmark_results[name] = {
                "Task Completion Rate": sum(all_task_successes) / len(all_task_successes),
                "Action Accuracy": sum(all_task_accuracies) / len(all_task_accuracies)
            }
        else:
            benchmark_results[name] = {"Task Completion Rate": 0.0, "Action Accuracy": 0.0}

    print("\n" + "=" * 25 + " FINAL RESULTS " + "=" * 25)
    print("--- Interpretability: Final Prompt Policies ---")
    for name, policy in policies_to_benchmark.items():
        print(f"\nPolicy: {name}\n" + "-" * 20 + f"\n{policy}\n" + "-" * 20)
    print("\n--- Performance Metrics (Averaged Across All Benchmark Tasks) ---")
    for name, results in benchmark_results.items():
        print(f"Policy: {name}")
        print(f"  -> Task Completion Rate: {results['Task Completion Rate']:.2%}")
        print(f"  -> Action Accuracy: {results['Action Accuracy']:.2%}")
    print("\n--- Reward Curve (ARPO Mixed-Task Training) ---")
    print(reward_history)

    results_data = {
        "prompt_improvement": policies_to_benchmark,
        "reward_curve": reward_history,
        "benchmark_results": benchmark_results,
        "benchmark_tasks": benchmark_task_names
    }
    os.makedirs("results", exist_ok=True)
    with open("results/arpo_full_pipeline_results_synthesis.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print("\nSaved final pipeline results to results/arpo_full_pipeline_results_synthesis.json")


if __name__ == "__main__":
    app.parse_flags_with_usage(sys.argv)
    main(sys.argv)

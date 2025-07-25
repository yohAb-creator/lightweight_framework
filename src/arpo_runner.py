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

# --- Import the policy from the Text2Grad stage ---
# Note: Ensure text2grad_opt.py is in a reachable path, e.g., src/
try:
    from src.text2grad_opt import run_text2grad_optimization
except ImportError:
    # Define a fallback function if the import fails
    def run_text2grad_optimization():
        print("WARNING: Could not import run_text2grad_optimization. Using a default baseline prompt.")
        return """Your high-level goal is: {goal}
Here is a history of your previous actions:
{history}
Here are the available UI elements on the current screen:
{observation}
Based on your history and the current screen, what is your next action? If the task is complete, respond with STATUS_COMPLETE.
Respond ONLY in the format:
Reason: [Your reasoning here]
Action: [Your action here in the format CLICK("text"), OPEN_APP("App Name"), SCROLL("direction"), TYPE("text_to_type", "target_field_text"), or STATUS_COMPLETE]"""

# --- Suppress verbose informational logs ---
logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# --- Flag Definitions ---
flags.DEFINE_string('task_name', 'ContactsAddContact', 'A representative android_world task to use for Text2Grad.')
flags.DEFINE_integer('num_training_iterations_per_task', 10,
                     'Number of RL iterations to run for each task in the training set.')
flags.DEFINE_integer('num_benchmark_episodes', 3,
                     'Number of episodes to run for each task in the final benchmark.')
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
    # MR. TERRIFIC'S FIX: Upgraded model to Gemini 2.5 Pro.
    llm = infer.GeminiGcpWrapper('gemini-2.5-pro')
    response, _, _ = llm.predict(prompt)
    return response


def parse_action_from_llm_response(response_text: str, ui_elements: list[Any]) -> dict[str, Any] | None:
    """Extracts an action dictionary from the LLM's full response."""
    action_str = response_text
    if "Action:" in response_text:
        action_str = response_text.split("Action:")[-1].strip()

    click_match = re.search(r'CLICK\("(.+?)"\)', action_str)
    if click_match:
        target_text = click_match.group(1)
        for i, element in enumerate(ui_elements):
            label = (element.content_description or element.text or "").strip()
            if target_text.lower() == label.lower():
                return {"action_type": "click", "index": i}
        return {"action_type": "click", "target_text": target_text}

    type_match = re.search(r'TYPE\("(.+?)", "(.+?)"\)', action_str)
    if type_match:
        text_to_type = type_match.group(1)
        target_field_text = type_match.group(2)
        for i, element in enumerate(ui_elements):
            label = (element.content_description or element.text or "").strip()
            if target_field_text.lower() == label.lower() and element.is_editable:
                return {"action_type": "input_text", "text": text_to_type, "index": i}
        return {"action_type": "input_text", "text": text_to_type, "target_text": target_field_text}

    open_app_match = re.search(r'OPEN_APP\("(.+?)"\)', action_str)
    if open_app_match:
        app_name = open_app_match.group(1)
        return {"action_type": "open_app", "app_name": app_name}

    scroll_match = re.search(r'SCROLL\("(.+?)"\)', action_str)
    if scroll_match:
        direction = scroll_match.group(1).lower()
        if direction in ["up", "down", "left", "right"]:
            return {"action_type": "scroll", "direction": direction}

    if "navigate_back" in action_str.lower():
        return {"action_type": "navigate_back"}

    if "STATUS_COMPLETE" in action_str.upper():
        return {"action_type": "status", "goal_status": "complete"}

    return None


class SimpleAgent(base_agent.EnvironmentInteractingAgent):
    """A simple, memory-enabled agent that uses a prompt policy."""

    def __init__(self, env: interface.AsyncEnv, prompt_policy: str):
        super().__init__(env)
        self.prompt_policy = prompt_policy
        self.history = []

    def reset(self, go_home: bool = False) -> None:
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

        is_done = False
        action_succeeded = False
        if action_params:
            if action_params.get("action_type") == "status":
                is_done = True
                action_succeeded = True
            else:
                try:
                    if "index" in action_params:
                        action = json_action.JSONAction(**action_params)
                        self.env.execute_action(action)
                        self.history.append(action_summary)
                        action_succeeded = True
                    else:
                        self.history.append(f"Action failed: Element for '{action_summary}' not found.")
                        action_succeeded = False
                except Exception as e:
                    self.history.append(f"Action failed with error: {e}")
                    action_succeeded = False
        else:
            self.history.append(f"Could not parse LLM response: {llm_response}")

        step_data = {
            "screen_fingerprint": screen_fingerprint,
            "observation": observation_text,
            "action_taken": action_summary,
            "action_params": action_params,
            "action_succeeded": action_succeeded,
            "llm_response": llm_response
        }
        return base_agent.AgentInteractionResult(done=is_done, data=step_data)


def run_episode_for_reward_and_accuracy(prompt_policy: str, task_class) -> Tuple[float, List[Dict]]:
    """
    This function runs a single episode, calculates a composite reward
    based on both task success and action accuracy, and returns the trajectory.
    """
    trajectory = []
    env = None
    try:
        env = env_launcher.load_and_setup_env(
            adb_path=FLAGS.adb_path, console_port=FLAGS.console_port, grpc_port=FLAGS.grpc_port
        )
        task_instance = task_class(params=task_class.generate_random_params())
        goal = task_instance.goal
        print(f"Goal: {goal}")
        task_instance.initialize_task(env)

        agent = SimpleAgent(env, prompt_policy)
        agent.reset(go_home=True)
        time.sleep(2)

        for _ in range(FLAGS.max_steps_per_episode):
            result = agent.step(goal)
            trajectory.append(result.data)
            if result.done:
                break
            time.sleep(1)

        task_success_reward = task_instance.is_successful(env)

        num_failed_actions = sum(1 for step in trajectory if not step['action_succeeded'])
        accuracy_reward = max(0, 1.0 - (num_failed_actions / len(trajectory))) if trajectory else 0.0

        composite_reward = 0.8 * task_success_reward + 0.2 * accuracy_reward

        print(
            f"  -> Rollout Result: Task Success: {task_success_reward}, Accuracy Heuristic: {accuracy_reward:.2f}, Composite Reward: {composite_reward:.2f}")
        return composite_reward, trajectory

    except Exception as e:
        print(f"An error occurred during reward episode: {e}")
        return 0.0, trajectory
    finally:
        if env:
            env.close()


def are_actions_equivalent(our_action_dict, expert_action_dict, ui_elements):
    """
    A more intelligent function to check if two actions are semantically the same,
    even if the indices or action formats are different.
    """
    if not our_action_dict or not expert_action_dict:
        return False

    our_action_type = our_action_dict.get("action_type")
    expert_action_type = expert_action_dict.get("action_type")

    if our_action_type != expert_action_type:
        return False

    # Universal actions
    if expert_action_type in ["status", "navigate_back", "navigate_home"]:
        return our_action_dict == expert_action_dict

    # Click actions
    if expert_action_type == "click":
        our_idx = our_action_dict.get("index")
        expert_idx = expert_action_dict.get("index")

        if our_idx is not None and expert_idx is not None:
            if our_idx == expert_idx:
                return True
            if our_idx < len(ui_elements) and expert_idx < len(ui_elements):
                our_text = (ui_elements[our_idx].content_description or ui_elements[our_idx].text or "").strip().lower()
                expert_text = (ui_elements[expert_idx].content_description or ui_elements[
                    expert_idx].text or "").strip().lower()
                if our_text and our_text == expert_text:
                    return True

        if "target_text" in our_action_dict and expert_idx is not None:
            if expert_idx < len(ui_elements):
                expert_text = (ui_elements[expert_idx].content_description or ui_elements[
                    expert_idx].text or "").strip().lower()
                if our_action_dict["target_text"].lower() == expert_text:
                    return True

    # Type actions
    if expert_action_type == "input_text":
        return our_action_dict.get("text") == expert_action_dict.get("text")

    # Scroll actions
    if expert_action_type == "scroll":
        return our_action_dict.get("direction") == expert_action_dict.get("direction")

    return False


def run_episode_for_benchmark_accuracy(prompt_policy: str, task_class) -> Tuple[float, bool]:
    """The original accuracy function, now using a robust comparison logic."""
    env = None
    try:
        env = env_launcher.load_and_setup_env(
            adb_path=FLAGS.adb_path, console_port=FLAGS.console_port, grpc_port=FLAGS.grpc_port
        )
        task_instance = task_class(params=task_class.generate_random_params())
        goal = task_instance.goal
        task_instance.initialize_task(env)
        time.sleep(2)

        expert_agent = t3a.T3A(env, infer.GeminiGcpWrapper('gemini-2.5-pro'))
        episode_log = []
        is_done = False

        for step_num in range(FLAGS.max_steps_per_episode):
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

            print(f"  Step {step_num + 1}: Our Action: {our_action_dict} vs. Expert Action: {expert_action_dict}")

            is_match = are_actions_equivalent(our_action_dict, expert_action_dict, ui_elements)
            episode_log.append(is_match)

            if expert_response.done:
                is_done = True
                break
            time.sleep(1)

        correct_steps = sum(1 for match in episode_log if match)
        step_accuracy = correct_steps / len(episode_log) if episode_log else 0.0
        task_success = is_done and task_instance.is_successful(env) == 1.0

        print(f"  -> Accuracy this episode: {step_accuracy:.2%}, Task Success: {task_success}")
        return step_accuracy, task_success
    except Exception as e:
        print(f"An error occurred during accuracy episode: {e}")
        return 0.0, False
    finally:
        if env:
            env.close()


def arpo_update_policy(base_policy: str, trajectory: List[Dict]) -> str:
    """Performs a truly action-aware update to the prompt policy by analyzing the trajectory for common failure patterns."""
    if len(trajectory) >= 2 and not trajectory[-1]['action_succeeded'] and not trajectory[-2]['action_succeeded'] and \
            trajectory[-1]['screen_fingerprint'] == trajectory[-2]['screen_fingerprint']:
        failed_action = trajectory[-1]['action_taken']
        correction = f"\n\nCRITICAL: Your previous attempts to perform actions like '{failed_action}' on this screen have failed. Re-examine the UI elements provided and choose a different, valid action."
        print("  -> ARPO Diagnosis: Invalid Action Repetition. Adding corrective instruction.")
        return base_policy + correction

    if len(trajectory) >= 4:
        s1, s2, s3, s4 = [t['screen_fingerprint'] for t in trajectory[-4:]]
        if s1 == s3 and s2 == s4 and s1 != s2:
            a1, a2 = trajectory[-2]['action_taken'], trajectory[-1]['action_taken']
            correction = f"\n\nCRITICAL: You are stuck in a loop, toggling between two screens with actions '{a1}' and '{a2}'. This strategy is failing. Re-read the goal and devise a new plan."
            print("  -> ARPO Diagnosis: State-Action Loop. Adding negative constraint.")
            return base_policy + correction

    print("  -> ARPO Diagnosis: Strategic Stagnation. Forcing meta-cognitive step.")
    return base_policy + "\n\nSTRATEGY FAILED. Your previous approach did not work. Re-read the goal carefully. Think step-by-step about a new strategy."


def main(argv):
    if FLAGS.gcp_api_key:
        os.environ["GCP_API_KEY"] = FLAGS.gcp_api_key

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    # --- Stage 1 & 2: Text2Grad & ARPO Training ---
    print("\n" + "=" * 25 + " Stage 1 & 2: Text2Grad & ARPO Training " + "=" * 25)

    gemini_baseline_policy = "Your goal is: {goal}\nHere are the UI elements on the screen:\n{observation}\nWhat is the next action? Respond ONLY in the format: Action: CLICK(\"element_text\")."
    text2grad_policy = run_text2grad_optimization()

    # Implement Structured Mixed-Task Training with a focused set of 3 tasks.
    training_task_names = [
        'ContactsAddContact',
        # 'BrowserDraw',
        'SystemWifiTurnOff',
        # 'ExpenseAddMultiple',
        # 'SimpleCalendarAddOneEvent',
        # 'SimpleCalendarAddRepeatingEvent',
        # 'SimpleSmsSend',
        # 'SystemBluetoothTurnOff',
        # 'MarkorCreateNote',
        'SystemBrightnessMax'
    ]
    training_tasks = [aw_registry[name] for name in training_task_names if name in aw_registry]
    if not training_tasks:
        print("FATAL: No valid training tasks found. Exiting.")
        return

    current_policy = text2grad_policy
    reward_history = []
    total_iterations = len(training_tasks) * FLAGS.num_training_iterations_per_task
    iteration_count = 0

    # Iterate through each task in the training set
    for training_task_class in training_tasks:
        task_name = [k for k, v in aw_registry.items() if v == training_task_class][0]
        print("\n" + "=" * 25 + f" Training on Task: '{task_name}' " + "=" * 25)

        # Run the specified number of iterations for this task
        for i in range(FLAGS.num_training_iterations_per_task):
            iteration_count += 1
            print(
                f"\n--- ARPO Iteration {iteration_count}/{total_iterations} (Task: '{task_name}', Run {i + 1}/{FLAGS.num_training_iterations_per_task}) ---")

            reward, trajectory = run_episode_for_reward_and_accuracy(current_policy, training_task_class)
            reward_history.append(reward)
            if reward < 1.0 and trajectory:
                print("  -> Policy is suboptimal. Updating with ARPO...")
                current_policy = arpo_update_policy(current_policy, trajectory)
            elif reward == 1.0:
                print("  -> Policy is optimal. Exploiting current prompt.")
                current_policy = text2grad_policy
    arpo_final_policy = current_policy

    # --- Final benchmark across the three training tasks ---
    print("\n" + "=" * 25 + " FINAL MULTI-TASK BENCHMARK " + "=" * 25)

    benchmark_task_names = [
        'ContactsAddContact',
        # 'BrowserDraw',
        'SystemWifiTurnOff',
        # 'ExpenseAddMultiple',
        # 'SimpleCalendarAddOneEvent',
        # 'SimpleCalendarAddRepeatingEvent',
        # 'SimpleSmsSend',
        # 'SystemBluetoothTurnOff',
        # 'MarkorCreateNote',
        'SystemBrightnessMax'
    ]

    benchmark_results = {}
    policies_to_benchmark = {
        "Gemini_Baseline": gemini_baseline_policy,
        "Text2Grad_Optimized": text2grad_policy,
        "ARPO_Optimized": arpo_final_policy
    }

    for name, policy in policies_to_benchmark.items():
        all_task_successes = []
        all_task_accuracies = []
        print(f"\n--- Benchmarking Policy: '{name}' ---")

        for task_name in benchmark_task_names:
            print(f"\n  -- Running Task: {task_name} --")
            task_class = aw_registry.get(task_name)
            if not task_class:
                print(f"    - WARNING: Task '{task_name}' not found in registry. Skipping.")
                continue

            task_successes = 0
            task_accuracies = 0.0
            for i in range(FLAGS.num_benchmark_episodes):
                print(f"    - Episode {i + 1}/{FLAGS.num_benchmark_episodes}")
                step_accuracy, success = run_episode_for_benchmark_accuracy(policy, task_class)
                if success:
                    task_successes += 1
                task_accuracies += step_accuracy

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
        print(f"\nPolicy: {name}")
        print("-" * 20)
        print(policy)
        print("-" * 20)

    print("\n--- Performance Metrics (Averaged Across All Benchmark Tasks) ---")
    for name, results in benchmark_results.items():
        print(f"Policy: {name}")
        print(f"  -> Task Completion Rate: {results['Task Completion Rate']:.2%}")
        print(f"  -> Action Accuracy: {results['Action Accuracy']:.2%}")

    print("\n--- Reward Curve (ARPO Mixed-Task Training) ---")
    print(reward_history)

    results_data = {
        "prompt_improvement": {
            "gemini_baseline": gemini_baseline_policy,
            "text2grad_optimized": text2grad_policy,
            "arpo_optimized": arpo_final_policy
        },
        "reward_curve": reward_history,
        "benchmark_results": benchmark_results,
        "benchmark_tasks": benchmark_task_names
    }
    os.makedirs("results", exist_ok=True)
    with open("../results/arpo_full_pipeline_results_3tasks.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print("\nSaved final pipeline results to results/arpo_full_pipeline_results_3tasks.json")


if __name__ == "__main__":
    app.parse_flags_with_usage(sys.argv)
    main(sys.argv)

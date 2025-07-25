import os
import re
import random
import time
import json
import sys
from typing import Any, List, Tuple
from absl import app, flags, logging
from android_world import registry
from android_world.env import env_launcher, json_action, interface
from android_world.agents import infer, base_agent

# --- Suppress verbose informational logs ---
logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# --- Flag Definitions ---
flags.DEFINE_string('task_name_tg', 'ContactsAddContact', 'The android_world task to use for evaluation.')
flags.DEFINE_integer('num_mutations_tg', 3, 'Number of prompt variations to generate and test.')
flags.DEFINE_string('gcp_api_key_tg', os.environ.get("GCP_API_KEY"), 'Your Google Cloud API Key.')
flags.DEFINE_string('adb_path_tg', "C:\\Users\\yohan\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe",
                    'Path to ADB.')
flags.DEFINE_integer('console_port_tg', 5554, 'Emulator console port.')
flags.DEFINE_integer('grpc_port_tg', 8554, 'Emulator gRPC port.')
flags.DEFINE_integer('max_steps_tg', 20, 'Maximum steps for the agent per episode.')


def get_llm_response(prompt: str) -> str:
    """Generic wrapper to call the LLM."""
    if not FLAGS.gcp_api_key_tg:
        return "ERROR: API Key not set"
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
        return None

    type_match = re.search(r'TYPE\("(.+?)", "(.+?)"\)', action_str)
    if type_match:
        text_to_type = type_match.group(1)
        target_field_text = type_match.group(2)
        for i, element in enumerate(ui_elements):
            label = (element.content_description or element.text or "").strip()
            if target_field_text.lower() == label.lower() and element.is_editable:
                return {"action_type": "input_text", "text": text_to_type, "index": i}
        return None

    open_app_match = re.search(r'OPEN_APP\("(.+?)"\)', action_str)
    if open_app_match:
        app_name = open_app_match.group(1)
        return {"action_type": "open_app", "app_name": app_name}

    scroll_match = re.search(r'SCROLL\("(.+?)"\)', action_str)
    if scroll_match:
        direction = scroll_match.group(1).lower()
        if direction in ["up", "down", "left", "right"]:
            return {"action_type": "scroll", "direction": direction}

    if "STATUS_COMPLETE" in action_str.upper():
        return {"action_type": "status", "goal_status": "complete"}

    return None


def mutate_prompt(base_prompt: str) -> str:
    """Creates a simple variation of a given prompt by prepending an instruction."""
    variations = ["Think step-by-step. ", "Be precise. ", "What is the most logical next action? "]
    prefix = random.choice(variations)
    return prefix + base_prompt


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

        clickable_elements = [f"- \"{(e.content_description or e.text or '').strip()}\"" for e in ui_elements if
                              e.is_clickable and (e.content_description or e.text)]
        observation_text = "\n".join(clickable_elements) if clickable_elements else "The screen is empty."
        history_text = "\n".join([f"Step {i + 1}: {h}" for i, h in enumerate(self.history)])

        current_prompt = self.prompt_policy.format(goal=goal, observation=observation_text, history=history_text)

        llm_response = get_llm_response(current_prompt)
        print(f"    -> LLM Reasoning & Action:\n{llm_response}")
        action_params = parse_action_from_llm_response(llm_response, ui_elements)

        is_done = False
        if action_params:
            if action_params.get("action_type") == "status":
                is_done = True
            else:
                try:
                    action = json_action.JSONAction(**action_params)
                    self.env.execute_action(action)
                    action_summary = llm_response.split('Action:')[-1].strip()
                    self.history.append(action_summary)
                except Exception as e:
                    self.history.append(f"Action failed with error: {e}")
        else:
            self.history.append(f"Could not parse LLM response: {llm_response}")

        return base_agent.AgentInteractionResult(done=is_done, data={})


def reward_function(prompt_template: str, task_class) -> float:
    """Runs an agent and returns 1.0 for success, 0.0 for failure."""
    env = None
    try:
        env = env_launcher.load_and_setup_env(
            adb_path=FLAGS.adb_path_tg, console_port=FLAGS.console_port_tg, grpc_port=FLAGS.grpc_port_tg
        )
        task_instance = task_class(params=task_class.generate_random_params())
        goal = task_instance.goal
        print(f"Goal: {goal}")

        task_instance.initialize_task(env)

        agent = SimpleAgent(env, prompt_template)
        agent.reset(go_home=True)
        time.sleep(2)

        for i in range(FLAGS.max_steps_tg):
            result = agent.step(goal)
            if result.done:
                break
            time.sleep(1)

        if task_instance.is_successful(env) == 1.0:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        print(f"An error occurred during reward calculation: {e}")
        return 0.0
    finally:
        if env:
            env.close()


def run_text2grad_optimization():
    """Runs the Text2Grad optimization loop and returns the best prompt."""
    print("\n" + "=" * 20 + " STAGE 2: Text2Grad Optimization " + "=" * 20)

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    task_class = aw_registry[FLAGS.task_name_tg]

    base_template = """Your high-level goal is: {goal}
Here is a history of your previous actions:
{history}
Here are the available UI elements on the current screen:
{observation}
Based on your history and the current screen, what is your next action? If the task is complete, respond with STATUS_COMPLETE.
Respond ONLY in the format:
Reason: [Your reasoning here]
Action: [Your action here in the format CLICK("text"), OPEN_APP("App Name"), SCROLL("direction"), TYPE("text_to_type", "target_field_text"), or STATUS_COMPLETE]"""

    print("\nEvaluating baseline prompt P0...")
    reward_p0 = reward_function(base_template, task_class)
    print(f"\nBaseline Reward for P0: {reward_p0:.2%}")

    best_prompt = base_template
    best_reward = reward_p0

    for i in range(FLAGS.num_mutations_tg):
        print(f"\n--- Mutation {i + 1}/{FLAGS.num_mutations_tg} ---")
        mutated_prompt = mutate_prompt(base_template)
        reward_mutated = reward_function(mutated_prompt, task_class)
        print(f"  -> Reward for Mutated Prompt: {reward_mutated:.2%}")

        if reward_mutated > best_reward:
            best_reward = reward_mutated
            best_prompt = mutated_prompt
            print("  -> Found a better prompt!")

    print("\n" + "=" * 25 + " Text2Grad Finished " + "=" * 25)
    print(f"Optimized Reward (P*): {best_reward:.2%}")
    return best_prompt


def main(argv):
    if FLAGS.gcp_api_key_tg:
        os.environ["GCP_API_KEY"] = FLAGS.gcp_api_key_tg
    run_text2grad_optimization()


if __name__ == "__main__":
    app.parse_flags_with_usage(sys.argv)
    main(sys.argv)

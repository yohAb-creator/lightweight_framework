# LLM Agent Evaluation Framework for Android World

This project contains a full-stack research pipeline for evaluating and optimizing Large Language Model (LLM) agents on the `android_world` benchmark. It is submitted as a solution to the QualGent Research Engineer coding challenges.

The project is divided into two main parts:

1. **Initial Evaluation Framework:** A system for benchmarking a custom-prompted agent against the expert `t3a` agent to measure baseline `step_accuracy`.

2. **Advanced Prompt Optimization Pipeline:** A three-stage pipeline that uses a multimodal model (Gemini 2.5), simulated black-box optimization (Text2Grad), and reinforcement learning (ARPO) to automatically discover and refine a high-performing prompt policy.

## 1. Setup

### Prerequisites

* Python 3.11+

* [Anaconda/Conda](https://www.anaconda.com/download)

* [Android Studio](https://developer.android.com/studio) with a configured Android Virtual Device (AVD).

* A Google Cloud Platform (GCP) API key with the "Generative Language API" enabled.

### Installation Steps

1. **Clone the `android_world` Repository:**
   This project relies on the `android_world` library. Clone it into a separate directory.

git clone https://github.com/google-research/android_world.git


2. **Create and Activate Conda Environment:**

conda create -n android_world python=3.11
conda activate android_world


3. **Install `android_world` Library:**
Navigate to the cloned `android_world` directory and install it in editable mode. This is required to apply a necessary workaround.

cd path/to/android_world
pip install -e .


4. **Install Project Dependencies:**
Navigate back to this project's root directory and install its requirements.

cd path/to/your/evaluation_framework
pip install google-generativeai Pillow absl-py


5. **Set API Key:**
Set your GCP API key as an environment variable in your terminal session.

In PowerShell

$env:GCP_API_KEY = "your-gcp-api-key"


### **Required Workaround for Windows**

The `android_world` library has a known issue where its setup process fails on Windows. To resolve this, a one-line code modification is required in the installed library.

* **File to Edit:** `.../android_world/android_world/env/android_world_controller.py`

* **Modification:** In the `__init__` method of the `AndroidWorldController` class, comment out the line that calls `apply_a11y_forwarder_app_wrapper`.

self._env = apply_a11y_forwarder_app_wrapper(

self._env, adb_controller, grpc_port

)


## 2. Usage

### Step 1: Launch the Emulator

Before running any scripts, you must start the Android emulator from a terminal with the correct gRPC port.

In PowerShell (replace YourName)

$emulator_path = "C:\Users\YourName\AppData\Local\Android\Sdk\emulator\emulator.exe"
$avd_name = "AndroidWorldAvd"
& $emulator_path -avd $avd_name -no-snapshot -grpc 8554


### Step 2: Running the Scripts

Navigate into the `src/` directory to run the main scripts.

cd src


#### Part 1: Initial Evaluation (First Challenge)

To run the initial evaluation comparing a custom prompter to the expert `t3a` agent:

python evaluation.py --num_episodes=10 --prompt_variant=few_shot


#### Part 2: Advanced Prompt Optimization Pipeline

**Stage 1: `gemini_prompting.py` (Vision-to-Prompt)**

This script generates a baseline prompt from a screenshot. Make sure you have a screenshot saved in the `screenshots/` directory at the project root.

python gemini_prompting.py --image="../screenshots/contacts_page.png" --goal="Create a new contact"


**Stage 2: `text2grad_opt.py` (Black-Box Optimization)**

This script runs a simulated Text2Grad loop to find an improved prompt.

python text2grad_opt.py


**Stage 3: `arpo_runner.py` (Reinforcement Learning)**

This is the main script for the advanced challenge. It runs the full ARPO training loop and the final comparative benchmark.

python arpo_runner.py


### Step 3: View Results

The scripts will print summaries to the console and sav

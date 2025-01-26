
# SAFEVAR Tool: Overview and File Placement

The `SAFEVAR Tool` provides essential modules for implementing, evaluating, and managing scenarios in the CARLA simulator. Below is a detailed description of each file and its purpose, along with instructions on where to place them in the project directory.

---

## Files in the `SAFEVAR Tool` Directory

### 1. **`SAFEVAR_nocrash_scenario.py`**
- **Purpose**: Implements the testing scenarios for the NoCrash benchmark, allowing agents to interact in controlled environments for training purposes.
- **Key Features**:
  - Configures ego vehicles and non-player characters (NPCs) for testing given VCS under specific scenarios.
  - Uses CARLA's background activity simulation (e.g., traffic and pedestrians).
- **Target Placement**: Move this file to the following directory:
  ```
  leaderboard/scenarios/
  ```

---

### 2. **`SAFEVAR_nocrash_eval_scenario.py`**
- **Purpose**: Defines the `NoCrashEvalScenario`, an evaluation framework for testing autonomous agents in CARLA under various conditions (e.g., weather, traffic, routes).
- **Key Features**:
  - Sets up routes between predefined waypoints.
  - Initializes NPCs, including vehicles and pedestrians.
  - Configures weather and other environmental factors.
  - Evaluates agent performance using metrics like TET, TIT, safetyDegree.
- **Target Placement**: Move this file to the following directory:
  ```
  leaderboard/scenarios/
  ```

---

### 3. **`SAFEVAR_problem.py`**
- **Purpose**: Defines the `CarlaProblem`, an optimization problem to explore the effects of different VCS on scenario outcomes.
- **Key Features**:
  - Configurable optimization objectives (e.g., minimizing parameter changes or improving driving safety metrics).
  - Interfaces with above mentioned modules to collect results.
- **Target Placement**: Move this file to the project base path

---

## Example Directory Structure

After moving the files to their appropriate locations, your project directory should look like this:

```
SAFEVAR Tool/
│
├── leaderboard/
│   ├── scenarios/
│       ├── SAFEVAR_nocrash_scenario.py
│       ├── SAFEVAR_nocrash_eval_scenario.py
├── SAFEVAR_problem.py
├── requirements.txt
├── README.md
└── ...
```

---

## Instructions for Use

### Step 1: Move the Files
Place the files as described in the "Target Placement" section above.

### Step 2: Install Dependencies
Run the following command to install the required Python libraries(depends on the selected ADS):
```bash
pip install -r requirements.txt
pip install jmetalpy
```

### Step 3: Run the Evaluation Scenario
Use the following command to run the `NoCrashEvalScenario`:
```bash
python nocrash_eval.py
```

### Step 4: Modify or Extend
- If you want to add more scenarios, extend `SAFEVAR_nocrash_scenario.py` or `SAFEVAR_nocrash_eval_scenario.py` and place them in the `leaderboard/scenarios/` directory.

---
- # Optimal Qubit Routing Using Reinforcement Learning
  
  ## Overview
  
  Quantum computers use physical qubits connected through a specific hardware topology. However, a quantum circuit may require interactions between qubits that are not directly connected.
  
  This creates a problem known as **Qubit Routing**.
  
  This project explores the use of **Reinforcement Learning (RL)** to optimize qubit routing. An RL agent learns to select SWAP operations to move logical qubits across the hardware topology while satisfying connectivity constraints and reducing routing overhead.
  
  The goal is to find efficient routing strategies that minimize circuit overhead, particularly the number of additional SWAP gates required.
  
  ## Objectives
  
  - Implement a Reinforcement Learning-based approach for quantum circuit routing.
  - Model qubit routing as a Markov Decision Process (MDP).
  - Train an agent to select suitable SWAP operations.
  - Optimize routing decisions based on the hardware connectivity graph.
  - Evaluate routing performance using circuit metrics such as SWAP count and circuit depth.
  - Visualize the routing process and resulting qubit mappings.
  
  ## Project Structure
  
  ```
  Optimal-Qubit-Routing-using-RL/
  │
  ├── main.py              # Main execution / RL workflow
  ├── routing_app.py       # Routing application and visualization
  │
  ├── routing/             # Qubit routing implementation
  ├── tests/               # Unit tests
  ├── runs/                # Experiment outputs and results
  │
  ├── README.md            # Project documentation
  └── requirements.txt     # Python dependencies
  ```
  
  ## Technologies Used
  
  - Python
  - Reinforcement Learning
  - Deep Q-Networks (DQN)
  - Quantum Computing
  - Qubit Routing
  - Graph-based Topology Representation
  - PyTorch
  - Qiskit (if used by the installed implementation)
  
  ## Installation
  
  ### 1. Clone the repository
  
  ```
  git clone https://github.com/pAvAn-SaBu/Optimal-Qubit-Routing-using-RL.git
  cd Optimal-Qubit-Routing-using-RL
  ```
  
  ### 2. Create a virtual environment
  
  ```
  python -m venv venv
  ```
  
  Activate the environment:
  
  **Linux / macOS:**
  
  ```
  source venv/bin/activate
  ```
  
  **Windows:**
  
  ```
  venv\Scripts\activate
  ```
  
  ### 3. Install dependencies
  
  ```
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
  
  ## Running the Project
  
  ### Run the main RL workflow
  
  ```
  python main.py
  ```
  
  ### Launch the routing application
  
  ```
  python routing_app.py
  ```
  
  The routing application provides an interface for exploring the qubit routing implementation and visualizing routing results, depending on the configured functionality.
  
  ## Running Tests
  
  The project uses `pytest` for testing.
  
  Run all tests using:
  
  ```
  pytest
  ```
  
  For verbose test output:
  
  ```
  pytest -v
  ```
  
  ## Experimental Results
  
  The `runs/` directory contains experiment outputs and results generated during the execution of the project.
  
  Performance can be evaluated using routing-related metrics, including:
  
  - Number of SWAP operations.
  - Circuit depth.
  - Routing overhead.
  - Qubit mapping and routing performance.
  
  ## Future Improvements
  
  - Explore additional RL algorithms for qubit routing.
  - Improve reward function design.
  - Evaluate performance across different hardware topologies.
  - Compare RL-based routing against conventional routing heuristics.
  - Improve generalization across different quantum circuit configurations.
  
  ## Authors
  
  Developed as an academic project exploring the application of Reinforcement Learning to quantum circuit optimization.
  
  **Project:** Optimal Qubit Routing Using Reinforcement Learning

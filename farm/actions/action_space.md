In reinforcement learning (especially in environments built with Gym-like APIs), **action spaces** define the set of all possible actions an agent can take. These action spaces are structured to support both discrete and continuous domains and can be composed into more complex spaces.

Here are the **core action space types**, with examples and use cases:

---

## 🔹 1. `Discrete(n)`

* **Meaning**: Integer actions from `0` to `n-1`
* **Use case**: Selecting one from a finite set of actions (e.g., move left, right, up, down)
* **Example**:

  ```python
  from gym.spaces import Discrete
  action_space = Discrete(4)  # actions: 0, 1, 2, 3
  ```

---

## 🔹 2. `Box(low, high, shape, dtype)`

* **Meaning**: Continuous n-dimensional action space with bounds
* **Use case**: Control tasks like robotic joint angles, car acceleration
* **Example**:

  ```python
  from gym.spaces import Box
  action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
  ```

---

## 🔹 3. `MultiBinary(n)`

* **Meaning**: `n`-dimensional binary vector (each element is 0 or 1)
* **Use case**: Toggle on/off switches, multi-choice actions where multiple can be on
* **Example**:

  ```python
  from gym.spaces import MultiBinary
  action_space = MultiBinary(5)  # actions like [0,1,1,0,0]
  ```

---

## 🔹 4. `MultiDiscrete([n1, n2, ..., nk])`

* **Meaning**: Tuple of discrete spaces, where each element has its own range
* **Use case**: Structured choices (e.g., multiple control knobs with different options)
* **Example**:

  ```python
  from gym.spaces import MultiDiscrete
  action_space = MultiDiscrete([5, 2, 3])  # e.g., [3, 1, 0]
  ```

---

## 🔹 5. `Tuple(spaces)`

* **Meaning**: Cartesian product of multiple different action spaces
* **Use case**: Combining different types of actions (e.g., discrete move + continuous control)
* **Example**:

  ```python
  from gym.spaces import Tuple, Discrete, Box
  action_space = Tuple((Discrete(2), Box(low=0, high=1, shape=(3,))))
  ```

---

## 🔹 6. `Dict({key: space, ...})`

* **Meaning**: Named collection of spaces; each part of the action is a dictionary entry
* **Use case**: Highly structured action APIs, like in games or modular robots
* **Example**:

  ```python
  from gym.spaces import Dict, Discrete, Box
  action_space = Dict({
      "direction": Discrete(4),
      "throttle": Box(0, 1, shape=(1,))
  })
  ```

---

## 🔹 7. `Sequence(space)` (Experimental / Gymnasium only)

* **Meaning**: Variable-length sequence of actions, where each element follows the same subspace
* **Use case**: Natural language commands, macro-actions
* **Example**:

  ```python
  from gymnasium.spaces import Sequence, Discrete
  action_space = Sequence(Discrete(10))  # e.g., [1,2,5,7]
  ```

---

## 🔹 8. `Text()` (Experimental or custom)

* **Meaning**: String-based action (e.g., natural language)
* **Use case**: Text-based games or instruction-following agents
* **Example**:
  Some environments use:

  ```python
  action = "go north"
  ```

---

## Summary Table

| Space Type       | Domain          | Example Value                 | Use Case                      |
| ---------------- | --------------- | ----------------------------- | ----------------------------- |
| `Discrete(n)`    | Categorical     | `2`                           | Move selection, buttons       |
| `Box`            | Continuous      | `[0.5, -1.2]`                 | Control inputs, motor torques |
| `MultiBinary(n)` | Binary vector   | `[1, 0, 1]`                   | Switches, toggles             |
| `MultiDiscrete`  | Categorical     | `[3, 1]`                      | Structured choices            |
| `Tuple`          | Mixed           | `(1, [0.5, 0.3])`             | Hybrid discrete-continuous    |
| `Dict`           | Mixed + named   | `{"move": 2, "force": [0.1]}` | Modular actions               |
| `Sequence`       | Variable-length | `[1, 2, 3]`                   | Macro-actions, instructions   |
| `Text`           | Textual input   | `"take sword"`                | Natural language commands     |

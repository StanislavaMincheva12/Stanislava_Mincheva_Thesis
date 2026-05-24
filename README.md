# Mesa simulation scaffold

Folder structure:
- `Agents/Firm.py`
- `Agents/Hospital.py`
- `Model/Model.py`
- `Primitives/Primitives.py`
- `run.py`

Run from the project root:

```bash
pip install mesa pandas
python run.py
```

This version is intentionally simple. It is designed to answer:
1. How high-bias delay changes with primitives.
2. Whether low-bias hospitals delay less than high-bias hospitals.
3. When the firm switches from `c_Y` to `c_S`.
4. How many low-bias hospitals accept each contract in equilibrium.

Main output:
- `results/summary.csv`
- per-case CSV files with adoptions, firm history, and posterior history.

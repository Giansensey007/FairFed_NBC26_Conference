# FairFed_NBC26_Conference

Supplementary code and data for **FairFed** (blockchain-incentivized federated learning with Shapley scoring and a delta-norm gate), submitted to NBC ’26 (Nanyang Blockchain Conference).

## Summary

The most widely used approach for individual contribution in federated learning—**Shapley value scoring**—fails to detect **free riders** (clients who return the global model unchanged): they still receive positive Shapley credit. **FairFed** combines Shapley scoring with a **delta-norm gate** so that only clients who meaningfully update the model receive rewards. Honest contributors are rewarded; free riders and Byzantine attackers are not. The experiment also includes an optional **Ethereum Sepolia** deployment of a FairToken ERC-20 and on-chain reward distribution for representative rounds.

## Repository layout

| Path | Contents |
|------|----------|
| [`scripts/fairfed_experiment.py`](scripts/fairfed_experiment.py) | End-to-end experiment: non-IID FL, baselines, optional Solidity compile + Sepolia deploy, 11 publication figures |
| [`results/`](results/) | Frozen run artifacts: `fl_results.json`, `gas_records.json`, `contract_address.txt`, `FairToken_abi.json` |
| [`figures/`](figures/) | PNG figures produced when you run the script (empty until generation) |

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/fairfed_experiment.py
```

- **Phases 1–2 (FL)** and **Phase 5 (figures)** run without blockchain credentials. `fl_results.json` is written under `results/`, and figures are written under `figures/`.
- **Phases 3–4 (Sepolia)** run only if both `FAIRFED_SEPOLIA_RPC` and `FAIRFED_PRIVATE_KEY` are set (e.g. via a `.env` file copied from [`.env.example`](.env.example)). Otherwise the script skips deployment and uses the committed files in `results/` for on-chain metadata in plots.
- The first run installs Solidity `0.8.20` via `py-solc-x` if needed.

## Security

Do **not** commit real private keys or RPC URLs with secrets. If a key was ever pasted into a file that was shared or committed, **rotate** it and use a new Sepolia test account.

## Citation

If you use this repository, please cite the NBC ’26 paper when it is available.

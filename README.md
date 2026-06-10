# NILM v5 — Smart Energy Disaggregation

Live demo: *(add your Streamlit Cloud URL here after deployment)*

Non-Intrusive Load Monitoring using a **Dilated Seq2Point** neural network with midpoint attention and gain-mix augmentation. Trained on UK-DALE data.

## What it does
Given a whole-home aggregate power signal (Watts, 6-second intervals), the app separates individual appliance consumption for:
- Kettle
- Fridge
- Washing Machine
- Dishwasher

## Deployment

### Streamlit Community Cloud (free, recommended)
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select your fork → `app.py`
4. Deploy

### Add model weights
Place your trained `.pt` files in the `models/` folder:
```
models/
├── nilm_kettle.pt
├── nilm_fridge.pt
├── nilm_washing_machine.pt
└── nilm_dish_washer.pt
```
Without weights, the app runs in demo mode using synthetic data.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture
| Component | Detail |
|---|---|
| Backbone | 5× dilated Conv1D (d=1,2,4,8,16) |
| Attention | Midpoint MHA (5 heads, d=50) |
| Augmentation | Gain-Mix (p=0.3) |
| Output | Single midpoint power value (seq2point) |
| Params | ~70 k per appliance |

data:    ; python scripts/download_data.py
train:   ; python train.py --csv data/paysim.csv --n-iter 8 --show-leakage
smoke:   ; python train.py --synthetic 100000 --n-iter 2 --graph-blocks 6
api:     ; uvicorn app.api:app --port 8000
ui:      ; streamlit run app/dashboard.py
test:    ; pytest -q

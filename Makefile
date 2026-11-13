.PHONY: install run flow test report
install: ; pip install -r requirements.txt
run: ; python -m src.pipeline
flow: ; python dags/reco_flow.py
test: ; python -m pytest -q tests
report: ; python -m src.reporting.build_artifacts

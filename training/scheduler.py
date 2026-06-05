def _hot_swap(novo_modelo, novo_scaler) -> None:
    import app.models.predictor as predictor_module

    # Garante a persistência dos novos binários em disco
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(novo_modelo, MODEL_PATH)
    joblib.dump(novo_scaler, SCALER_PATH)

    # Executa a substituição atómica das referências globais de memória
    predictor_module._modelo = novo_modelo
    predictor_module._scaler = novo_scaler
    print("Hot-swap concluído — Novo Random Forest Regressor e Scaler ativos em memória RAM.")

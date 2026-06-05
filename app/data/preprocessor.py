def calcular_status(biomassa: float) -> str:
    """Classifica o status de desenvolvimento biológico do tanque."""
    if biomassa >= 15.0:
        return "CRESCIMENTO_IDEAL"
    elif biomassa >= 10.0:
        return "CRESCIMENTO_MODERADO"
    elif biomassa >= 5.0:
        return "CRESCIMENTO_LENTO"
    else:
        return "ALERTA_BAIXA_BIOMASSA"

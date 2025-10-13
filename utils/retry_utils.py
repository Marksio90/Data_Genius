import time
from typing import Callable, Tuple, Type
import random

def retry_with_backoff(
    func: Callable,
    exceptions: Tuple[Type[Exception], ...],
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
) -> any:
    """
    Wykonuje funkcję z exponential backoff przy błędach
    
    Args:
        func: Funkcja do wykonania
        exceptions: Tupla wyjątków do przechwycenia
        max_retries: Maksymalna liczba prób
        base_delay: Bazowe opóźnienie w sekundach
        max_delay: Maksymalne opóźnienie w sekundach
    
    Returns:
        Wynik funkcji
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries:
                raise
            
            # Oblicz opóźnienie z exponential backoff + jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter
            
            print(f"⏳ Próba {attempt + 1}/{max_retries + 1} nie powiodła się: {e}")
            print(f"⏱️ Czekam {total_delay:.1f}s przed kolejną próbą...")
            
            time.sleep(total_delay)
    
    raise Exception("Nie udało się wykonać funkcji po wszystkich próbach")
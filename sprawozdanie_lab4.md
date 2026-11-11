# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 4 / Obsługa długich sekwencji: SimpleRNN, LSTM, GRU i Conv1D
**Autor / grupa:** Michał Pakuła / GL0.3
**Data:** 04.11.2026

> **Informacja o użyciu AI.** Do redakcji analizy i wniosków, złożenia kodu (notebook) oraz przygotowania układu i formatowania sprawozdania użyto asystenta AI (Claude, Anthropic).

---

## 1. Cel ćwiczenia

Porównanie czterech architektur (SimpleRNN, LSTM, GRU, Conv1D) w zadaniu sumowania wartości wskazywanych z opóźnieniem 8 kroków oraz zbadanie wpływu długości sekwencji (L = 50 i 100), rozmiaru stanu ukrytego SimpleRNN i dylatacji (pola recepcyjnego) Conv1D.

## 2. Metoda i implementacja

### 2.1 Dane

Każda sekwencja ma 2 kanały: wartości z rozkładu jednostajnego [0, 1) oraz znaczniki 0/1. Dwie różne pozycje znaczników losowano z kroków 9..L (numeracja od 1). Znacznik w kroku `t` wskazuje wartość z kroku `t − 8`, a celem jest suma dwóch wskazanych wartości (wejście: cała sekwencja, wyjście: jedna liczba).

```python
p = np.sort(rng.choice(np.arange(d + 1, L + 1), size=2, replace=False))  # kroki 9..L
mark[i, p - 1, 0] = 1.0                 # numeracja od 1 -> indeks od 0
y[i, 0] = val[i, p - 1 - d, 0].sum()    # wartości z kroków t-8
```

**Ręczna kontrola generatora (L = 20):** [wklej wydruk komórki „Ręczna kontrola”: pozycje znaczników, wskazywane kroki, wartości i suma]. Dodatkowo test automatyczny (1000 przykładów) potwierdził: dokładnie 2 znaczniki, pozycje w zakresie 9..L, cel równy sumie wartości z kroków `t − 8`.

Zbiory generowano **raz na długość** i przekazano te same dane wszystkim modelom danej długości: 4000 przykładów treningowych i 800 testowych (osobny zbiór) dla L = 50 oraz L = 100. Punkt odniesienia to predykcja stała równa średniej celu ze zbioru treningowego.

### 2.2 Modele i ustawienia

- **SimpleRNN, LSTM, GRU:** stan ukryty 32, predykcja z ostatniego stanu ukrytego + warstwa liniowa.
- **Conv1D:** 3 warstwy, 64 kanały, jądro 3, krok 1, dylatacje [1, 2, 4], symetryczne dopełnienie (każda warstwa zachowuje długość), ReLU, uśrednianie mapy cech w czasie + warstwa liniowa.
- **Trening (jednakowy dla wszystkich):** MSE, Adam, `lr = 5e-3`, 30 epok, `batch = 128`, przycinanie gradientu 1,0.
- **Ziarna:** dane ze stałym ziarnem generatora (0), modele trenowano na 3 ziarnach inicjalizacji i tasowania (0, 1, 2); podano średnią ± odchylenie.
- **Urządzenie:** GPU T4. Czas mierzono wyłącznie dla uczenia (z synchronizacją CUDA), w sekundach.
- Ustawień nie dobierano na podstawie wyników, więc osobnej walidacji nie wydzielano.

Kod: notebook `cwiczenie_lab4_sprawozdanie.ipynb`, oddany z wykonanymi komórkami.

### 2.3 Pole recepcyjne

Dla stosu o jądrze `k = 3` i kroku 1: `R = 1 + (k − 1)·Σd`.

- dylatacje [1, 2, 4]: `R = 1 + 2·7 = 15`,
- dylatacje [1, 1, 1]: `R = 1 + 2·3 = 7`.

Znacznik w kroku `t` i wskazywana próbka w kroku `t − 8` leżą w odległości 8 kroków, więc jedna pozycja objęłaby oba tylko przy `R ≥ 9`. Wersja [1, 2, 4] (R = 15) może to zrobić, wersja [1, 1, 1] (R = 7) nie.

## 3. Wyniki

### 3.1 Porównanie główne

| Model | L | Parametry | Test MSE (śr. ± odch.) | Czas uczenia [s] |
|---|---|---|---|---|
| predykcja stała | 50 | – | 0,1655 | – |
| SimpleRNN | 50 | 1 185 | 0,1690 ± 0,0037 | 7,0 |
| LSTM | 50 | 4 641 | 0,1660 ± 0,0003 | 2,6 |
| GRU | 50 | 3 489 | **0,1353 ± 0,0085** | 2,6 |
| Conv1D [1,2,4] | 50 | 25 217 | 0,1639 ± 0,0030 | 2,7 |
| predykcja stała | 100 | – | 0,1838 | – |
| SimpleRNN | 100 | 1 185 | 0,1853 ± 0,0009 | 2,6 |
| LSTM | 100 | 4 641 | 0,1840 ± 0,0002 | 2,8 |
| GRU | 100 | 3 489 | 0,1846 ± 0,0024 | 2,5 |
| Conv1D [1,2,4] | 100 | 25 217 | 0,1833 ± 0,0006 | 2,6 |

![Test MSE w funkcji L](wykresy/lab4_01_mse_vs_L.png)

*Rys. 1. Test MSE czterech modeli dla L = 50 i L = 100 (średnia ± odch. z 3 ziaren) na tle predykcji stałej.*

### 3.2 Warianty (L = 100, te same dane i ustawienia)

| Wariant | Parametry | Test MSE (śr. ± odch.) | Czas uczenia [s] | Pole recepcyjne |
|---|---|---|---|---|
| predykcja stała | – | 0,1838 | – | – |
| SimpleRNN, stan 32 | 1 185 | 0,1853 ± 0,0009 | 2,6 | – |
| SimpleRNN, stan 64 | 4 417 | 0,1848 ± 0,0007 | 2,3 | – |
| Conv1D [1,2,4] | 25 217 | 0,1833 ± 0,0006 | 2,6 | 15 |
| Conv1D [1,1,1] | 25 217 | 0,1838 ± 0,0007 | 2,5 | 7 |

### 3.3 Przykład z dużym błędem

Model: SimpleRNN, L = 100, przykład testowy nr 739 (ziarno 0).

| Pozycje znaczników | Wskazywane pozycje | Wskazywane wartości | Prawidłowa suma | Predykcja | Błąd kwadratowy |
|---|---|---|---|---|---|
| 72 i 99 | 64 i 91 | 0,9921 i 0,9762 | 1,9683 | 0,9800 | 0,9766 |

Średni błąd kwadratowy tego modelu na teście wynosi 0,1865.

![Przykład z dużym błędem](wykresy/lab4_02_przyklad_bledu.png)

*Rys. 2. Przykład testowy z dużym błędem (SimpleRNN, L = 100): wartości, znaczniki (czerwone linie) i wskazywane próbki (zielone punkty).*

## 4. Analiza i interpretacja

**Które modele poprawiły wynik względem predykcji stałej?** Wyraźnie tylko **GRU dla L = 50**: 0,1353 wobec 0,1655 (o ok. 18% mniej), choć przy dużym rozrzucie między ziarnami (± 0,0085). Conv1D dla L = 50 (0,1639) jest minimalnie lepszy od stałej, ale różnica (0,0016) jest mniejsza niż jego odchylenie (0,0030). LSTM (0,1660) jest równy stałej, a SimpleRNN (0,1690) nieco gorszy. Predykcja stała ma tu MSE ≈ 1/6, bo jest wariancją celu.

**Jak zmienił się błąd przy dłuższym wejściu (L = 100)?** Wszystkie cztery modele są w granicach rozrzutu na poziomie predykcji stałej (0,1833–0,1853 wobec 0,1838), czyli żaden nie nauczył się zadania w przyjętych ustawieniach (30 epok, 4000 przykładów). Przewaga GRU z L = 50 zanikła. Wartości predykcji stałej różnią się między L (0,1655 i 0,1838), bo każda długość ma osobny zbiór testowy 800 przykładów, więc porównujemy modele z ich własnym punktem odniesienia.

**Stan ukryty 32 → 64 (SimpleRNN).** Błąd praktycznie się nie zmienił (0,1853 → 0,1848, różnica poniżej odchyleń), a liczba parametrów wzrosła ok. 3,7 raza (1 185 → 4 417). Czas nie wzrósł (2,6 → 2,3 s, przy tak małym modelu różnice są na granicy dokładności pomiaru). Większy stan nie pomógł.

**Dylatacje [1,2,4] → [1,1,1] (Conv1D).** Zmiana nie dodaje parametrów (obie wersje mają 25 217) i nie zmienia czasu, a zmniejsza pole recepcyjne z 15 do 7. Wersja [1,1,1] (R = 7 < 9) strukturalnie nie może objąć znacznika i wskazywanej próbki jedną pozycją, a [1,2,4] (R = 15) może. W wynikach tej różnicy nie widać (0,1833 vs 0,1838, oba na poziomie stałej), więc sama możliwość objęcia obu pozycji nie wystarczyła, by model nauczył się zadania w naszych ustawieniach.

**Przykład z dużym błędem.** W przykładzie nr 739 (SimpleRNN, L = 100) znaczniki są w krokach 72 i 99, więc wskazywane są wartości z kroków 64 i 91 (0,9921 i 0,9762). Prawidłowa suma (1,9683) leży blisko górnego krańca możliwych wartości, a model zwrócił 0,98, czyli ok. średniej celu (ok. 1). Błąd (0,9766) jest więc ok. 5 razy większy od średniego (0,1865). SimpleRNN odczytuje wynik z ostatniego stanu ukrytego (po kroku 100), więc informacja o wartościach z kroków 64 i 91 musiałaby przetrwać odpowiednio 36 i 9 kroków aktualizacji stanu. Zachowanie modelu (wynik ok. średniej niezależnie od wejścia) wskazuje, że tej informacji w stanie końcowym nie ma.

**Ograniczenia.** Tylko 3 ziarna, jedna para zbiorów na długość, 30 epok bez strojenia (stąd część modeli mogła się nie nauczyć zadania), brak krzywych uczenia. Czas dla SimpleRNN przy L = 50 (7,0 s) jest wyraźnie wyższy od pozostałych i prawdopodobnie zawiera narzut inicjalizacji pierwszego treningu, więc porównania czasu między modelami są tylko orientacyjne. Modele mają różne liczby parametrów (1 185 do 25 217), więc nie są porównywalne pojemnością. Uśrednianie w Conv1D umożliwia rozpoznawanie wskazanych wartości lokalnie, więc jego wynik nie mierzy wyłącznie pamięci długiego zasięgu.

## 5. Wnioski

W naszych ustawieniach (30 epok, 4000 przykładów, 3 ziarna) jedynym modelem, który wyraźnie pokonał predykcję stałą, był GRU dla L = 50 (0,1353 wobec 0,1655), przy dużym rozrzucie między ziarnami. Dla L = 100 żaden z czterech modeli nie odróżnił się od predykcji stałej, więc w tym eksperymencie dłuższa sekwencja zniwelowała przewagę GRU. Zwiększenie stanu ukrytego SimpleRNN z 32 do 64 nie poprawiło błędu, a kosztowało ok. 3,7 razy więcej parametrów. Zmiana dylatacji Conv1D z [1,2,4] na [1,1,1] zmniejsza pole recepcyjne z 15 do 7 (poniżej wymaganych 9 kroków) bez kosztu w parametrach, ale nie zmieniła wyniku, bo oba warianty zostały na poziomie predykcji stałej. Nie mamy więc w tych danych dowodu na wyraźną przewagę LSTM ani dilated Conv1D nad prostym RNN. Wnioski dotyczą jednego zadania i krótkiego treningu. Dalsze kroki to więcej epok i danych, sprawdzenie krzywych uczenia, dobór `lr` na osobnej walidacji oraz więcej ziaren.

# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 2 / Sieci rekurencyjne: budowa i uczenie
**Autor / grupa:** [imię i nazwisko] / [grupa]
**Data:** 28.10.2026

> **Informacja o użyciu AI.** Do redakcji wniosków i hipotez, złożenia kodu (notebook) oraz do przygotowania układu i formatowania sprawozdania użyto asystenta AI (Claude, Anthropic). Hipotezy zostały sformułowane z pomocą AI jako oczekiwania wynikające ze wstępu teoretycznego do laboratorium, już po wykonaniu eksperymentów. Wszystkie wyniki liczbowe pochodzą z uruchomienia notebooka przez autora w Google Colab.

---

## 1. Cel ćwiczenia

Celem ćwiczenia było zbudowanie w PyTorch sieci rekurencyjnej, która na podstawie okna poprzednich próbek przewiduje kolejną próbkę szeregu czasowego (sinusoida z szumem), oraz eksperymentalne sprawdzenie wpływu trzech czynników na jakość prognozy:

- **A:** liczby jednostek ukrytych (`hidden`),
- **B:** długości okna wejściowego (`OKNO`),
- **C:** typu komórki rekurencyjnej (RNN, LSTM, GRU) dla skrajnych długości okna.

Jakość oceniano za pomocą błędu średniokwadratowego (MSE) na zbiorze testowym, porównując ją z prostą heurystyką „prognoza = ostatnia próbka okna".

## 2. Metoda i implementacja

### 2.1 Narzędzia i dane

- Python, PyTorch, NumPy, Matplotlib; środowisko Google Colab.
- Kod: notebook `cwiczenie_lab2_sprawozdanie.ipynb` ([link do repozytorium](#)), oddany z wykonanymi komórkami.
- Dane: szereg `y = sin(t) + 0,1·sin(3t) + 0,05·ε` (`ε` ~ N(0, 1)), 2000 próbek dla `t` ∈ [0, 100], ziarno 0; generowany raz i wspólny dla wszystkich eksperymentów.
- Przykłady uczące: okno `OKNO` kolejnych próbek → następna próbka (zadanie sequence-to-vector). Podział **chronologiczny** 80% / 20% (bez tasowania); dla `OKNO = 30`: 1576 okien treningowych i 394 testowe.

![Fragment szeregu](wykresy/lab2_00_szereg.png)

*Rys. 1. Fragment szeregu (200 próbek).*

### 2.2 Model i trening

Jedna warstwa rekurencyjna (`nn.RNN` / `nn.LSTM` / `nn.GRU`) i warstwa liniowa na ostatnim stanie ukrytym. Funkcja straty: MSE; optymalizator: Adam; przycinanie gradientu (norma 5,0). Funkcję `trenuj` uzupełniono o zapis straty treningowej w każdej epoce.

```python
class ModelRNN(nn.Module):
    def __init__(self, hidden=32, typ='RNN'):
        super().__init__()
        rnn = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[typ]
        self.rnn = rnn(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])   # ostatni krok -> predykcja


def trenuj(model, Xtr, Ytr, Xte, Yte, epoki=30, lr=1e-2, batch=64):
    """Zwraca (strata testowa, strata treningowa) po każdej epoce."""
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    strata = nn.MSELoss()
    hist, hist_tr = [], []
    n = len(Xtr)
    for ep in range(epoki):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            loss = strata(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            hist.append(strata(model(Xte), Yte).item())
            hist_tr.append(strata(model(Xtr), Ytr).item())
    return hist, hist_tr
```

Każdą konfigurację trenowano 3 razy (ziarna 0, 1, 2). Wynik to średnia i odchylenie standardowe z `np.mean(hist[-5:])`; po każdej zmianie okna okna i podział danych tworzone są od nowa (funkcja `przygotuj(okno)`):

```python
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    model = ModelRNN(hidden=hidden, typ=typ)
    h, h_tr = trenuj(model, Xtr, Ytr, Xte, Yte, epoki=20, lr=1e-2, batch=64)
    mse.append(np.mean(h[-5:]))
```

### 2.3 Konfiguracja eksperymentów

Stałe we wszystkich eksperymentach: `epoki = 20`, `lr = 1e-2`, `batch = 64`, 3 ziarna. W każdym eksperymencie zmieniany jest tylko jeden czynnik.

| Eksperyment | Stałe | Zmieniane |
|---|---|---|
| A | GRU, `OKNO = 30` | `hidden` = 8, 16, 32, 64 |
| B | GRU, `hidden = 32` | `OKNO` = 10, 30, 60, 120 |
| C | `hidden = 32` | typ = RNN, LSTM, GRU; `OKNO` = 10 i 120 |

Konfiguracje powtarzające się między eksperymentami (GRU, `hidden = 32`, okna 10, 30 i 120) policzono raz i użyto w kilku tabelach.

### 2.4 Hipotezy (oczekiwania wynikające z materiałów teoretycznych)

- **Hipoteza A:** Najmniejszy model (`hidden = 8`) będzie wyraźnie gorszy od większych, a różnice między `hidden = 32` i `64` będą niewielkie.
- **Hipoteza B:** Wynik poprawi się wraz z długością okna, ale przy oknie 120 zysk się wyrówna lub zaniknie, bo dla GRU dłuższy łańcuch propagacji wstecznej jest trudniejszy w uczeniu.

## 3. Wyniki

### 3.1 Punkt odniesienia (baseline)

Model bazowy: RNN, `hidden = 32`, `OKNO = 30`, 30 epok (notebook bez zmian).

- Test MSE modelu bazowego: **[wpisz z wydruku komórki z baseline'em]** (z wykresu: ok. 0,0037 po 30 epokach).
- Heurystyka „ostatnia próbka" (`OKNO = 30`): **0,00631**.

![Baseline: krzywa uczenia](wykresy/lab2_01_baseline_krzywa.png)

*Rys. 2. Baseline: strata testowa i treningowa (skala logarytmiczna).*

### 3.2 Eksperyment A: liczba jednostek (GRU, `OKNO = 30`)

| Konfiguracja | Parametry | Test MSE (średnia ± odch.) | Czas treningu [s] |
|---|---|---|---|
| heurystyka „ostatnia próbka" | – | 0,00631 | – |
| GRU, hidden=8 | 273 | 0,00380 ± 0,00004 | 6,3 |
| GRU, hidden=16 | 929 | 0,00374 ± 0,00002 | 6,3 |
| GRU, hidden=32 | 3 393 | 0,00376 ± 0,00001 | 7,4 |
| GRU, hidden=64 | 12 929 | 0,00375 ± 0,00008 | 11,1 |

![Eksperyment A](wykresy/lab2_02_ekspA.png)

*Rys. 3. Eksperyment A: krzywe straty testowej (średnia z 3 ziaren).*

### 3.3 Eksperyment B: długość okna (GRU, `hidden = 32`)

| Konfiguracja | Parametry | Test MSE (średnia ± odch.) | Czas treningu [s] | Heurystyka dla tego okna |
|---|---|---|---|---|
| GRU, okno=10 | 3 393 | 0,00414 ± 0,00019 | 3,1 | 0,00628 |
| GRU, okno=30 | 3 393 | 0,00376 ± 0,00001 | 7,4 | 0,00631 |
| GRU, okno=60 | 3 393 | 0,00343 ± 0,00021 | 15,3 | 0,00633 |
| GRU, okno=120 | 3 393 | 0,00344 ± 0,00019 | 27,2 | 0,00631 |

Heurystykę liczono osobno dla każdego okna, bo zmienia się zbiór testowy (różna liczba okien).

![Eksperyment B](wykresy/lab2_03_ekspB.png)

*Rys. 4. Eksperyment B: krzywe straty testowej (średnia z 3 ziaren).*

### 3.4 Eksperyment C: typ komórki (`hidden = 32`)

| Konfiguracja | Parametry | Test MSE (średnia ± odch.) | Czas treningu [s] |
|---|---|---|---|
| heurystyka (okno=10) | – | 0,00628 | – |
| RNN, okno=10 | 1 153 | 0,00417 ± 0,00025 | 1,4 |
| LSTM, okno=10 | 4 513 | 0,00415 ± 0,00011 | 2,7 |
| GRU, okno=10 | 3 393 | 0,00414 ± 0,00019 | 3,1 |
| heurystyka (okno=120) | – | 0,00631 | – |
| RNN, okno=120 | 1 153 | 0,00400 ± 0,00001 | 9,8 |
| LSTM, okno=120 | 4 513 | 0,00375 ± 0,00013 | 16,8 |
| GRU, okno=120 | 3 393 | 0,00344 ± 0,00019 | 27,2 |

![Eksperyment C](wykresy/lab2_04_ekspC.png)

*Rys. 5. Eksperyment C: krzywe straty testowej (średnia z 3 ziaren).*

### 3.5 Najlepsza konfiguracja

Najniższy średni test MSE uzyskano dla **GRU, `hidden = 32`, `OKNO = 60`** (0,00343 ± 0,00021).

![Strata treningowa i testowa](wykresy/lab2_05_najlepsza_train_test.png)

*Rys. 6. Najlepsza konfiguracja: strata treningowa i testowa.*

![Predykcja](wykresy/lab2_06_najlepsza_predykcja.png)

*Rys. 7. Najlepsza konfiguracja: predykcja na tle rzeczywistego przebiegu (200 punktów testowych, model z ziarna 2).*

## 4. Analiza i interpretacja

### 4.1 Hipotezy a obserwacje

| Hipoteza | Obserwacja | Wynik |
|---|---|---|
| A: `hidden = 8` wyraźnie gorszy od większych; różnice 32 vs 64 niewielkie | Test MSE: 0,00380 / 0,00374 / 0,00376 / 0,00375 dla `hidden` = 8 / 16 / 32 / 64, czyli różnice mieszczą się w rozrzucie między ziarnami (Rys. 3). Model `hidden = 8` uczył się wolniej tylko w pierwszych epokach. Część o 32 vs 64 zgadza się z obserwacją | **niepotwierdzona** (część o 32 vs 64 potwierdzona) |
| B: poprawa z długością okna, zanik zysku przy 120 | MSE: 0,00414 → 0,00376 → 0,00343 → 0,00344 dla okien 10 / 30 / 60 / 120; między 60 a 120 praktycznie bez zmiany, a czas rośnie z 15,3 do 27,2 s (Rys. 4) | **potwierdzona** |

### 4.2 Interpretacja wyników

- **Liczba jednostek.** Sygnał jest prosty (dwie sinusoidy i szum), więc już 8 jednostek ukrytych (273 parametry) wystarcza do osiągnięcia poziomu ok. 0,0038. Dalsze zwiększanie modelu (do 12 929 parametrów) tylko wydłuża trening.
- **Długość okna.** Okno 10 obejmuje niewielki ułamek okresu głównej składowej (ok. 126 próbek), więc model ma za mało kontekstu, by rozpoznać fazę i kierunek sygnału. Przy oknach 60 i 120 wynik jest najlepszy i praktycznie jednakowy. Czas treningu rośnie w przybliżeniu proporcjonalnie do długości okna (3,1 → 7,4 → 15,3 → 27,2 s), bo dłuższy jest łańcuch obliczeń propagacji wstecznej.
- **Typ komórki.** Przy oknie 10 RNN, LSTM i GRU dają praktycznie ten sam wynik. Przy oknie 120 przewagę mają GRU (0,00344) i LSTM (0,00375) nad RNN (0,00400), co jest zgodne z oczekiwaniem, że komórki bramkowe lepiej radzą sobie z dłuższym kontekstem. Różnice nie są jednak duże wobec odchyleń standardowych (do 0,00019), a modele mają różną liczbę parametrów (1 153, 4 513, 3 393), więc nie jest to porównanie modeli tej samej wielkości.
- **Względem heurystyki.** Wszystkie konfiguracje są lepsze od heurystyki „ostatnia próbka" (ok. 0,0063): o ok. 34% (najsłabsza, 0,00417) do ok. 46% (najlepsza, 0,00343). Wyniki pozostają jednak powyżej dolnej granicy wyznaczonej przez wariancję szumu (ok. 0,0025), czyli model nie wykorzystuje całej struktury sygnału.
- **Niedouczenie, przeuczenie, stabilność.** Strata treningowa i testowa najlepszej konfiguracji praktycznie się pokrywają (Rys. 6), więc nie zaobserwowano przeuczenia. Krzywe są poszarpane (pojedyncze skoki, np. ok. 12. i 23. epoki w baseline'ie, ok. 11.–13. w eksperymentach), co przy `lr = 1e-2` jest spodziewane, ale nie wystąpił wybuch gradientu ani wartości `nan`. Predykcja dobrze śledzi sygnał i wygładza szum (Rys. 7).

### 4.3 Ograniczenia

- Tylko 3 ziarna na konfigurację; część różnic (np. okno 30 vs 60, typy komórek przy oknie 10) jest porównywalna z rozrzutem między ziarnami.
- Konfigurację najlepszą wybrano na podstawie zbioru testowego, a nie walidacyjnego, więc jej wynik może być lekko zawyżony.
- Przy różnych oknach zbiór testowy ma nieco inny rozmiar i skład.
- Wykres predykcji pochodzi z modelu z ostatniego ziarna (2), a krzywe uczenia są średnią z 3 ziaren.
- Wnioski dotyczą jednego, prostego, syntetycznego sygnału.

## 5. Wnioski

Większa liczba jednostek ukrytych nie poprawiła wyniku: dla GRU przy oknie 30 test MSE wyniósł 0,00380, 0,00374, 0,00376 i 0,00375 dla `hidden` = 8, 16, 32 i 64, czyli różnice mieszczą się w rozrzucie między ziarnami, a liczba parametrów rosła od 273 do 12 929. Najlepsze okazały się okna 60 i 120 (0,00343 i 0,00344), a okno 10 było wyraźnie gorsze (0,00414), co jest zgodne z tym, że 10 próbek to ułamek okresu sygnału (ok. 126 próbek) i daje za mało kontekstu. Zysk z wydłużania okna zanika między 60 a 120, a czas treningu rośnie z 15,3 do 27,2 s, więc w naszym eksperymencie okno 60 jest rozsądnym kompromisem. Typy komórek reagują na wydłużenie okna różnie: przy oknie 10 RNN, LSTM i GRU dają praktycznie ten sam wynik (0,00417, 0,00415, 0,00414), a przy oknie 120 GRU (0,00344) i LSTM (0,00375) są lepsze od RNN (0,00400), przy czym te modele mają różną liczbę parametrów. Wszystkie modele są lepsze od heurystyki „ostatnia próbka" (ok. 0,0063), o ok. 34–46%, ale nadal powyżej poziomu szumu (ok. 0,0025). Przeuczenia nie zaobserwowano (strata treningowa i testowa najlepszej konfiguracji praktycznie się pokrywają) i nie wystąpiła niestabilność typu wybuch gradientu, choć krzywe są poszarpane przy `lr = 1e-2`. Ograniczeniem jest tylko 3 ziarna, część różnic bliska rozrzutowi, wybór konfiguracji po zbiorze testowym zamiast walidacyjnego i jeden syntetyczny sygnał. Hipoteza A (hidden = 8 wyraźnie gorszy od większych) nie potwierdziła się: wynik dla `hidden = 8` jest praktycznie taki sam jak dla 64 (0,00380 wobec 0,00375), choć uczył się wolniej w pierwszych epokach. Hipoteza B (poprawa z długością okna i zanik zysku przy 120) potwierdziła się: MSE spadło z 0,00414 (okno 10) do 0,00343 (okno 60), a między oknami 60 i 120 praktycznie się nie zmieniło. Możliwe kierunki poprawy to dobór hiperparametrów na zbiorze walidacyjnym, więcej ziaren oraz sprawdzenie mniejszego `lr`.

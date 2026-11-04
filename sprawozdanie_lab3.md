# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 3 / Prognozowanie szeregów czasowych: prognoza wielokrokowa i porównanie z baseline'ami
**Autor / grupa:** Michał Pakuła / GL.03
**Data:** 29.10.2026

> **Informacja o użyciu AI.** Do redakcji analizy, interpretacji i wniosków, przygotowania pomocniczej komórki kodu składającej tabele wyników oraz do przygotowania układu i formatowania sprawozdania użyto asystenta AI

---

## 1. Cel ćwiczenia

Celem ćwiczenia było sprawdzenie, czy model GRU wnosi coś ponad proste reguły (baseline'y) w prognozowaniu dziennej sprzedaży (szereg syntetyczny), zarówno jednokrokowo, jak i na horyzoncie 14 dni, przy uczciwej walidacji czasowej (podział chronologiczny, standaryzacja bez przecieku). Zbadano też wpływ długości okna wejściowego (3, 7 i 28 dni) oraz zinterpretowano wyniki, w tym ich związek z modelami mowy.

## 2. Metoda i implementacja

### 2.1 Narzędzia i dane

- Python, PyTorch, NumPy, Matplotlib; środowisko Google Colab.
- Kod: notebook `cwiczenie_lab_03.ipynb` ([link do repozytorium](#)), oddany z wykonanymi komórkami. Do notebooka dodano jedną komórkę składającą tabele wyników do sprawozdania.
- Dane syntetyczne (1000 dni): `sprzedaż(t) = 100 + 0,02·t + 8·sin(2πt/7) + 15·sin(2πt/365) + szum`, szum ~ N(0, 3²). Z zadania wynika dolna granica błędu: MAE ≈ 2,39 i RMSE = 3,0 (wariancja szumu jest nieusuwalna).
- Podział chronologiczny 70 / 15 / 15: trening dni 0–699, walidacja 700–849, test 850–999.
- Standaryzacja `z = (y − μ) / σ` ze statystykami wyłącznie z części treningowej: μ = 106,96, σ = 12,00. Metryki liczono po powrocie do oryginalnej skali.
- Okna: `OKNO = 28` dni → następny dzień; kształty wejść `(672, 28, 1)` dla treningu oraz `(150, 28, 1)` dla walidacji i testu.

<img width="938" height="340" alt="image" src="https://github.com/user-attachments/assets/50de2c06-cb4e-4bbe-bf31-fa806d02529a" />


*Rys. 1. Dzienna sprzedaż i podział chronologiczny na trening, walidację i test.*

### 2.2 Model i trening

Jedna warstwa `nn.GRU(1, 32)` i warstwa liniowa na stanie po całym oknie (3 393 parametry). Funkcja straty MSE (w skali standaryzowanej), optymalizator Adam (`lr = 1e-2`), 40 epok, `batch = 64`. Po każdej epoce zapisywana jest strata treningowa i walidacyjna, a na końcu przywracane są wagi z epoki o najniższej stracie walidacyjnej (early stopping).

### 2.3 Prognoza wielokrokowa i baseline'y

Prognoza wielokrokowa jest **rekurencyjna**: model przewiduje jeden dzień, prognoza trafia do okna i model przewiduje następny (`H = 14`). Ocenę wykonano z 10 punktów startowych co 14 dni w zbiorze testowym (dni 850, 864, …, 976). Baseline'y nie korzystają z przyszłości:

- **sezonowy:** powtórzenie ostatniego znanego tygodnia, `y[t0 − 7 + (h mod 7)]`,
- **naiwny:** ostatnia znana wartość `y[t0 − 1]` powtórzona przez 14 dni.

Dodatkowo w prognozie jednokrokowej policzono średnią ruchomą z okna. Wszystkie metody oceniano na tych samych dniach testowych.

```python
def rollout(model, s, start, H, okno):
    """Prognoza rekurencyjna na H kroków; model widzi tylko dane sprzed `start`."""
    hist = list(s[start-okno:start]); wyn = []
    model.eval()
    with torch.no_grad():
        for _ in range(H):
            x = torch.tensor(np.array(hist[-okno:], np.float32).reshape(1, okno, 1)).to(DEVICE)
            y = model(x).item(); wyn.append(y); hist.append(y)
    return np.array(wyn)
```

### 2.4 Eksperyment: długość okna

Okna `OKNO` ∈ {3, 7, 28}, każde na 2 ziarnach (`torch.manual_seed(0)` i `(1)`), pozostałe parametry jak w modelu bazowym (`hidden = 32`, `lr = 1e-2`, 40 epok, `batch = 64`). Dla każdego okna tworzono nowe okna, trenowano model od zera i liczono stratę walidacyjną oraz MAE jedno- i wielokrokowe na teście. Najlepsze okno wybrano **według straty walidacyjnej** (średnia z 2 ziaren), a nie według wyników testowych.

## 3. Wyniki

### 3.1 Model bazowy i krzywe uczenia

Model bazowy (GRU, `hidden = 32`, `OKNO = 28`): **najlepsza epoka: 36** (numeracja od 0, z 40), strata walidacyjna w tej epoce **0,0822** (skala standaryzowana), 3 393 parametry.

<img width="792" height="366" alt="image" src="https://github.com/user-attachments/assets/571fc331-b40f-406e-8078-ec54b106bed1" />


*Rys. 2. Krzywe uczenia modelu bazowego: strata treningowa i walidacyjna (skala logarytmiczna).*

### 3.2 Prognoza jednokrokowa na teście

| Metoda | MAE | RMSE |
|---|---|---|
| GRU | 2,91 | 3,54 |
| naiwny (t−1) | 5,44 | 6,51 |
| sezonowy (t−7) | 3,21 | 4,07 |
| średnia ruchoma | 5,80 | 6,84 |
| granica szumu | 2,39 | 3,00 |

<img width="908" height="297" alt="image" src="https://github.com/user-attachments/assets/e162c9bd-47c1-48f0-8001-71a07b3dd13b" />


*Rys. 3. Prognoza jednokrokowa GRU na zbiorze testowym (150 dni) na tle wartości rzeczywistych.*

### 3.3 Prognoza wielokrokowa (H = 14) na teście

| Metoda | MAE 1-krok | RMSE 1-krok | MAE 14-krok | RMSE 14-krok |
|---|---|---|---|---|
| GRU | 2,91 | 3,54 | 3,88 | 4,75 |
| sezonowy | 3,21 | 4,07 | 3,52 | 4,42 |
| naiwny | 5,44 | 6,51 | 10,09 | 11,85 |

Liczba punktów startowych: 10.

<img width="716" height="415" alt="image" src="https://github.com/user-attachments/assets/00f952b6-caa1-4ed9-a100-befe49d0a397" />


*Rys. 4. MAE w funkcji horyzontu h (uśrednione po 10 punktach startowych).*

### 3.4 Wpływ długości okna

| Okno | Ziarno | Val MSE | MAE 1-krok | MAE 14-krok |
|---|---|---|---|---|
| 3 | 0 | 0,2567 | 5,00 | 7,18 |
| 3 | 1 | 0,2486 | 5,04 | 7,41 |
| **3** | **średnia** | **0,2527** | **5,02** | **7,30** |
| 7 | 0 | 0,0855 | 2,90 | 4,21 |
| 7 | 1 | 0,0896 | 2,88 | 3,89 |
| **7** | **średnia** | **0,0876** | **2,89** | **4,05** |
| 28 | 0 | 0,0822 | 2,91 | 3,88 |
| 28 | 1 | 0,0850 | 3,01 | 4,29 |
| **28** | **średnia** | **0,0836** | **2,96** | **4,09** |

**Najlepsze okno według walidacji: 28 dni** (średni val MSE 0,0836). Wynik dla okna 28 i ziarna 0 pokrywa się z modelem bazowym (MAE 2,91 i 3,88), co potwierdza powtarzalność eksperymentu.

### 3.5 Prognoza poza końcem danych (ilustracja)

<img width="908" height="297" alt="image" src="https://github.com/user-attachments/assets/472e45b7-5466-4b46-ac04-bde77a85c20c" />


*Rys. 5. Prognoza 14 dni poza końcem danych (dni 1000–1013). Tej prognozy nie można ocenić, bo tych dni nie ma w danych; ocenę zawiera sekcja 3.3.*

## 4. Analiza i interpretacja

### 4.1 Krzywe uczenia: niedouczenie i przeuczenie

- **Niedouczenie:** nie widać. Strata treningowa spada szybko do ok. 0,09 w ciągu ok. 10 epok i potem utrzymuje się na ok. 0,07–0,09. W skali standaryzowanej szum o σ = 3 odpowiada wariancji (3 / 12)² ≈ 0,0625, więc strata treningowa jest blisko dolnej granicy.
- **Przeuczenie:** wyraźnego nie ma, bo strata walidacyjna nie rośnie systematycznie. Jest za to **bardzo poszarpana** (wahania od ok. 0,08 do ok. 0,2, np. skoki w okolicach epok 12, 20, 35 i 38) i przez większość treningu leży powyżej treningowej. Wskazuje to na niestabilność przy `lr = 1e-2` oraz małym zbiorze walidacyjnym (150 przykładów), a nie na zapamiętywanie treningu.
- **Wybór epoki:** najlepsza epoka (36) wypadła w końcówce treningu, a ostatnia epoka (39) ma wyraźnie gorszą walidację (ok. 0,12), więc przywrócenie wag z najlepszej epoki ma znaczenie.

### 4.2 Odpowiedzi na pytania z zadania

**Czy GRU bije baseline'y w prognozie jedno- i wielokrokowej? Dlaczego wyniki się różnią?**
Jednokrokowo tak: GRU (MAE 2,91) jest lepszy od sezonowego (3,21) o ok. 9% i od naiwnego (5,44) o ok. 46%. Wielokrokowo bije tylko naiwny (MAE 3,88 wobec 10,09), a przegrywa z sezonowym (3,52). Naiwny jest silny tylko wtedy, gdy horyzont jest wielokrotnością tygodnia (na Rys. 4 jego błąd osiąga ok. 17 dla h = 3 i h = 10, a spada do ok. 3–4 dla h = 7 i h = 14). Sezonowy jednokrokowo kopiuje dzień razem z jego szumem, więc GRU, który uśrednia informację z kilku tygodni, wygrywa. Wielokrokowo GRU traci przewagę przez **kumulację błędu** w strategii rekurencyjnej: od pierwszego kroku (błąd ok. 2,7, blisko granicy szumu) kolejne prognozy opierają się coraz bardziej na własnych błędach. MAE wzrasta z 2,91 do 3,88, czyli o ok. 33%. Błąd GRU nie rośnie monotonicznie, lecz oscyluje z okresem ok. tygodnia i jest wyraźnie większy m.in. dla h = 5–6 oraz 12–13 (odczyt z wykresu), a sezonowy ma błąd prawie stały (ok. 3–4).

**Jak daleko jest wynik jednokrokowy od granicy szumu (MAE ≈ 2,39)?**
GRU osiąga MAE 2,91, czyli o ok. 0,52 (ok. 22%) powyżej granicy; RMSE 3,54 wobec 3,00 to ok. 18% więcej. Model jest więc blisko granicy, ale jej nie osiąga. Część różnicy może wynikać z niedoskonałego odwzorowania trendu i sezonowości rocznej oraz z niestabilnego treningu.

**Dlaczego okno 3 dni daje gorszy wynik? Czy dłuższe okno coś poprawia?**
Okno 3 dni jest krótsze niż okres sezonowości tygodniowej (7 dni), więc model nie widzi pełnego rytmu tygodnia. Jego MAE jednokrokowe (ok. 5,0) jest zbliżone do naiwnego (5,44), a strata walidacyjna jest ok. 3 razy wyższa niż dla okien 7 i 28 (0,2527 wobec 0,0876 i 0,0836). Wydłużenie okna z 7 do 28 dni niczego wyraźnie nie poprawia: MAE jednokrokowe wynosi 2,89 i 2,96, a wielokrokowe 4,05 i 4,09. Wystarcza więc okno obejmujące co najmniej jeden okres sezonowy.

**Czy różnice między wariantami są większe niż rozrzut między ziarnami?**
Różnica między oknem 3 a oknami 7 i 28 jest zdecydowanie większa niż rozrzut (jednokrokowo ok. 5,0 wobec ok. 2,9 przy rozrzucie ≤ 0,1; wielokrokowo ok. 7,3 wobec ok. 4,1 przy rozrzucie do 0,4). Różnica między oknami 7 i 28 jest mniejsza niż rozrzut (średnie MAE 2,89 vs 2,96 i 4,05 vs 4,09, przy rozrzucie między ziarnami do 0,1 jednokrokowo i do 0,4 wielokrokowo), więc nie można rozstrzygnąć, które z nich jest lepsze. Wybór okna 28 według walidacji (0,0836 wobec 0,0876) opiera się na różnicy porównywalnej z rozrzutem między ziarnami (np. 0,0855–0,0896 dla okna 7). Przewaga GRU nad sezonowym w prognozie jednokrokowej (0,30 MAE) jest większa niż rozrzut jednokrokowy, a w prognozie wielokrokowej wszystkie uruchomienia GRU dla okien 7 i 28 (3,88–4,29) są gorsze od sezonowego (3,52).

**Które obserwacje przeniosą się na modele mowy?**
Zgodnie z częścią 1 wstępu sygnał mowy to szereg czasowy: po ramkowaniu powstaje sekwencja wektorów cech (ok. 100 ramek na sekundę).
- **Dobór okna:** okno krótsze niż okres sezonowy zawiodło, więc analogicznie kontekst w modelu mowy musi obejmować co najmniej skalę zjawiska, które ma opisać (głoska trwa ok. 50–150 ms, a koartykulacja wymaga głosek sąsiednich), czyli kontekst kilkudziesięciu do kilkuset ramek. Okres sezonowy odpowiada okresowości tonu krtaniowego (F0) i rytmowi sylab.
- **Kumulacja błędu w prognozie rekurencyjnej:** przeniesie się na dekodowanie autoregresyjne (synteza mowy, modele seq2seq, np. dekoder Whispera, wokoder WaveRNN generujący kolejne próbki), gdzie błąd jednego kroku przenosi się na następne, więc jakość jednokrokowa nie przesądza o jakości generowania całej sekwencji.
- **Uczciwa ocena:** standaryzacja ze statystykami z treningu odpowiada normalizacji cech (CMVN), a podział chronologiczny odpowiada podziałowi rozłącznemu po mówcach i nagraniach.
- **Porównanie z prostą regułą:** w mowie także trzeba zestawiać model z prostym punktem odniesienia.

### 4.3 Ograniczenia

- Ocena wielokrokowa opiera się na 10 punktach startowych z jednego okresu testowego, więc jest dość zaszumiona.
- Eksperyment z oknem używa tylko 2 ziaren, a część różnic (okno 7 vs 28) jest porównywalna z rozrzutem.
- Zbiór walidacyjny (150 przykładów) jest mały, co przy `lr = 1e-2` daje poszarpaną krzywą walidacyjną.
- Dane są syntetyczne i znamy ich wzór, więc wnioski dotyczą tego jednego szeregu.
- Wartości w tekście odczytane z wykresów (np. błąd w funkcji horyzontu) są przybliżone.

## 5. Wnioski

Model GRU jest lepszy od wszystkich baseline'ów w prognozie jednokrokowej (MAE 2,91 wobec 3,21 dla sezonowego i 5,44 dla naiwnego), ale w prognozie na 14 dni przegrywa z baseline'em sezonowym (3,88 wobec 3,52), co przypisujemy kumulacji błędu w strategii rekurencyjnej. Wynik jednokrokowy jest o ok. 0,52 MAE (ok. 22%) od granicy szumu (2,39). Okno 3 dni jest wyraźnie gorsze (MAE ok. 5,0), bo nie obejmuje okresu sezonowości tygodniowej, a wydłużenie okna z 7 do 28 dni nie dało rozstrzygającej poprawy (różnice mniejsze niż rozrzut między ziarnami). Zgodnie z regułą wyboru według walidacji najlepszym oknem jest 28 dni, choć jej przewaga nad oknem 7 jest niewielka. Na krzywych uczenia nie widać niedouczenia ani wyraźnego przeuczenia, a strata walidacyjna jest poszarpana przy `lr = 1e-2`. Wnioski dotyczą jednego syntetycznego szeregu, 10 punktów startowych i 2 ziaren. Obserwacje o doborze okna i kumulacji błędu przenoszą się na modele mowy (dekodowanie autoregresyjne, kontekst w ramkach, normalizacja i podział bez przecieku). Możliwe kierunki poprawy to mniejszy `lr` lub przycinanie gradientu, więcej ziaren i punktów startowych oraz modele multi-output lub direct, które nie kumulują błędu rekurencji.
